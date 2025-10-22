import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pytz
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .database import (
    atualizar_status_lead,
    load_conversation_history_from_db,
    load_user_stage_from_db,
    save_conversation_state,
    save_message_to_history,
    upsert_qualified_lead,
)
from .evolution_api import deletar_mensagem, send_whatsapp_message
from .llm_service import (
    detect_intent,
    get_custom_prompt,
    get_info,
    get_reativacao_prompt,
    is_qualification_message,
    is_stop_request,
    make_answer,
)
from .reactivation import REACTIVATION_SEQUENCE
from .settings import settings
from .state import bot_active_per_chat, bot_state_lock, conversation_history

logger = logging.getLogger(__name__)


class MessageBuffer:
    def __init__(self, timeout: int = 20, handler=None) -> None:
        self.timeout = timeout
        self.handler = handler
        self.buffers: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def set_handler(self, handler) -> None:
        self.handler = handler

    def add_message(self, user_id: str, message_content: str, name: str) -> None:
        with self.lock:
            if user_id not in self.buffers:
                self.buffers[user_id] = {"messages": [], "name": name, "timer": None}

            if self.buffers[user_id]["timer"]:
                self.buffers[user_id]["timer"].cancel()

            self.buffers[user_id]["messages"].append(message_content)

            self.buffers[user_id]["timer"] = threading.Timer(
                self.timeout, self._process_buffer, [user_id]
            )
            self.buffers[user_id]["timer"].start()

    def _process_buffer(self, user_id: str) -> None:
        with self.lock:
            if user_id not in self.buffers:
                return
            buffer_data = self.buffers[user_id]
            messages = buffer_data["messages"]
            name = buffer_data["name"]
            del self.buffers[user_id]

        concatenated_message = " ".join(messages).strip()
        if self.handler:
            self.handler(user_id, concatenated_message, name)

    def clear_buffer(self, user_id: str) -> None:
        with self.lock:
            if user_id in self.buffers:
                if self.buffers[user_id]["timer"]:
                    self.buffers[user_id]["timer"].cancel()
                del self.buffers[user_id]


message_buffer = MessageBuffer(timeout=10)


def is_group_message(remote_jid: str) -> bool:
    return "@g.us" in remote_jid or (
        "-" in remote_jid.split("@")[0] if "@" in remote_jid else "-" in remote_jid
    )


def cleanup_expired_histories() -> None:
    while True:
        current_time = time.time()
        expired_keys = []

        for user_id, data in conversation_history.items():
            elapsed = current_time - data["last_activity"]
            if elapsed > settings.history_expiration_minutes * 60:
                expired_keys.append(user_id)

        for key in expired_keys:
            del conversation_history[key]
            logger.info("Removido historico expirado para: %s", key)

        time.sleep(60)


def process_user_message(sender_number: str, message: str, name: str) -> None:
    qualified_status = False
    stage_from_db = load_user_stage_from_db(sender_number)
    if stage_from_db == 4:
        logger.info(
            "Usuario %s esta no estagio 4 (nao reativar). Ignorando mensagem.",
            sender_number,
        )
        return
    if stage_from_db == 3:
        logger.info("Usuario %s respondeu em reativacao; removendo do funil.", sender_number)
        settings.supabase.table("conversation_states").update(
            {
                "qualified": True,
                "stage": 0,
                "reminder_step": 0,
                "next_reminder": None,
            }
        ).eq("phone", sender_number).eq("client_id", settings.client_id).execute()

        if is_stop_request(message):
            logger.info(
                "Usuario %s solicitou parar reativacao. Atualizando estado para 4.",
                sender_number,
            )
            settings.supabase.table("conversation_states").update({"stage": 4}).eq(
                "phone", sender_number
            ).eq("client_id", settings.client_id).execute()

            send_whatsapp_message(
                sender_number,
                "Ok! Sem problema. Conte conosco em uma proxima oportunidade.",
            )
            return
        logger.info(
            "Usuario %s no estagio 3, mas nao e stop request. Continuando conversa.",
            sender_number,
        )

    if sender_number not in conversation_history:
        conversation_id = str(uuid.uuid4())
        history_from_db = load_conversation_history_from_db(sender_number)
        if history_from_db:
            conversation_history[sender_number] = {
                "messages": history_from_db,
                "conversation_id": conversation_id,
                "stage": stage_from_db,
                "intent": detect_intent(message),
                "bant": {"budget": None, "authority": None, "need": None, "timing": None},
                "last_activity": time.time(),
            }
    else:
        conversation_id = conversation_history[sender_number].get(
            "conversation_id", str(uuid.uuid4())
        )

    current_intent = detect_intent(message)

    if sender_number not in conversation_history:
        conversation_history[sender_number] = {
            "messages": [],
            "conversation_id": conversation_id,
            "stage": stage_from_db if stage_from_db is not None else 0,
            "intent": current_intent,
            "bant": {"budget": None, "authority": None, "need": None, "timing": None},
            "last_activity": time.time(),
        }
    else:
        conversation_history[sender_number]["last_activity"] = time.time()

    conversation_history[sender_number]["messages"].append(HumanMessage(content=message))

    history = conversation_history[sender_number]["messages"][-20:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])

    if stage_from_db != 3:
        prompt_text = get_custom_prompt(message, history_str, current_intent, name)
    else:
        qualified_status = True
        settings.supabase.table("conversation_states").update(
            {"qualified": qualified_status}
        ).eq("phone", sender_number).eq("client_id", settings.client_id).execute()
        prompt_text = get_reativacao_prompt(history_str, message)

    prompt_messages: List[Any] = [SystemMessage(content=prompt_text)] + history
    response: AIMessage = make_answer(prompt_messages)

    conversation_history[sender_number]["messages"].append(response)
    response_content = response.content

    save_message_to_history(sender_number, "bot", response_content, conversation_id)
    logger.info("Resposta para o usuario %s: %s", sender_number, response_content)

    suffix = "@s.whatsapp.net"
    numero = sender_number[:-len(suffix)] if sender_number.endswith(suffix) else sender_number

    if "orcamento" in response_content.lower():
        conversation_history[sender_number]["stage"] = 2
    elif is_qualification_message(response_content):
        logger.info("Qualificacao detectada para %s", sender_number)
        infos_raw = get_info(history_str)
        conversation_history[sender_number]["stage"] = 3
        logger.info(
            "Lead qualificado: %s - Intent: %s",
            sender_number,
            conversation_history[sender_number]["intent"],
        )

        if isinstance(infos_raw, str):
            try:
                infos = json.loads(infos_raw)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Erro ao converter infos para dict: %s", exc)
                infos = {}
        else:
            infos = infos_raw or {}

        demanda = infos.get("DEMANDA", "Nao Informado")
        interesse = infos.get("INTERESSE", "Produto nao especificado")
        budget = infos.get("BUDGET/FORMA PAGAMENTO", "Valor nao especificado")
        urgency = infos.get("URGENCIA", "Nao especificado")
        pesquisando = infos.get("ESTA-PESQUISANDO", "Nao Informado")

        msg_qualificacao = f"""
Lead Qualificado:
Nome: {name},
Telefone: {numero},
Interesse: {interesse},
Budget: {budget},
Urgencia: {urgency},
Esta-Pesquisando: {pesquisando},
Link: https://wa.me/{numero}
        """

        logger.info("Mensagem de qualificacao: %s", msg_qualificacao)
        group_id = settings.get_client_config().get("group_id", "")
        send_whatsapp_message(group_id, msg_qualificacao)
        upsert_qualified_lead(sender_number, settings.client_id)
        atualizar_status_lead(numero, "hot")
        logger.info("Lead %s atualizado para status 'hot' no CRM.", numero)

    if response_content.strip() != "#no-answer":
        send_whatsapp_message(sender_number, response_content)
        current_stage = conversation_history[sender_number]["stage"]
        next_interval = REACTIVATION_SEQUENCE[0][0] if REACTIVATION_SEQUENCE else None
        save_conversation_state(
            sender_number=sender_number,
            last_user_message=message,
            last_bot_message=response_content,
            stage=current_stage,
            last_activity=datetime.now(pytz.utc),
            qualified_status=qualified_status,
            next_interval_minutes=next_interval,
        )


message_buffer.set_handler(process_user_message)
