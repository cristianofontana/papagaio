import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pytz
from langchain_core.messages import AIMessage, HumanMessage
from supabase import Client, create_client

from .settings import settings

logger = logging.getLogger(__name__)


def get_client_name_from_db(phone: str) -> Optional[str]:
    try:
        response = (
            settings.supabase.table("client_profiles")
            .select("name")
            .eq("phone", phone)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0].get("name")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao buscar nome do cliente: %s", exc)
        return None


def save_client_name_to_db(phone: str, name: str) -> None:
    try:
        data = {
            "phone": phone,
            "name": name,
            "updated_at": datetime.now(pytz.utc).isoformat(),
        }
        settings.supabase.table("client_profiles").upsert(data).execute()
        logger.info("Nome do cliente %s salvo como %s", phone, name)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao salvar nome do cliente: %s", exc)


def upsert_qualified_lead(phone: str, client_id: str) -> None:
    try:
        now = datetime.now(pytz.utc)
        active_until = now + timedelta(days=10)
        data = {
            "phone": phone,
            "client": client_id,
            "qualified_at": now.isoformat(),
            "active_until": active_until.isoformat(),
        }
        settings.supabase.table("qualified_leads").upsert(data).execute()
        logger.info("Lead %s marcado como qualificado por 10 dias", phone)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao upsert qualified lead: %s", exc)


def is_lead_qualified_recently(phone: str, client_id: str) -> bool:
    try:
        response = (
            settings.supabase.table("qualified_leads")
            .select("active_until")
            .eq("phone", phone)
            .eq("client", client_id)
            .limit(1)
            .execute()
        )
        if response.data:
            active_until_str = response.data[0]["active_until"]
            active_until = datetime.fromisoformat(active_until_str.replace("Z", "+00:00"))
            return datetime.now(pytz.utc) < active_until
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao verificar lead qualificado: %s", exc)
        return False


def inserir_dados_crm(payload: Dict[str, Any]) -> Optional[Any]:
    supabase_url = os.getenv("SUPABASE_CRM_URL")
    supabase_key = os.getenv("SUPABASE_CRM_KEY")
    if not supabase_url or not supabase_key:
        logger.warning("Credenciais do Supabase CRM não configuradas.")
        return None

    supabase_crm: Client = create_client(supabase_url, supabase_key)

    clean_payload = {k: v for k, v in payload.items() if v is not None}
    if not clean_payload:
        logger.error("Payload vazio, não será inserido.")
        return None

    try:
        response = supabase_crm.table("wa_inbound").insert(clean_payload).execute()
        return response.data
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao inserir no Supabase wa_inbound: %s", exc)
        return None


def montar_payload_wa_inbound(payload: Dict[str, Any], foto_url: Optional[str]) -> Dict[str, Any]:
    data = payload.get("data", {})
    key = data.get("key", {})
    message = data.get("message", {})

    tz_sp = pytz.timezone("America/Sao_Paulo")
    dt_sp = datetime.now(tz_sp)
    moment = dt_sp.isoformat()

    return {
        "moment": moment,
        "chat_name": data.get("pushName"),
        "chat_lid": key.get("remoteJid"),
        "from_me": key.get("fromMe"),
        "instance_id": data.get("instanceId"),
        "message_id": key.get("id"),
        "status": data.get("status"),
        "sender_name": data.get("pushName"),
        "type": data.get("messageType"),
        "text_message": message.get("conversation"),
        "phone": key.get("remoteJid").split("@")[0] if key.get("remoteJid") else None,
        "photo": foto_url,
        "is_group": False,
    }


def atualizar_status_lead(phone: str, novo_status: str) -> Optional[Any]:
    supabase_url = os.getenv("SUPABASE_CRM_URL")
    supabase_key = os.getenv("SUPABASE_CRM_KEY")
    if not supabase_url or not supabase_key:
        logger.warning("Credenciais do Supabase CRM não configuradas.")
        return None

    supabase_crm: Client = create_client(supabase_url, supabase_key)

    if not phone or not novo_status:
        logger.error("Telefone ou status vazio, não será atualizado.")
        return None
    try:
        logger.info("Atualizando lead: phone=%s, status=%s", phone, novo_status)
        response = (
            supabase_crm.table("leads").update({"status": novo_status}).eq("phone", phone).execute()
        )
        logger.info("Lead atualizado: %s", response.data)
        return response.data
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao atualizar status do lead: %s", exc)
        return None


def save_conversation_state(
    sender_number: str,
    last_user_message: str,
    last_bot_message: str,
    stage: int,
    last_activity: datetime,
    qualified_status: bool,
    next_interval_minutes: Optional[int] = None,
) -> None:
    data = {
        "phone": sender_number,
        "client_id": settings.client_id,
        "last_user_message": last_user_message,
        "last_bot_message": last_bot_message,
        "stage": stage,
        "last_activity": last_activity.isoformat(),
        "next_reminder": (
            last_activity + timedelta(minutes=next_interval_minutes)
        ).isoformat()
        if next_interval_minutes is not None
        else None,
        "reminder_step": 0,
        "qualified": qualified_status,
    }
    try:
        settings.supabase.table("conversation_states").upsert(data).execute()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao salvar estado no Supabase: %s", exc)


def update_reminder_step(phone: str, step: int, interval_minutes: int) -> None:
    try:
        next_reminder_time = datetime.now(pytz.utc) + timedelta(minutes=interval_minutes)
        settings.supabase.table("conversation_states").update(
            {
                "reminder_step": step,
                "next_reminder": next_reminder_time.isoformat(),
                "qualified": False,
            }
        ).eq("phone", phone).eq("client_id", settings.client_id).execute()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao atualizar passo de lembrete: %s", exc)


def save_message_to_history(
    phone_number: str, sender: str, message: str, conversation_id: Optional[str] = None
) -> None:
    try:
        nome_da_loja = settings.get_client_config().get("nome_da_loja", "Nao Informado")
        data = {
            "phone_number": phone_number,
            "sender": sender,
            "message": message,
            "conversation_id": conversation_id,
            "loja": nome_da_loja,
            "created_at": datetime.now(pytz.utc).isoformat(),
        }
        settings.supabase.table("chat_history").insert(data).execute()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao salvar mensagem no historico: %s", exc)


def is_bot_active(phone: str) -> bool:
    try:
        response = (
            settings.supabase.table("profiles")
            .select("is_active")
            .eq("phone", phone)
            .limit(1)
            .execute()
        )
        logger.info("Status do bot para %s: %s", phone, response.data)
        if response.data:
            return response.data[0].get("is_active", False)
        return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao verificar status do bot: %s", exc)
        return False


def load_conversation_history_from_db(phone_number: str) -> List[Union[HumanMessage, AIMessage]]:
    try:
        expiry_time = datetime.now(pytz.utc) - timedelta(minutes=20)
        response = (
            settings.supabase.table("chat_history")
            .select("*")
            .eq("phone_number", phone_number)
            .gte("created_at", expiry_time.isoformat())
            .order("created_at", desc=False)
            .execute()
        )

        messages: List[Union[HumanMessage, AIMessage]] = []
        for row in response.data:
            if row["sender"] == "user":
                messages.append(HumanMessage(content=row["message"]))
            elif row["sender"] == "bot":
                messages.append(AIMessage(content=row["message"]))
        return messages
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao carregar historico do DB: %s", exc)
        return []


def load_full_conversation_history(phone_number: str) -> List[Dict[str, Any]]:
    try:
        nome_da_loja = settings.get_client_config().get("nome_da_loja", "Nao Informado")
        response = (
            settings.supabase.table("chat_history")
            .select("*")
            .eq("phone_number", phone_number)
            .eq("loja", nome_da_loja)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao carregar historico completo do banco: %s", exc)
        return []


def load_user_stage_from_db(phone: str) -> int:
    try:
        response = (
            settings.supabase.table("conversation_states")
            .select("stage")
            .eq("phone", phone)
            .eq("client_id", settings.client_id)
            .limit(1)
            .execute()
        )
        if response.data:
            stage = response.data[0].get("stage", 0)
            logger.info("Stage carregado do DB para %s: %s", phone, stage)
            return stage
        logger.info("Stage não encontrado para %s, usando padrão 0", phone)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao carregar stage do DB: %s", exc)
        return 0
