import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pytz
from langchain_openai import ChatOpenAI

from .database import (
    load_full_conversation_history,
    save_message_to_history,
    update_reminder_step,
)
from .evolution_api import send_whatsapp_message
from .settings import settings

logger = logging.getLogger(__name__)


REACTIVATION_SEQUENCE: List[Tuple[int, str]] = [
    (180, "reengajamento"),
    (360, "oferta_limitada"),
    (1440, "fechamento_urgencia"),
    (0, "stop_reativation"),
]


def should_send_reactivation_llm(phone_number: str, conversation_history: List[Dict[str, Any]]) -> bool:
    try:
        if not conversation_history:
            return False

        history_text = "\n".join(
            [f"{msg['sender']}: {msg['message']}" for msg in conversation_history]
        )

        prompt = f"""
        ## ANALISE DE INTENCAO DE COMPRA

        Analise o historico de conversa abaixo e determine se este cliente
        demonstrou interesse genuino em COMPRAR um celular/produto.

        ## CRITERIOS PARA REATIVACAO (RESPONDER "true"):
        - Cliente perguntou sobre precos, modelos, estoque
        - Demonstrou intencao de compra ("quero comprar", "estou interessado")
        - Pediu orcamento ou condicoes de pagamento
        - Estava comparando precos com outras lojas
        - Perguntou sobre entrada/troca de aparelhos
        - Mostrou interesse especifico em produtos ("iPhone 13", "Samsung S23")

        ## CRITERIOS PARA NAO REATIVAR (RESPONDER "false"):
        - Cliente explicitamente disse que nao quer comprar
        - Apenas duvidas tecnicas ("como faz backup?", "nao consigo conectar")
        - Solicitacoes de conserto/reparo ("quebrou a tela", "nao liga")
        - Reclamacoes sobre produtos ja comprados
        - Informacoes gerais ("que horas fecham?", "onde fica?")
        - Orcamentos para acessorios apenas (capinhas, carregadores)
        - Conversa muito curta sem demonstracao de interesse

        ## HISTORICO DA CONVERSA:
        {history_text}

        ## RESPOSTA:
        Responda APENAS com "true" ou "false" (sem aspas, sem explicacoes).
        """

        chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = chat.invoke(prompt)
        decision = response.content.strip().lower()
        logger.info("LLM decision for %s: %s", phone_number, decision)
        return decision == "true"
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro na analise LLM de reativacao: %s", exc)
        return False


def get_stage_instructions(stage_type: str) -> str:
    if stage_type == "reengajamento":
        return """
        - Voltar a conversa relembrando o interesse inicial e adicionando um novo argumento de valor.
        - Usar gatilhos como: Disponibilidade, Garantia, Promocao Relampago.
        - Nunca fale sobre precos ou valores
        """
    if stage_type == "oferta_limitada":
        return """
        - Verifique no historico se a mensagem de reengajamento foi respondida ou ignorada, use esta informacao no comeco da mensagem.
        - Criar um senso de urgencia e escassez. A oferta deve ser um beneficio, nao um desconto.
        - Ofereca acessorios gratis ou condicoes especiais.
        """
    if stage_type == "fechamento_urgencia":
        return """
        - Explorar a urgencia da oferta (estoque limitado, condicao termina hoje).
        - Incentivar o cliente a responder com um "sim" ou "quero".
        - Reforcar que voce cuidara pessoalmente do atendimento.
        """
    return ""


def generate_reactivation_message(phone_number: str, stage_type: str) -> str | None:
    try:
        messages = load_full_conversation_history(phone_number)
        if not messages:
            return None

        history_str = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in messages])
        user_messages = [msg for msg in messages if msg["sender"] == "user"]
        name = user_messages[0]["message"].split(":")[0] if user_messages else "Cliente"

        client_config = settings.get_client_config()
        nome_do_agent_local = client_config.get("nome_do_agent", "Agente")
        nome_da_loja_local = client_config.get("nome_da_loja", "Loja")

        prompt = f"""
        ## Missao
        Voce e {nome_do_agent_local}, agente virtual da {nome_da_loja_local}. 
        Voce ja teve uma conversa com {name}, agora sua missao sera tentar reativa-lo para que ele possa ser qualificado posteriormente. 

        ## Estagio: {stage_type}
        {get_stage_instructions(stage_type)}

        ## Historico da Conversa
        {history_str}

        ## Regras
        - Comece com um cumprimento personalizado e mais informal. 
        - Nao se apresente mais de uma vez, consulte o historico. 
        - Nao repita informacoes ja dadas.
        - Nao use emojis.
        - Se souber use o nome do cliente 
        - Seja breve e direto, maximo 30 palavras.
        - Nao fale sobre precos.
        - Termine com uma pergunta clara (CTA).

        Gere uma mensagem de reativacao para o estagio {stage_type}.
        """

        chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = chat.invoke(prompt)
        return response.content.strip()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao gerar mensagem de reativacao: %s", exc)
        return None


def send_reactivation_message() -> None:
    tz_sp = pytz.timezone("America/Sao_Paulo")
    while True:
        now_utc = datetime.now(pytz.utc)
        local_now = now_utc.astimezone(tz_sp)
        if local_now.hour < 6 or local_now.hour >= 22:
            time.sleep(300)
            continue

        try:
            result = (
                settings.supabase.table("conversation_states")
                .select("*")
                .lte("next_reminder", now_utc.isoformat())
                .eq("qualified", False)
                .eq("client_id", settings.client_id)
                .neq("stage", 4)
                .execute()
            )
            logger.info(
                "Reativacao - Leads encontrados para reativacao: %s", len(result.data)
            )

            for row in result.data:
                phone = row["phone"]
                step = row.get("reminder_step", 0)
                conversation_history = load_full_conversation_history(phone)

                if not should_send_reactivation_llm(phone, conversation_history):
                    logger.info(
                        "Pulando reativacao para %s - LLM determinou que nao e lead de compra",
                        phone,
                    )
                    settings.supabase.table("conversation_states").update(
                        {"qualified": True}
                    ).eq("phone", phone).eq("client_id", settings.client_id).execute()
                    continue

                if step < len(REACTIVATION_SEQUENCE) - 1:
                    interval, stage_type = REACTIVATION_SEQUENCE[step]
                    message = generate_reactivation_message(phone, stage_type)
                    if message:
                        send_whatsapp_message(phone, message)
                        save_message_to_history(phone, "bot", message)
                        logger.info(
                            "Mensagem de reativacao enviada para %s: %s", phone, message
                        )
                        settings.supabase.table("conversation_states").update(
                            {"stage": 3}
                        ).eq("phone", phone).eq("client_id", settings.client_id).execute()

                    new_step = step + 1
                    if new_step < len(REACTIVATION_SEQUENCE) - 1:
                        next_interval = REACTIVATION_SEQUENCE[new_step][0]
                        update_reminder_step(phone, new_step, next_interval)
                    else:
                        settings.supabase.table("conversation_states").update(
                            {"stage": 4}
                        ).eq("phone", phone).eq("client_id", settings.client_id).execute()
                else:
                    settings.supabase.table("conversation_states").update(
                        {"stage": 4}
                    ).eq("phone", phone).eq("client_id", settings.client_id).execute()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Erro no envio de reativacao: %s", exc)

        time.sleep(300)
