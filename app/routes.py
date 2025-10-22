import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .audio import buscar_midia_por_id, transcrever_audio_base64
from .conversation import is_group_message, message_buffer
from .database import (
    is_bot_active,
    is_lead_qualified_recently,
    save_message_to_history,
)
from .evolution_api import deletar_mensagem, send_whatsapp_message
from .settings import settings
from .state import bot_active_per_chat, bot_state_lock

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/send-message")
async def send_message_webhook(request: Request):
    data = await request.json()
    numero = data.get("phone")
    mensagem = data.get("text_message", "")
    full_jid = data.get("chat_lid")
    name = data.get("chat_name")

    if not numero or not mensagem:
        return JSONResponse(
            content={"error": "numero e mensagem são obrigatórios"}, status_code=400
        )

    mensagem_lower = mensagem.strip().lower()
    if mensagem_lower == "#off":
        with bot_state_lock:
            bot_active_per_chat[full_jid] = False
        message_buffer.clear_buffer(full_jid)
        return JSONResponse(
            content={"status": f"maintenance OFF for {numero}"}, status_code=200
        )

    if mensagem_lower == "#on":
        with bot_state_lock:
            bot_active_per_chat[full_jid] = True
        return JSONResponse(
            content={"status": f"maintenance ON for {numero}"}, status_code=200
        )

    response = send_whatsapp_message(numero, mensagem)
    if response.status_code in (200, 201):
        return JSONResponse(
            content={"status": "Mensagem enviada", "numero": numero}, status_code=200
        )

    return JSONResponse(
        content={"error": "Falha ao enviar mensagem", "detalhe": response.text},
        status_code=500,
    )


@router.post("/messages-upsert")
async def messages_upsert(request: Request):
    data = await request.json()
    key = data["data"]["key"]
    full_jid = key.get("senderPn") or key.get("remoteJid")
    msg_type = data["data"]["messageType"]
    msg_id = key.get("id")
    from_me_flag = key.get("fromMe")

    sender_number = (
        full_jid.split("@")[0] if full_jid.endswith("@s.whatsapp.net") else full_jid
    )

    if msg_type not in ["audioMessage", "imageMessage"]:
        save_message_to_history(
            full_jid, "bot" if from_me_flag else "user", data["data"]["message"]["conversation"]
        )

    bot_sender = data["sender"]
    bot_number = bot_sender.split("@")[0]
    bot_active_flag = is_bot_active(bot_number)

    if not bot_active_per_chat[full_jid]:
        logger.info(
            "Ignorando mensagem de %s - Bot inativo para este número", sender_number
        )
        return JSONResponse(content={"status": "Bot Inativo"}, status_code=200)

    if not bot_active_flag:
        logger.info(
            "Bot inativado manualmente via aplicativo, %s: %s", bot_number, bot_active_flag
        )
        return JSONResponse(content={"status": "Bot Inativo"}, status_code=200)

    if is_group_message(full_jid):
        group_name = settings.ignored_groups.get(full_jid, "Grupo Desconhecido")
        logger.info("Mensagem de grupo ignorada: %s", group_name)
        return JSONResponse(content={"status": "group_message_ignored"}, status_code=200)

    if settings.authorized_numbers:
        numero = full_jid.split("@")[0]
        if numero not in settings.authorized_numbers:
            logger.info("Número %s não cadastrado na whitelist", numero)
            return JSONResponse(content={"status": "number ignored"}, status_code=200)

    if (
        is_lead_qualified_recently(full_jid, settings.client_id)
        and settings.verificar_lead_qualificado
    ):
        logger.info(
            "Ignorando mensagem de lead qualificado recentemente: %s", sender_number
        )
        return JSONResponse(
            content={"status": "qualified_lead_ignored"}, status_code=200
        )

    with bot_state_lock:
        bot_status = bot_active_per_chat.get(sender_number, True)

    if msg_type == "audioMessage" and not from_me_flag:
        message_data = data["data"]["message"]
        base64_audio = message_data.get("base64")

        if not base64_audio:
            instance = data.get("instance") or "default"
            message_id = key.get("id")
            if instance and message_id:
                base64_audio = buscar_midia_por_id(instance, message_id)
        if base64_audio:
            message = transcrever_audio_base64(base64_audio)
            if not message:
                send_whatsapp_message(
                    full_jid,
                    "Desculpe, estou tendo dificuldades com este audio. Se possivel envie sua mensagem em texto.",
                )
                return JSONResponse(content={"status": "number ignored"}, status_code=200)
        else:
            send_whatsapp_message(
                full_jid,
                "Desculpe, estou tendo dificuldades com este audio. Se possivel envie sua mensagem em texto.",
            )
            return JSONResponse(content={"status": "number ignored"}, status_code=200)
    elif msg_type == "audioMessage" and from_me_flag:
        return JSONResponse(
            content={"status": "Msg de audio enviada pela loja"}, status_code=200
        )
    else:
        message = data["data"]["message"]["conversation"]

    name = data["data"]["pushName"]

    message_lower = message.strip().lower()
    if any(word in message_lower for word in ["#off", "off"]):
        deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[full_jid] = False
        return JSONResponse(
            content={"status": f"maintenance off for {sender_number}"}, status_code=200
        )

    if message_lower == "#on":
        deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[full_jid] = True
        return JSONResponse(
            content={"status": f"maintenance on for {sender_number}"}, status_code=200
        )

    if from_me_flag:
        logger.info("Mensagem enviada pelo bot, ignorando...")
        return JSONResponse(
            content={"status": "message from me ignored"}, status_code=200
        )

    if msg_type == "imageMessage":
        if bot_status:
            send_whatsapp_message(
                full_jid,
                "Desculpe, não consigo abrir imagens. Por favor, envie a mensagem em texto.",
            )
        else:
            logger.info("Mensagem ignorada: imagem recebida e bot está off.")
        return JSONResponse(content={"status": "image ignored"}, status_code=200)

    if not bot_active_per_chat[full_jid]:
        logger.info("Ignorando mensagem de %s - Bot inativo para este número", sender_number)
    else:
        message_buffer.add_message(full_jid, message, name)
        try:
            settings.supabase.table("conversation_states").delete().eq("phone", sender_number).eq(
                "client_id", settings.client_id
            ).execute()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Erro ao resetar reativação: %s", exc)

    return JSONResponse(content={"status": "received"}, status_code=200)
