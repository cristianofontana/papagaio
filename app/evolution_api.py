import json
import logging
import uuid
from typing import Any, Dict, Optional

import requests

from .settings import settings

logger = logging.getLogger(__name__)


def deletar_mensagem(message_id: str, remote_jid: str, from_me: bool) -> bool:
    headers = {
        "apikey": settings.evolution_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "id": message_id,
        "remoteJid": remote_jid,
        "fromMe": from_me,
    }

    logger.info("Deletando mensagem %s de %s (fromMe=%s)", message_id, remote_jid, from_me)
    url = f"{settings.evolution_server_url}chat/deleteMessageForEveryone/{settings.cliente_evo}"
    response = requests.delete(url, headers=headers, json=payload, timeout=30)
    logger.info("Resposta delecao: %s - %s", response.status_code, response.text)
    return response.ok


def get_text_message_input(recipient: str, text: str) -> str:
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


def send_whatsapp_message(number: str, text: str) -> requests.Response:
    url = f"{settings.evolution_server_url}message/sendText/{settings.cliente_evo}"
    payload = {
        "number": number,
        "text": text,
    }
    headers = {
        "apikey": settings.evolution_api_key,
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    return response


def obter_foto_perfil(remote_jid: str) -> Optional[str]:
    url = f"{settings.evolution_server_url}chat/fetchProfilePictureUrl/{settings.cliente_evo}"
    headers = {
        "Content-Type": "application/json",
        "apikey": settings.evolution_api_key,
    }
    payload = {"number": remote_jid}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("profilePictureUrl")
    except requests.exceptions.RequestException as exc:
        logger.error("Erro na requisicao de foto de perfil: %s", exc)
    except ValueError as exc:
        logger.error("Erro ao decodificar resposta JSON: %s", exc)
    return None


def make_json_response_bot(
    chat_name: str,
    chat_lid: str,
    from_me: bool,
    instance_id: str,
    message_id: Optional[str],
    status: str,
    sender_name: str,
    message_type: str,
    message_content: str,
    phone: str,
) -> Dict[str, Any]:
    from datetime import datetime

    import pytz

    tz_sp = pytz.timezone("America/Sao_Paulo")
    dt_sp = datetime.now(tz_sp)
    moment = dt_sp.isoformat()
    return {
        "moment": moment,
        "chat_name": chat_name,
        "chat_lid": chat_lid,
        "from_me": from_me,
        "instance_id": instance_id,
        "message_id": message_id if message_id else str(uuid.uuid4()),
        "status": status,
        "sender_name": sender_name,
        "type": message_type,
        "text_message": message_content,
        "phone": phone,
        "photo": "",
        "is_group": False,
    }
