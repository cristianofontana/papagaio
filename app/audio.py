import base64
import logging
import os
import tempfile
from typing import Optional

import requests
from openai import OpenAI

from .settings import settings

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=settings.openai_api_key)


def buscar_midia_por_id(instance: str, message_id: str) -> Optional[str]:
    try:
        url = f"{settings.evolution_server_url}chat/getBase64FromMediaMessage/{instance}"
        headers = {
            "apikey": settings.evolution_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "message": {
                "key": {"id": message_id},
            },
            "convertToMp4": False,
        }
        logger.info("Buscando midia no Evolution API: %s", url)
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code in (200, 201):
            data = response.json()
            base64_audio = data.get("base64")
            if base64_audio:
                logger.info("Base64 encontrado via API Evolution.")
                return base64_audio
            logger.warning("API retornou sem campo base64. Resposta: %s", data)
        else:
            logger.error(
                "Erro ao buscar midia: %s - %s", response.status_code, response.text
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Excecao ao buscar midia: %s", exc)
    return None


def transcrever_audio_base64(audio_base64: str) -> Optional[str]:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        logger.info("Arquivo de audio salvo temporariamente em %s", tmp_path)

        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        os.unlink(tmp_path)
        return transcript.text
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro na transcricao: %s", exc)
        return None
