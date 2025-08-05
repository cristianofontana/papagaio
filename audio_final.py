import os
import logging
import base64
import tempfile
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

# === Configurações ===
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVOLUTION_API_KEY = os.getenv("EVO_API_KEY")
EVOLUTION_SERVER_URL = 'https://saraevo-evolution-api.jntduz.easypanel.host/'  # Ex.: https://meu-servidor-evolution.com

if not OPENAI_API_KEY:
    raise ValueError("❌ Variável OPENAI_API_KEY não configurada.")
if not EVOLUTION_API_KEY:
    raise ValueError("❌ Variável EVOLUTION_API_KEY não configurada.")
if not EVOLUTION_SERVER_URL:
    raise ValueError("❌ Variável EVOLUTION_SERVER_URL não configurada.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# === Função para buscar mídia via Evolution API ===
def buscar_midia_por_id(instance: str, message_id: str) -> str:
    """
    Busca o áudio em base64 usando o Evolution API.
    """
    try:
        url = f"{EVOLUTION_SERVER_URL}/media/{instance}/{message_id}"
        headers = {"apikey": EVOLUTION_API_KEY}
        logger.info(f"🔄 Buscando mídia no Evolution API: {url}")
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            base64_audio = data.get("media", {}).get("base64")
            if base64_audio:
                logger.info("✅ Base64 encontrado via API Evolution.")
                return base64_audio
            else:
                logger.warning("⚠️ API retornou, mas sem campo base64.")
        else:
            logger.error(f"❌ Erro ao buscar mídia: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Exceção ao buscar mídia: {e}")
    return None

# === Função para transcrever áudio ===
def transcrever_audio_base64(audio_base64: str) -> str:
    """
    Transcreve áudio a partir de um base64 usando Whisper.
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        logger.info(f"📁 Arquivo de áudio salvo temporariamente em {tmp_path}")

        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",  # ou whisper-1
                file=audio_file
            )

        return transcript.text
    except Exception as e:
        logger.error(f"❌ Erro na transcrição: {e}")
        return None

# === Webhook ===
@app.post("/messages-upsert")
async def receber_webhook(request: Request):
    body = await request.json()
    logger.info(f"📩 Webhook recebido: {body}")

    data = body.get("data", {})
    message = data.get("message", {})

    # 1️⃣ Tenta pegar base64 do próprio webhook
    base64_audio = message.get("base64")

    # 2️⃣ Se não vier no payload, busca via Evolution API
    if not base64_audio:
        logger.warning("⚠️ Webhook sem base64, buscando via API Evolution...")
        instance = body.get("instance") or data.get("instance") or "default"
        message_id = data.get("key", {}).get("id")
        if instance and message_id:
            base64_audio = buscar_midia_por_id(instance, message_id)
        else:
            logger.error("❌ Não foi possível obter instance ou message_id para buscar mídia.")

    # 3️⃣ Se conseguiu o áudio, transcreve
    if base64_audio:
        logger.info("🎙️ Iniciando transcrição...")
        texto_transcrito = transcrever_audio_base64(base64_audio)
        if texto_transcrito:
            logger.info(f"📝 Transcrição: {texto_transcrito}")
        else:
            logger.warning("⚠️ Não foi possível transcrever o áudio.")
    else:
        logger.warning("⚠️ Nenhum áudio disponível para transcrição.")

    return {"status": "ok"}

# === Inicialização ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
