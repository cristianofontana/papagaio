import os
import logging
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# === Configurações ===
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cliente_evo = 'Iclub_Castanhal'

EVOLUTION_API_KEY = os.getenv("EVO_API_KEY")
EVOLUTION_SERVER_URL = 'https://saraevo-evolution-api.jntduz.easypanel.host/message/sendText/{cliente_evo}'

if not EVOLUTION_API_KEY:
    raise ValueError("❌ Variável EVOLUTION_API_KEY não configurada.")
if not EVOLUTION_SERVER_URL:
    raise ValueError("❌ Variável EVOLUTION_SERVER_URL não configurada.")

app = FastAPI()

# === Função para processar mensagem de configuração ===
def processar_mensagem_configuracao(message_data: dict):
    """
    Processa a mensagem de configuração do bot.
    """
    try:
        # Extrai informações relevantes da mensagem
        message_text = message_data.get("message", {}).get("conversation", "").lower()
        sender = message_data.get("key", {}).get("remoteJid", "")  # ID do remetente
        is_group = "g.us" in sender  # Verifica se é mensagem de grupo
        
        logger.info(f"🔍 Analisando mensagem: {message_text[:50]}...")
        
        # Verifica se é a mensagem de configuração em um grupo
        if is_group and "msg_bot" in message_text:
            group_id = sender
            logger.info(f"✅ Mensagem de configuração encontrada no grupo {group_id}")
            
            # Aqui você pode adicionar lógica para processar a configuração
            # Ex: extrair parâmetros, salvar no banco de dados, etc.
            
            return {
                "status": "configuracao_recebida",
                "group_id": group_id,
                "message": message_text
            }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar mensagem: {e}")
        return None

# === Webhook ===
@app.post("/messages-upsert")
async def receber_webhook(request: Request):
    body = await request.json()
    logger.info(f"📩 Webhook recebido: {body.get('data', {}).get('key', {}).get('id')}")

    data = body.get("data", {})
    message = data.get("message", {})
    
    # Processa a mensagem para verificar se é de configuração
    resultado = processar_mensagem_configuracao(data)
    
    if resultado:
        logger.info(f"⚙️ Configuração processada: {resultado}")
        # Aqui você pode adicionar ações adicionais como enviar uma resposta
        # ou registrar a configuração em um banco de dados
        
    return {"status": "ok"}

# === Inicialização ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)