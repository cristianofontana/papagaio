from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi import FastAPI, HTTPException, Request, Query
import json
import aiohttp
import asyncio
import re
import uuid

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client 
import os 
from langchain.schema import (
    SystemMessage
    ,HumanMessage
    ,AIMessage
)
# Substituído Groq por OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging
import requests 

from supabase import create_client, Client
import os
import time
from datetime import datetime, timedelta
import pytz
import threading
from threading import Lock
from typing import Dict, Any, List, Optional, Union

from fastapi.responses import JSONResponse
#import spacy
#from spacy.matcher import Matcher
from collections import defaultdict

import base64
import hashlib
from Crypto.Cipher import AES
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import tempfile
import openai


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVOLUTION_API_KEY = os.getenv("EVO_API_KEY")
EVOLUTION_SERVER_URL = 'https://saraevo-evolution-api.jntduz.easypanel.host/'  # Ex.: https://meu-servidor-evolution.com

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

bot_active_per_chat = defaultdict(lambda: True)  # Estado do bot por número do cliente
bot_state_lock = Lock()  # Lock para sincronização de estado

# ================== API FastAPI ================== #
app = FastAPI(title="WhatsApp Transcription API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#nlp = spacy.load('pt_core_news_sm')

#matcher = Matcher(nlp.vocab)

patterns = [
    [{"LOWER": {"IN": ["passar", "encaminhar", "transferir"]}}, {"LOWER": "para"}, {"LOWER": {"IN": ["gerente", "vendedor", "humano"]}}],
    [{"LOWER": "chamei"}, {"LOWER": "um"}, {"LOWER": {"IN": ["vendedor", "especialista"]}}],
    [{"LOWER": "finalizar"}, {"LOWER": "atendimento"}],
    [{"LOWER": "encaminhamento"}, {"LOWER": "para"}, {"LOWER": "humanos"}]
]

IGNORED_GROUPS = {
    "120363420079107628@g.us": "Grupo Admin",
    # adicione outros grupos se quiser
}

def is_group_message(remote_jid: str) -> bool:
    return "@g.us" in remote_jid or (
        "-" in remote_jid.split("@")[0] if "@" in remote_jid else "-" in remote_jid
    )

################################# CONFIG PERSONALIZADA CLIENTE #################################
def load_client_config(client_id: str) -> dict:
    try:
        response = supabase.table("client_config") \
            .select("*") \
            .eq("client_id", client_id) \
            .limit(1) \
            .execute()
        
        if response.data:
            config = response.data[0]

            prompt_id = config.get('prompt_id')
            
            prompt_text = None
            if prompt_id:
                prompt_response = supabase.table("prompts") \
                    .select("prompt_text") \
                    .eq("id", prompt_id) \
                    .limit(1) \
                    .execute()
                if prompt_response.data:
                    prompt_text = prompt_response.data[0].get('prompt_text')
                    
            return {
                'nome_do_agent': config.get('nome_do_agent', 'Agente'),
                'nome_da_loja': config.get('nome_da_loja', 'Loja'),
                'horario_atendimento': config.get('horario_atendimento', 'Seg a Sex 9:00-18:00'),
                'endereco_da_loja': config.get('endereco_da_loja', 'Endereco nao especificado'),
                'categorias_atendidas': config.get('categorias_atendidas', 'Produtos em geral'),
                'lugares_que_faz_entrega': config.get('lugares_que_faz_entrega', ''),
                'forma_pagamento_iphone': config.get('forma_pagamento_iphone', 'à vista ou parcelado'),
                'forma_pagamento_android': config.get('forma_pagamento_android', 'à vista ou parcelado'),
                'collection_name': config.get('collection_name', 'default_collection'),
                'authorized_numbers': config.get('authorized_numbers', []),
                'group_id': config.get('id_grupo_cliente', ''),
                # Novos campos
                'lista_iphone': config.get('lista_iphone', 'iPhone 11 até iPhone 16 Pro Max'),
                'lista_android': config.get('lista_android', 'Xiaomi, Redmi, Poco'),
                'msg_abertura': config.get('msg_abertura', ''),
                'msg_fechamento': config.get('msg_fechamento', ''),
                'prompt_text': prompt_text  # Novo campo com o texto do prompt
            }
        else:
            logger.error(f"Configuração não encontrada para cliente: {client_id}")
            return {}
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {str(e)}")
        return {}

# Carregar configurações do Supabase
CLIENT_ID = 'five_store'  # ID do cliente no Supabase
verificar_lead_qualificado = True  # Ativar verificação de lead qualificado
HISTORY_EXPIRATION_MINUTES = 180 # 3 horas de buffer das mensagens

def get_client_config() -> dict:
    client_config = load_client_config(CLIENT_ID)
    return client_config

client_config = get_client_config()
# Usar valores padrão se a configuração não for encontrada
nome_do_agent = client_config.get('nome_do_agent', 'Eduardo')
nome_da_loja = client_config.get('nome_da_loja', 'Não Informado')
horario_atendimento = client_config.get('horario_atendimento', 'Não Informado')
endereco_da_loja = client_config.get('endereco_da_loja', 'Não Informado')
categorias_atendidas = client_config.get('categorias_atendidas', 'Iphone e Acessórios')
lugares_que_faz_entrega = client_config.get('lugares_que_faz_entrega', '')
forma_pagamento_iphone = client_config.get('forma_pagamento_iphone', 'à vista e cartão em até 21X')
forma_pagamento_android = client_config.get('forma_pagamento_android', 'à vista, no cartão em até 21X ou boleto')
COLLECTION_NAME = client_config.get('collection_name', 'Não Informado')
cliente_evo = 'Papagaio_dev'  #COLLECTION_NAME
AUTHORIZED_NUMBERS = client_config.get('authorized_numbers', [''])

id_grupo_cliente =  client_config.get('group_id', 'Não Informado')#'120363420079107628@g.us' #120363420079107628@g.us id grupo papagaio 

#for pattern in patterns:
#    matcher.add("TRANSFER_PATTERNS", [pattern])


# Adicione esta classe antes da definição do app

class MessageBuffer:
    def __init__(self, timeout=20): ### 12 horas 
        self.timeout = timeout
        self.buffers: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def add_message(self, user_id: str, message_content: str, name: str):
        with self.lock:
            if user_id not in self.buffers:
                self.buffers[user_id] = {
                    'messages': [],
                    'name': name,
                    'timer': None
                }
            
            # Cancela o timer anterior se existir
            if self.buffers[user_id]['timer']:
                self.buffers[user_id]['timer'].cancel()
            
            self.buffers[user_id]['messages'].append(message_content)
            
            # Agenda novo timer
            self.buffers[user_id]['timer'] = threading.Timer(
                self.timeout, 
                self._process_buffer, 
                [user_id]
            )
            self.buffers[user_id]['timer'].start()
    
    def _process_buffer(self, user_id: str):
        with self.lock:
            if user_id not in self.buffers:
                return
                
            buffer_data = self.buffers[user_id]
            messages = buffer_data['messages']
            name = buffer_data['name']
            del self.buffers[user_id]  # Remove o buffer processado
            
        # Concatena as mensagens
        concatenated_message = " ".join(messages).strip()
        
        # Chama a função de processamento principal
        process_user_message(user_id, concatenated_message, name)
        
    # Novo método para limpar o buffer de um usuário
    def clear_buffer(self, user_id: str):
        with self.lock:
            if user_id in self.buffers:
                if self.buffers[user_id]['timer']:
                    self.buffers[user_id]['timer'].cancel()
                del self.buffers[user_id]

########################################################################## INICIO RAG SYSTEM #####################################################################################
from qdrant_client import QdrantClient

# Configurações do Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"  # Modelo usado para embeddings

# Inicializar cliente Qdrant
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def query_qdrant(query: str, k: int = 10) -> list:
    """Consulta o Qdrant e retorna os documentos mais relevantes"""
    logging.info(f"Consultando Qdrant com a query: {query}")

    try:
        # Gerar embedding da pergunta
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        query_embedding = embeddings.embed_query(query)
        
        # Fazer a consulta
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        # Processar resultados - corrigido para estrutura aninhada
        context = []
        for result in results:
            payload = result.payload
            metadata = payload.get('metadata', {})
            
            # Tratamento para campos que podem estar ausentes
            item = metadata.get('Item', '')
            descricao = payload.get('content', '')  # Descrição está no payload principal
            
            # Se a descrição estiver vazia, tente criar uma básica
            if not descricao and item:
                descricao = f"Produto: {item}"
                
            context.append({
                'content': descricao,
                'item': item,
                'aceita_como_entrada': metadata.get('aceita_como_entreda', ''),
                'preco_novo': metadata.get('preco_novo', ''),
                'preco_semi_novo': metadata.get('preco_semi_novo', '')
            })
        logging.info(f"Resultados encontrados: {context}")
        return context
        
    except Exception as e:
        logging.error(f"Erro ao consultar Qdrant: {str(e)}")
        return []

def is_technical_question(text: str) -> bool:
    """Determina se a pergunta requer consulta ao Qdrant"""
    technical_keywords = [
        'especificação', 'tela', 'câmera', 'processador', 'memória', 'armazenamento', 
        'bateria', 'carregamento', 'ios', 'resolução', 'peso', 'dimensão', 'tamanho',
        'modelo', 'iphone', 'comparar', 'diferença', 'qual é o', 'quanto custa', 'quais são os modelos', 'quais modelos','voces tem','vcis tem',
        'entrada', 'troca', 'aceita troca', 'aceita como entrada',
        # Novas palavras-chave específicas
        'mais novo', 'novo ou usado', 'mais memoria', 'modelos de celular', 
        'acessorios', 'não sejam iphone', 'outros modelos', 'qual iphone', 'tem estoque', 'estoque'
        'disponível', 'características', 'especificações'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in technical_keywords)

########################################################################## Inicio config horario de inatividade #####################################################################################
def no_horario_inatividade():
    """
    Verifica se está no horário de inatividade (segunda a sexta, 8:00-18:00)
    Retorna True se estivermos no horário de inatividade, False caso contrário
    """
    try:
        # Obter horário atual (fuso horário de São Paulo)
        tz = pytz.timezone('America/Sao_Paulo')
        agora = datetime.now(tz)
        hora_atual = agora.time()
        dia_semana = agora.weekday()  # 0=segunda, 6=domingo
        
        # Verificar se é dia útil (segunda a sexta)
        ##if dia_semana < 5:  # 0-4 = segunda a sexta
        # Verificar se está entre 8:00 e 18:00
        #if dia_semana < 5:  # 0-4 = segunda a sexta
        inicio = datetime.strptime('08:00', '%H:%M').time()
        fim = datetime.strptime('15:00', '%H:%M').time()
        
        if inicio <= hora_atual <= fim:
            return True
                
        return False
        
    except Exception as e:
        logger.error(f"Erro ao verificar horário de inatividade: {str(e)}")
        return False  # Em caso de erro, considera que não está no horário de inatividade


########################################################################## Inicio Delete Message #######################################################################################

# === Função para deletar mensagem ===
def deletar_mensagem(message_id: str, remote_jid: str, from_me: bool):
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "id": message_id,
        "remoteJid": remote_jid,
        "fromMe": from_me
    }

    logger.info(f"🗑️ Deletando mensagem {message_id} de {remote_jid} (fromMe={from_me})")

    url_evo = f'{EVOLUTION_SERVER_URL}chat/deleteMessageForEveryone/{cliente_evo}'
    #https://saraevo-evolution-api.jntduz.easypanel.host/chat/deleteMessageForEveryone/ReconvertAI
    #https://saraevo-evolution-api.jntduz.easypanel.host/chat/deleteMessageForEveryone/papagaio
    logging.info(f'URL_EVO -> {url_evo}')

    resp = requests.delete(url_evo, headers=headers, json=payload)
    logger.info(f"Resposta deleção: {resp.status_code} - {resp.text}")
    return resp.ok


############################################################# INICIO SUPABASE ##########################################################################################

## webhook CRM 
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou especifique seu domínio, ex: ["https://preview--post-comment-insight.lovable.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/send-message")
async def send_message_webhook(request: Request):
    """f
    Recebe o número do cliente e mensagem via POST e envia mensagem via WhatsApp.
    Exemplo de payload:
    {
        "numero": "554196137682",
        "mensagem": "Olá, esta é uma mensagem automática!"
    }
    """
    
    data = await request.json()
    numero = data.get("phone")
    mensagem = data.get("text_message")
    full_jid = data.get("chat_lid")  # Ex.:
    name = data.get("chat_name")

    if mensagem.strip().lower() == "#off":
        #deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[full_jid] = False
            
        #json_responde_bot = make_json_response_bot(chatName=name, chatLid=full_jid, fromMe=True, instanceId='', messageId='', status='SENT', senderName='CRM', messageType='text', messageContent='#off', phone=numero)
        #inserir_dados_crm(json_responde_bot)
        
        message_buffer.clear_buffer(full_jid)  # Limpa o buffer para este usuário
        return JSONResponse(content={"status": f"maintenance OFF for {numero}"}, status_code=200)

    elif mensagem.strip().lower() == "#on":
        #deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[full_jid] = True 
            
        #json_responde_bot = make_json_response_bot(chatName=name, chatLid=full_jid, fromMe=True, instanceId='', messageId='', status='SENT', senderName='CRM', messageType='text', messageContent='#on', phone=numero)
        #inserir_dados_crm(json_responde_bot)
        
        return JSONResponse(content={"status": f"maintenance ON for {numero}"}, status_code=200)

    if not numero or not mensagem:
        return JSONResponse(content={"error": "numero e mensagem são obrigatórios"}, status_code=400)

    # Envia a mensagem
    resp = send_whatsapp_message(numero, mensagem)
    if resp.status_code in [200, 201]:
        #json_responde_bot = make_json_response_bot(chatName=name, chatLid=full_jid, fromMe=True, instanceId='', messageId='', status='SENT', senderName='CRM', messageType='text', messageContent=mensagem, phone=numero)

        #inserir_dados_crm(json_responde_bot)
        return JSONResponse(content={"status": "Mensagem enviada", "numero": numero}, status_code=200)
    else:
        return JSONResponse(content={"error": "Falha ao enviar mensagem", "detalhe": resp.text}, status_code=500)

#### Trecho para obter e salvar o nome do cliente 

def get_client_name_from_db(phone: str) -> Optional[str]:
    """Busca o nome do cliente no banco de dados pelo número de telefone."""
    try:
        response = supabase.table("client_profiles") \
            .select("name") \
            .eq("phone", phone) \
            .limit(1) \
            .execute()
        if response.data:
            return response.data[0].get('name')
        return None
    except Exception as e:
        logger.error(f"Erro ao buscar nome do cliente: {str(e)}")
        return None

def save_client_name_to_db(phone: str, name: str):
    """Salva ou atualiza o nome do cliente no banco de dados."""
    try:
        data = {
            "phone": phone,
            "name": name,
            "updated_at": datetime.now(pytz.utc).isoformat()
        }
        supabase.table("client_profiles").upsert(data).execute()
        logger.info(f"Nome do cliente {phone} salvo como {name}")
    except Exception as e:
        logger.error(f"Erro ao salvar nome do cliente: {str(e)}")

#### Leads qualificados 

def upsert_qualified_lead(phone: str, client_id: str):
    """Insere ou atualiza um lead qualificado na tabela"""
    try:
        now = datetime.now(pytz.utc)
        active_until = now + timedelta(days=10)
        
        data = {
            "phone": phone,
            "client": client_id,
            "qualified_at": now.isoformat(),
            "active_until": active_until.isoformat()
        }
        
        supabase.table("qualified_leads").upsert(data).execute()
        logger.info(f"Lead {phone} marcado como qualificado por 10 dias")
    except Exception as e:
        logger.error(f"Erro ao upsert qualified lead: {str(e)}")

def is_lead_qualified_recently(phone: str, CLIENT_ID: str) -> bool:
    """Verifica se o lead foi qualificado nos últimos 10 dias"""
    try:
        response = supabase.table("qualified_leads") \
            .select("active_until") \
            .eq("phone", phone) \
            .eq("client", CLIENT_ID) \
            .limit(1) \
            .execute()
            
        if response.data:
            active_until_str = response.data[0]['active_until']
            active_until = datetime.fromisoformat(active_until_str.replace('Z', '+00:00'))
            return datetime.now(pytz.utc) < active_until
        return False
    except Exception as e:
        logger.error(f"Erro ao verificar lead qualificado: {str(e)}")
        return False

#### Inserção dados CRM 
def inserir_dados_crm(payload):
    #logger.info(f"Payload para wa_inbound: {payload}")
    
    SUPABASE_URL = os.getenv("SUPABASE_CRM_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_CRM_KEY")
    supabase_crm: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    
    # Remover campos None
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    if not clean_payload:
        logger.error("Payload está vazio, não será inserido.")
        return None
    try:
        response = supabase_crm.table("wa_inbound").insert(clean_payload).execute()
        return response.data
    except Exception as e:
        logger.error(f"Erro ao inserir no Supabase wa_inbound: {e}")
        return None

def make_json_response_bot(chatName, chatLid, fromMe, instanceId, messageId, status, senderName, messageType, messageContent, phone):
    tz_sp = pytz.timezone('America/Sao_Paulo')
    dt_sp = datetime.now(tz_sp)
    moment = dt_sp.isoformat()
    return {
        "moment": moment,
        "chat_name": chatName,
        "chat_lid": chatLid,
        "from_me": fromMe,
        "instance_id": instanceId,
        "message_id": messageId if messageId else str(uuid.uuid4()),
        "status": status,
        "sender_name": senderName,
        "type": messageType,
        "text_message": messageContent,
        "phone": phone,
        "photo": '',
        'is_group': False
    }
    
def obter_foto_perfil(server_url, instance_name, api_key, remote_jid):
    """
    Obtém a URL da foto de perfil do usuário usando a Evolution API
    
    Args:
        server_url (str): URL do servidor da Evolution API
        instance_name (str): Nome da instância
        api_key (str): Chave de API para autenticação
        remote_jid (str): ID do usuário (no formato 5511999999999@s.whatsapp.net)
    
    Returns:
        str: URL da foto de perfil ou None se não encontrada
    """
    url = f"{server_url}chat/fetchProfilePictureUrl/{instance_name}"
    
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key
    }
    
    payload = {
        "number": remote_jid
    }
    #logging.info(f"Obtendo foto de perfil para {remote_jid} na URL {url}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return data.get("profilePictureUrl")
        
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None
    except ValueError as e:
        print(f"Erro ao decodificar resposta JSON: {e}")
        return None

def montar_payload_wa_inbound(payload, foto_url):
    """
    Monta o payload para inserir na tabela wa_inbound a partir do payload recebido e da foto de perfil

    Args:
        payload (dict): Payload recebido do webhook
        foto_url (str): URL da foto de perfil do usuário

    Returns:
        dict: Payload formatado para wa_inbound
    """
    data = payload.get("data", {})
    key = data.get("key", {})
    message = data.get("message", {})

    tz_sp = pytz.timezone('America/Sao_Paulo')
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
        'is_group': False
    }
    
def atualizar_status_lead(phone: str, novo_status: str):
    """
    Atualiza o status de um lead na tabela leads pelo número de telefone.
    """
    SUPABASE_URL = os.getenv("SUPABASE_CRM_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_CRM_KEY")
    supabase_crm: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
    
    if not phone or not novo_status:
        logger.error("Telefone ou status vazio, não será atualizado.")
        return None
    try:
        logger.info(f"Atualizando lead: phone={phone}, status={novo_status}")
        response = supabase_crm.table("leads") \
            .update({"status": novo_status}) \
            .eq("phone", phone) \
            .execute()
        logger.info(f"Lead atualizado: {response.data}")
        return response.data
    except Exception as e:
        logger.error(f"Erro ao atualizar status do lead: {e}")
        return None

# Sequência de reativação (tempo em minutos, mensagem)

def save_conversation_state(sender_number: str, last_user_message: str, 
                           last_bot_message: str, stage: int, last_activity: datetime, qualified_status: bool):
    qualified = stage >= 3

    data = {
        "phone": sender_number,
        "client_id": CLIENT_ID,  # Add client_id
        "last_user_message": last_user_message,
        "last_bot_message": last_bot_message,
        "stage": stage,
        "last_activity": last_activity.isoformat(),
        "next_reminder": (last_activity + timedelta(minutes=REACTIVATION_SEQUENCE[0][0])).isoformat(),
        "reminder_step": 0,
        "qualified": qualified_status
    }
    
    try:
        supabase.table("conversation_states").upsert(data).execute()
    except Exception as e:
        logger.error(f"Erro ao salvar estado no Supabase: {str(e)}")

def update_reminder_step(phone: str, step: int):
    try:
        interval, _ = REACTIVATION_SEQUENCE[step]
        next_reminder_time = datetime.now(pytz.utc) + timedelta(minutes=interval)
        supabase.table("conversation_states").update({
            "reminder_step": step,
            "next_reminder": next_reminder_time.isoformat(),
            "qualified": False
        }).eq("phone", phone).eq("client_id", CLIENT_ID).execute()  # Add client_id filter
    except Exception as e:
        logger.error(f"Erro ao atualizar passo de lembrete: {str(e)}")

# Função para enviar mensagens de reativação
def send_reactivation_message():
    while True:
        try:
            now = datetime.now(pytz.utc)
            # Add client_id filter to the query
            result = supabase.table("conversation_states") \
                .select("*") \
                .lte("next_reminder", now.isoformat()) \
                .eq("qualified", False) \
                .eq("client_id", CLIENT_ID) \
                .neq("stage", 4) \
                .execute()
            logging.info(f"Reativação - Leads encontrados para reativação: {len(result.data)}")
                
            for row in result.data:
                phone = row["phone"]
                step = row["reminder_step"]
                
                conversation_history = load_full_conversation_history(phone, nome_da_loja)
                
                if not should_send_reactivation_llm(phone, conversation_history):
                    logger.info(f"Pulando reativação para {phone} - LLM determinou que não é lead de compra")
                    # Marcar como qualificado=false para não reprocessar
                    supabase.table("conversation_states").update({
                        "qualified": True  # Marca como "não qualificado para reativação"
                    }).eq("phone", phone).eq("client_id", CLIENT_ID).execute()
                    continue
                
                if step < len(REACTIVATION_SEQUENCE)-1:
                    interval, stage_type = REACTIVATION_SEQUENCE[step]
                    message = generate_reactivation_message(phone, stage_type)
                    if message:
                        send_whatsapp_message(phone, message)
                        save_message_to_history(phone, "bot", message)
                        
                        logger.info(f"Mensagem de reativação enviada para {phone}: {message}")
                        
                        ## ATUALIZA STATE para 3 REATIVACAO 
                        supabase.table("conversation_states") \
                        .update({"stage": 3}) \
                        .eq("phone", phone) \
                        .eq("client_id", CLIENT_ID) \
                        .execute()
                    
                    new_step = step + 1
                    if new_step < len(REACTIVATION_SEQUENCE)-1:
                        update_reminder_step(phone, new_step)
                    elif new_step >= 3:
                        # Último lembrete enviado, deletar da tabela
                        #supabase.table("conversation_states").delete().eq("phone", phone).eq("client_id", CLIENT_ID).execute()
                        supabase.table("conversation_states") \
                        .update({"stage": 4}) \
                        .eq("phone", phone) \
                        .eq("client_id", CLIENT_ID) \
                        .execute()
                    else:
                        supabase.table("conversation_states") \
                        .update({"stage": 4}) \
                        .eq("phone", phone) \
                        .eq("client_id", CLIENT_ID) \
                        .execute()
                        # Add client_id filter to delete
                        #supabase.table("conversation_states").delete().eq("phone", phone).eq("client_id", CLIENT_ID).execute()
        
        except Exception as e:
            logger.error(f"Erro no envio de reativação: {str(e)}")
        
        time.sleep(60)

def save_message_to_history(phone_number: str, sender: str, message: str, conversation_id: str = None):
    """
    Salva uma mensagem no histórico do Supabase
    """
    try:
        data = {
            "phone_number": phone_number,
            "sender": sender,
            "message": message,
            "conversation_id": conversation_id,
            "loja": nome_da_loja,
            "created_at": datetime.now(pytz.utc).isoformat()
        }
        supabase.table("chat_history").insert(data).execute()
    except Exception as e:
        logger.error(f"Erro ao salvar mensagem no histórico: {str(e)}")

def is_bot_active(phone: str) -> bool:
    """Verifica se o bot está ativo para este número na tabela profiles"""
    try:
        response = supabase.table("profiles") \
            .select("is_active") \
            .eq("phone", phone) \
            .limit(1) \
            .execute()
        
        logging.info(f"Status do bot para {phone}: {response.data}")
        if response.data:
            return response.data[0].get('is_active', False)
        return False
    except Exception as e:
        logger.error(f"Erro ao verificar status do bot: {str(e)}")
        return False

def load_conversation_history_from_db(phone_number: str) -> List[Union[HumanMessage, AIMessage]]:
    """
    Carrega o histórico de conversa do banco de dados para um número específico
    """
    try:
        # Calcular timestamp de expiração
        expiry_time = datetime.now(pytz.utc) - timedelta(minutes=20)
        
        response = supabase.table("chat_history") \
            .select("*") \
            .eq("phone_number", phone_number) \
            .gte("created_at", expiry_time.isoformat()) \
            .order("created_at", desc=False) \
            .execute()
        
        messages = []
        for row in response.data:
            if row['sender'] == 'user':
                messages.append(HumanMessage(content=row['message']))
            elif row['sender'] == 'bot':
                messages.append(AIMessage(content=row['message']))
        
        return messages
        
    except Exception as e:
        logger.error(f"Erro ao carregar histórico do banco: {str(e)}")
        return []

def load_full_conversation_history(phone_number: str, nome_da_loja) -> List[dict]:
    """
    Carrega todo o histórico de conversa do banco de dados para um número específico
    """
    try:
        response = supabase.table("chat_history") \
            .select("*") \
            .eq("phone_number", phone_number) \
            .eq("loja", nome_da_loja) \
            .order("created_at", desc=False) \
            .execute()
        return response.data
    except Exception as e:
        logger.error(f"Erro ao carregar histórico completo do banco: {str(e)}")
        return []

##################################################### FIM SUPABASE ##########################################################################################


#### Inicio Reativacao de conversa

def is_stop_request(message: str) -> bool:
    """
    Usa LLM para detectar se a mensagem contém solicitação para parar reativação
    """
    try:
        # Prompt otimizado para detecção de stop requests
        prompt = f"""
        ## ANALISE DE SOLICITAÇÃO DE INTERRUPÇÃO
        
        Analise a mensagem do usuário e determine se ele está solicitando PARAR 
        as mensagens de reativação/promoção.
        
        ## CRITÉRIOS PARA CONSIDERAR COMO "STOP REQUEST":
        - Pedidos explícitos para parar/envios ("pare", "chega", "stop")
        - Menção de que já comprou em outro lugar
        - Solicitação para não receber mais mensagens
        - Expressões de desinteresse final ("não quero mais", "já resolvi")
        - Pedidos para ser removido da lista
        - Frases indicando que o assunto está encerrado
        
        ## CRITÉRIOS PARA IGNORAR (NÃO É STOP REQUEST):
        - Dúvidas sobre produtos
        - Solicitação de informações
        - Mensagens de saudação/despedida normais
        - Perguntas sobre preços/estoque
        - Expressões de interesse temporário ("mais tarde", "depois")
        
        ## MENSAGEM DO USUÁRIO:
        "{message}"
        
        ## RESPOSTA:
        Responda APENAS com "true" se for uma solicitação de parada OU "false" se não for.
        Não inclua explicações, apenas "true" ou "false".
        """
        
        # Chamar o LLM
        chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = chat.invoke(prompt)
        response_content = response.content.strip().lower()
        
        logger.info(f"LLM stop request analysis for: '{message}' -> {response_content}")
        
        # Verificar resposta
        if response_content == "true":
            return True
        elif response_content == "false":
            return False
        else:
            # Fallback para detecção por palavras-chave se a LLM retornar algo inesperado
            logger.warning(f"Resposta inesperada da LLM: {response_content}, usando fallback")
            return fallback_stop_detection(message)
            
    except Exception as e:
        logger.error(f"Erro na análise LLM de stop request: {str(e)}")
        # Fallback em caso de erro
        return fallback_stop_detection(message)

def fallback_stop_detection(message: str) -> bool:
    """
    Fallback com palavras-chave para quando a LLM falha
    """
    stop_keywords = [
        "pare de enviar", "para de mandar", "chega de mensagem", "não quero receber",
        "já comprei", "comprei em outro", "stop", "cancelar", "interromper",
        "não me envie", "não quero mais", "basta", "suficiente", "chega",
        "remover da lista", "não enviar mais", "parar mensagens", "cancelar envio",
        "não estou interessado", "já resolvi", "não preciso mais"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in stop_keywords)

def load_user_stage_from_db(phone: str) -> int:
    """
    Carrega o estágio atual do usuário a partir do banco de dados
    Retorna o valor do stage (0-4) ou 0 se não encontrado
    """
    try:
        response = supabase.table("conversation_states") \
            .select("stage") \
            .eq("phone", phone) \
            .eq("client_id", CLIENT_ID) \
            .limit(1) \
            .execute()
        
        if response.data:
            stage = response.data[0].get('stage', 0)
            logger.info(f"Stage carregado do DB para {phone}: {stage}")
            return stage
        else:
            logger.info(f"Stage não encontrado para {phone}, usando padrão 0")
            return 0
            
    except Exception as e:
        logger.error(f"Erro ao carregar stage do DB: {str(e)}")
        return 0

REACTIVATION_SEQUENCE = [
    (1, "reengajamento"),   # 3 horas
    (2, "oferta_limitada"),  # 6 horas
    (3, "fechamento_urgencia"),  # 24 horas
    (4, "stop_reativation")  # never more 
]

def should_send_reactivation_llm(phone_number: str, conversation_history: list) -> bool:
    """
    Usa LLM para analisar se deve enviar mensagens de reativação
    baseado na intenção de compra no histórico
    """
    try:
        if not conversation_history:
            return False
            
        # Formatar o histórico para análise
        history_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in conversation_history])
        
        prompt = f"""
        ## ANÁLISE DE INTENÇÃO DE COMPRA
        
        Analise o histórico de conversa abaixo e determine se este cliente
        demonstrou interesse genuíno em COMPRAR um celular/produto.
        
        ## CRITÉRIOS PARA REATIVAÇÃO (RESPONDER "true"):
        - Cliente perguntou sobre preços, modelos, estoque
        - Demonstrou intenção de compra ("quero comprar", "estou interessado")
        - Pediu orçamento ou condições de pagamento
        - Estava comparando preços com outras lojas
        - Perguntou sobre entrada/troca de aparelhos
        - Mostrou interesse específico em produtos ("iPhone 13", "Samsung S23")
        
        ## CRITÉRIOS PARA NÃO REATIVAR (RESPONDER "false"):
        - Cliente explicitamente disse que não quer comprar
        - Apenas dúvidas técnicas ("como faz backup?", "não consigo conectar")
        - Solicitações de conserto/reparo ("quebrou a tela", "não liga")
        - Reclamações sobre produtos já comprados
        - Informações gerais ("que horas fecham?", "onde fica?")
        - Orçamentos para acessórios apenas (capinhas, carregadores)
        - Conversa muito curta sem demonstração de interesse
        
        ## HISTÓRICO DA CONVERSA:
        {history_text}
        
        ## RESPOSTA:
        Responda APENAS com "true" ou "false" (sem aspas, sem explicações).
        """
        
        # Chamar o LLM para análise
        chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = chat.invoke(prompt)
        decision = response.content.strip().lower()
        
        logger.info(f"LLM decision for {phone_number}: {decision}")
        
        return decision == "true"
        
    except Exception as e:
        logger.error(f"Erro na análise LLM de reativação: {str(e)}")
        return False

def get_stage_instructions(stage_type: str) -> str:
    if stage_type == "reengajamento":
        return """
        - Voltar à conversa relembrando o interesse inicial e adicionando um novo argumento de valor.
        - Usar gatilhos como: Disponibilidade, Garantia, Promoção Relâmpago.
        - Nunca fale sobre preços ou valores
        """
    elif stage_type == "oferta_limitada":
        return """
        - Verifique no historico se a mensagem de reengajamento foi respondida ou ignorada, use esta informação no começo da mensagem.
        - Criar um senso de urgência e escassez. A oferta deve ser um benefício, não um desconto.
        - Ofereça acessórios grátis ou condições especiais.
        - Deixe claro que a nossa loja pode oferecer condições melhores, e o melhor pós venda da região.
        """
    elif stage_type == "fechamento_urgencia":
        return """
        - Forçar uma decisão final dando uma "última chance" ou se despedindo educadamente.
        - Mencione últimas unidades, fim da promoção.
        """
    else:
        return ""

def generate_reactivation_message(phone_number: str, stage_type: str) -> str:
    """
    Gera uma mensagem de reativação personalizada com base no histórico e no estágio.
    stage_type: 'reengajamento', 'oferta_limitada', 'fechamento_urgencia'
    """
    try:
        # Carregar histórico completo
        messages = load_full_conversation_history(phone_number, nome_da_loja)
        if not messages:
            return None

        # Formatar o histórico
        history_str = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in messages])

        # Obter o nome do lead (assumindo que a primeira mensagem do user é o nome)
        user_messages = [msg for msg in messages if msg['sender'] == 'user']
        name = user_messages[0]['message'].split(':')[0] if user_messages else "Cliente"

        # Obter configuração do cliente
        client_config = get_client_config()
        nome_do_agent_local = client_config.get('nome_do_agent', 'Agente')
        nome_da_loja_local = client_config.get('nome_da_loja', 'Loja')

        # Construir o prompt para o LLM
        prompt = f"""
        ## Missão
        Você é {nome_do_agent_local}, agente virtual da {nome_da_loja_local}. 
        Você já teve uma conversa com {name}, agora sua missão será tentar reativa-lo para que ele possa ser qualificado posteriormente. 

        ## Estágio: {stage_type}
        {get_stage_instructions(stage_type)}

        ## Histórico da Conversa
        {history_str}

        ## Regras
        - Começe com um cumprimento personalizado e mais informal. 
        - Não se apresente mais de uma vez, consulte o histórico. 
        - Não repita informações já dadas.
        - Não use emojis.
        - Se souber use o nome do cliente 
        - Seja breve e direto, máximo 30 palavras.
        - Não fale sobre preços.
        - Termine com uma pergunta clara (CTA).

        Gere uma mensagem de reativação para o estágio {stage_type}.
        """

        # Chamar o LLM
        chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = chat.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Erro ao gerar mensagem de reativação: {str(e)}")
        return None

### Fim Reativacao de conversa

def cleanup_expired_histories():
    while True:
        current_time = time.time()
        expired_keys = []
        
        # Identifica históricos expirados
        for user_id, data in conversation_history.items():
            elapsed = current_time - data['last_activity']
            if elapsed > HISTORY_EXPIRATION_MINUTES * 60:  # Converte para segundos
                expired_keys.append(user_id)
        
        # Remove históricos expirados
        for key in expired_keys:
            del conversation_history[key]
            logger.info(f"Removido histórico expirado para: {key}")
        
        # Verifica a cada minuto
        time.sleep(60)

# Variável global para o buffer MEGABUFFER
message_buffer = MessageBuffer(timeout=10)

def process_user_message(sender_number: str, message: str, name: str):

    qualified_status = False
    #valida stage do usuario - 0 conversa, 1 orçamento, 2 qualificação, 3 reativação, 4 não reativar
    stage_from_db = load_user_stage_from_db(sender_number)
    if stage_from_db == 4:
            logging.info(f"Usuário {sender_number} está no estágio 4 (não reativar). Ignorando mensagem.")
            return
    elif stage_from_db == 3:
        #state == 3 em reativação, verificar se a msg é de stop request

        logging.info(f"Usuário {sender_number} respondeu em reativação; removendo do funil.")
        supabase.table("conversation_states").update({
            "qualified": True,       # ou use delete() se preferir remover o registro
            "stage": 0,
            "reminder_step": 0,
            "next_reminder": None,
        }).eq("phone", sender_number).eq("client_id", CLIENT_ID).execute()   

        if is_stop_request(message):
            logging.info(f"Usuário {sender_number} solicitou parar reativação. Atualizando estado para 4.")
            #atualiza state para 4 stop_reativation
            supabase.table("conversation_states") \
            .update({"stage": 4}) \
            .eq("phone", sender_number) \
            .eq("client_id", CLIENT_ID) \
            .execute()
            
            msg = 'Ok! Sem problema. Conte conosco em uma próxima oportunidade.'
            send_whatsapp_message(sender_number, msg)
            
            return
        else:
            logging.info(f"Usuário {sender_number} no estágio 3, mas não é stop request. Continuando conversa.")
    
    logging.info(f"STAGE ATUAL: {stage_from_db} para {sender_number}")
    
    # Gerar ID único para a conversa se for uma nova
    if sender_number not in conversation_history:
        conversation_id = str(uuid.uuid4())
        # Carregar histórico do banco de dados se disponível
        history_from_db = load_conversation_history_from_db(sender_number)
        
        # CARREGAR STAGE DO BANCO DE DADOS
        
        if history_from_db:
            logging.info(f"Histórico carregado do DB para {sender_number}, mensagens: {len(history_from_db)}")
            conversation_history[sender_number] = {
                'messages': history_from_db,
                'conversation_id': conversation_id,
                'stage': stage_from_db,
                'intent': detect_intent(message),
                'bant': {'budget': None, 'authority': None, 'need': None, 'timing': None},
                'last_activity': time.time()
            }
    else:
        conversation_id = conversation_history[sender_number].get('conversation_id', str(uuid.uuid4()))
    
    #(sender_number, 'user', message, conversation_id)
    
    # Se nenhuma skill aplicável, continua com o fluxo normal
    current_intent = detect_intent(message)
    
    # Se não existir no conversation_history, inicializar vazio
    if sender_number not in conversation_history:
        conversation_history[sender_number] = {
            'messages': [],
            'conversation_id': conversation_id,
            'stage': stage_from_db if stage_from_db is not None else 0,
            'intent': detect_intent(message),
            'bant': {'budget': None, 'authority': None, 'need': None, 'timing': None},
            'last_activity': time.time()
        }
    else:
        conversation_history[sender_number]['last_activity'] = time.time()
    
    # Adiciona a mensagem do usuário ao histórico
    conversation_history[sender_number]['messages'].append(HumanMessage(content=message))
    
    history = conversation_history[sender_number]['messages'][-20:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    
    if stage_from_db != 3 :
        prompt = get_custom_prompt(message, history_str, current_intent, name)
    else:
        logging.info("-------------------------------- CUSTOM PROMPT")
        logging.info(f"phone: {sender_number}, client: {CLIENT_ID}")
        qualified_status = True
        response = supabase.table("conversation_states").update({
            "qualified": qualified_status
        }).eq("phone", sender_number).eq("client_id", CLIENT_ID).execute()  # Add client_id filter
        
        logging.info(f"response update: {response}")
        
        prompt = get_reativacao_prompt( history_str, message) #history_str, current_message
    
    ## constroi resposta para o usuario final     
    response = make_answer([SystemMessage(content=prompt)] + history)
    
    conversation_history[sender_number]['messages'].append(response)
    response_content = response.content

    save_message_to_history(sender_number, 'bot', response_content, conversation_id)
    
    logging.info(f"BANT STATUS {conversation_history[sender_number]['bant']}")

    sufixo = "@s.whatsapp.net"

    if sender_number.endswith(sufixo):
        numero = sender_number[:-len("@s.whatsapp.net")]
    else:
        numero = sender_number  # Fallback se não tiver o sufixo
    
    if "orcamento" in response_content.lower() or "orçamento" in response_content.lower():
        conversation_history[sender_number]['stage'] = 2
    elif is_qualification_message(response_content):
        logging.info(f"Qualificação detectada para {sender_number}")
        infos = get_info(history_str)
        conversation_history[sender_number]['stage'] = 3
        logging.info(f"Lead qualificado: {sender_number} - Intent: {conversation_history[sender_number]['intent']}")
        
        if isinstance(infos, str):
            try:
                infos = json.loads(infos)
            except Exception as e:
                logging.error(f"Erro ao converter infos para dict: {e}")
                infos = {}
        
        logging.info(f"Informações do lead: {infos}")
        

        demanda = infos.get('DEMANDA', 'Não Informado')
        interesse = infos.get('INTERESSE', "Produto não especificado")
        budget = infos.get('BUDGET/FORMA PAGAMENTO', "Valor não especificado")
        urgency = infos.get('URGENCIA', "Não especificado")
        pesquisando = infos.get('ESTA-PESQUISANDO', 'Não Informado')
        
        logging.info(f"DEMANDA: {demanda}, INTERESSE: {interesse}, BUDGET: {budget}, URGENCIA: {urgency}, ESTA-PESQUISANDO: {pesquisando}")
        
        msg_qualificacao = f"""
    Lead Qualificado 🔥:
    Nome: {name},
    Telefone: {numero},
    Interesse: {interesse},
    Budget: {budget},
    Urgencia: {urgency},
    Esta-Pesquisando: {pesquisando},
    Link: https://wa.me/{numero}
        """
        
        logging.info('enviando msg para grupode qualficacao')
        
        logging.info(f'Mensagem de qualificação: {msg_qualificacao}')
        id_grupo_cliente =  client_config.get('group_id', 'Não Informado')
        response = send_whatsapp_message(id_grupo_cliente, msg_qualificacao)
        upsert_qualified_lead(sender_number, CLIENT_ID)
        
        atualizar_status_lead(numero, "hot")
        logging.info(f"Lead {numero} atualizado para status 'hot' no CRM.")
    
    logging.info(f'Resposta para o Usuario: {response_content}')
    if response_content.strip() != "#no-answer":
        send_whatsapp_message(sender_number, response_content)
        current_stage = conversation_history[sender_number]['stage']
        save_conversation_state(
            sender_number=sender_number,
            last_user_message=message,
            last_bot_message=response_content,
            stage=current_stage,
            last_activity=datetime.now(pytz.utc),
            qualified_status=qualified_status
        )
        
        #insere resposta bot no crm
        #json_responde_bot = make_json_response_bot(chatName=name, chatLid=sender_number, fromMe=True, instanceId='', messageId='', status='SENT', senderName='CRM', messageType='text', messageContent=response_content, phone=numero)

        #inserir_dados_crm(json_responde_bot)
        

def is_qualification_detected(response_text: str, conversation_stage: int) -> bool:
    logging.info(f"Verificando qualificação para Estágio: {conversation_stage}")
    doc = nlp(response_text.lower())
    
    # 1. Verificação com spaCy Matcher
    #if len(matcher(doc)) > 0:
    #    return True
    
    # 2. Verificação contextual com palavras-chave
    keywords = {
        "lead quente": ["condição especial", "vendedor vai cuidar", "eles vao te ajudar"],
        "outras demandas": ["responsável vai cuidar", "grupo outras demandas"]
    }
    for _, phrases in keywords.items():
        if any(phrase in response_text.lower() for phrase in phrases):
            return True
    
    # 3. Verificação de estágio + intenção implícita
    if conversation_stage == 2:  # Estágio de qualificação
        if any(word in response_text.lower() for word in ["show", "perfeito", "beleza", "ótimo"]):
            return True
    
    return False

def is_qualification_message(message: str) -> bool:
    """
    Determina se a mensagem do agente indica que o lead foi qualificado,
    baseando-se em padrões como "vou notificar um vendedor" ou equivalentes.
    """
    # Configuração do modelo
    chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # Prompt mais confiável com formato de resposta simplificado
    prompt = f"""
    ## INSTRUÇÕES
    Analise a mensagem abaixo e responda APENAS com:
    - "true" se ela indicar que o cliente será transferido para um humano/vendedor
    - "false" caso contrário

    ## CRITÉRIOS
    Considere como TRUE se a mensagem contiver:
    - Menção a "notificar", "passar contato", "encaminhar" ou "transferir"
    - Referência a "vendedor", "especialista", "gerente" ou "humano"
    - Indicação de que outra pessoa entrará em contato
    - Frases como "vou te conectar com", "nosso time vai entrar em contato"

    ## MENSAGEM
    {message}
    
    ## RESPOSTA (APENAS true OU false):
    """
    
    try:
        # Chamada ao modelo
        response = chat.invoke(prompt)
        response_content = response.content.strip().lower()
        
        logging.info(f"Resposta do modelo para qualificação: {response_content}")
        
        # Verificação direta da resposta
        if response_content == "true":
            return True
        elif response_content == "false":
            return False
        else:
            # Se a resposta não for válida, usar fallback
            logging.warning(f"Resposta inesperada do modelo: {response_content}")
            return fallback_qualification_check(message)
            
    except Exception as e:
        logging.error(f"Erro na verificação de qualificação: {str(e)}")
        return fallback_qualification_check(message)

def fallback_qualification_check(message: str) -> bool:
    """Fallback com expressões regulares para detecção de qualificação"""
    patterns = [
        r"vou (notificar|passar|encaminhar|transferir) (para |o )?(vendedor|especialista|humano|gerente|equipe)",
        r"vou (notificar|passar|encaminhar|transferir) (seu contato|o contato)",
        r"vou (chamar|solicitar) (um|o) (vendedor|especialista|humano|gerente)",
        r"transferindo (para|o) (vendedor|especialista|humano|gerente|equipe)",
        r"passando (para|o) (vendedor|especialista|humano|gerente|equipe)",
        r"vamos te conectar",
        r"vou repassar (seu contato|para o time)",
        r"nosso time (vai|irá) entrar em contato",
        r"um (vendedor|especialista|consultor) (vai|irá) entrar em contato",
        r"aguarde um momento (que|enquanto) (vou|irei) (conectar|transferir|encaminhar)",
        r"encaminhamento (para|ao) (vendedor|especialista|humano|gerente|equipe)"
    ]
    
    for pattern in patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True
            
    return False
########################################################################## Inicio Supabase ##########################################################################################


# Inicialização do histórico de conversas (global)
conversation_history = {}

# Estados da conversa
CONVERSATION_STATES = {
    "INITIAL": 1,
    "CONVERSATION": 2,
    "REATIVACAO": 3,
    "STOP_REQUEST": 4,
    "REATIVADO": 5
}

##########################################################################  Transcrição de áudio ##########################################################################################
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def buscar_midia_por_id(instance: str, message_id: str) -> str:
    """
    Busca o áudio em base64 usando o NOVO endpoint da Evolution API.
    """
    try:
        #url = f"{EVOLUTION_SERVER_URL.rstrip('/')}/chat/getBase64FromMediaMessage/{instance}"
        url = f"{EVOLUTION_SERVER_URL}chat/getBase64FromMediaMessage/{instance}" 
        headers = {
            "apikey": EVOLUTION_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": {
                "key": {
                    "id": message_id
                }
            },
            "convertToMp4": False  # Para áudio, não precisa converter para MP4
        }
        
        logger.info(f"🔄 Buscando mídia no Evolution API: {url}")
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code in [200, 201]:
            data = response.json()
            logging.debug(f"Resposta da API: {data}")
            base64_audio = data.get("base64")
            logging.info(f"🔍 Base64 length: {len(base64_audio) if base64_audio else 'None'}")
            if base64_audio:
                logger.info("✅ Base64 encontrado via API Evolution.")
                return base64_audio
            else:
                logger.warning("⚠️ API retornou, mas sem campo base64.")
                logger.warning(f"Resposta completa: {data}")
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
                model="whisper-1",  # Mudei para whisper-1 que é mais compatível
                file=audio_file
            )

        # Limpar arquivo temporário
        os.unlink(tmp_path)
        
        return transcript.text
    except Exception as e:
        logger.error(f"❌ Erro na transcrição: {e}")
        return None
    
########################################################################## INICIO LLM ###############################################################################################

# Habilitar chave da OpenAI
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def get_info(history: list) -> str:

    prompt = f"""
    ## TAREFA
    Analise o histórico de conversa abaixo e extraia 
    - o *INTERESSE* principal do cliente
    - a *DEMANDA* (se o interesse do cliente é comprar um celular ou outro produto/serviço)
    - o *BUDGET/FORMA PAGAMENTO* (valor total que ele tem para comprar o produto, e a forma de pagamento escolhida)
    - a *URGENCIA* (Quando o cliente pretende comprar o produto)
    - *ESTA-PESQUISANDO* (Quando o cliente está fazendo o orçamento ou pesquisando em outras lojas)

    ## INSTRUÇÕES
    
    ### INTERESSE
    1. Identifique o produto/serviço que o cliente demonstrou interesse. Os serviços incluem: compra, venda, troca, conserto e impressão de documentos.
    2. Seja específico com modelos quando possível (ex: "iPhone 15 Pro" em vez de apenas "iPhone").
    3. Se mencionar troca, inclua ambos os aparelhos (ex: "Troca de iPhone X por iPhone 12").
    4. Para consertos, especifique o problema (ex: "Conserto de tela quebrada").
    5. Priorize o interesse MAIS RECENTE.
    6. Para impressão de documentos, especifique o tipo (ex: "Impressão de documentos").
    
    ### DEMANDA
    - Caso o Interesse do cliente seja a COMPRA de um celular retorne o valor "Compra"
    - Caso o Interesse do cliete não seja a COMPRA de um celular retorne o valor "Outro"

    ### BUDGET/FORMA PAGAMENTO
    - Exemplo com budget e forma de pagamento: "Budget/Forma Pagamento": "5000,00 - Pix" 
    - Exemplo com budget e sem forma de pagamento: "Budget/Forma Pagamento": "5000,00"
    - Exemplo sem budget e sem forma de pagamento : "Budget/Forma Pagamento": "Não Informado"

    ### URGENCIA
    - Idenfique a urgencia do cliente, exemplo: hoje, amanha, semana que vem, mes que vem
    - Se não houver menção de valor, retorne: "Não especificado".
    
    ### ESTA-PESQUISANDO
    - Idenrifique se o cliente está pesquisando ou orçando em outro estabelecimento 
    - Exemplo: "ESTA-PESQUISANDO": "Tem orçamento de outra loja, valor: 5200,00"

    ## IMPORTANTE
    - A resposta deve conter apenas o JSON.
    - Não adicione comentários, explicações ou qualquer outro texto fora do JSON.
    - Certifique-se de que o JSON está formatado corretamente sem ``` e sem a palavra "json" escrito, apenas as keys, valores e chaves.

    ## HISTÓRICO
    {history}
    """

    # Substituído Groq por OpenAI
    chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    response = chat.invoke(prompt)

    return response.content.strip()

def format_prompt(template, format_vars):
    """Substitui placeholders no template pelos valores fornecidos"""
    for key, value in format_vars.items():
        placeholder = "{" + key + "}"
        template = template.replace(placeholder, str(value))
    return template

def get_reativacao_prompt(history_str, current_message):
    return f"""
    ## CONTEXTO DE REATIVAÇÃO

    Você está retomando uma conversa com um lead que estava inativo. O cliente demonstrou interesse anteriormente em produtos Apple/iPhone mas não finalizou a compra. Agora ele respondeu a uma de suas mensagens de reativação.

    ## HISTÓRICO ANTERIOR DA CONVERSA (contexto importante):
    {history_str}

    ## MENSAGEM DE REATIVAÇÃO DO CLIENTE:
    "{current_message}"

    ## INSTRUÇÕES ESPECÍFICAS PARA REATIVAÇÃO:

    ### 🎯 COMPORTAMENTO NA REATIVAÇÃO:
    1. **Reconheça o retorno**: "Que bom ver você de volta!", "Obrigado por retornar!"
    2. **Relembre rapidamente o contexto anterior**: "Você estava interessado em [produto mencionado anteriormente]"
    3. **Seja mais direto e objetivo**: O cliente já conhece a loja, não precisa se reapresentar completamente
    4. **Foque em resolver objeções**: Pergunte se ainda tem interesse ou se surgiu alguma dúvida
    5. **Mantenha o entusiasmo**: Mostre que está feliz com o retorno dele

    ### 🔄 FLUXO DE QUALIFICAÇÃO NA REATIVAÇÃO:
    1. **Confirme o interesse atual**: "Você ainda está interessado em algum iPhone específico?"
    2. **Verifique mudanças**: "Alguma coisa mudou desde nossa última conversa?"
    3. **Repita qualificação rápida**: 
    - Interesse em modelos específicos
    - Forma de pagamento preferida
    - Urgência na compra

    ### 💡 GATILHOS ESPECIAIS PARA REATIVAÇÃO:
    - "Lembro que você tinha interesse em [produto], posso verificar as condições atuais?"
    - "Temos algumas novidades desde nossa última conversa que podem te interessar"
    - "Vou te passar direto para nosso especialista para condições exclusivas"

    ### 🚫 RESTRIÇÕES:
    - NÃO repita toda a apresentação inicial
    - NÃO pergunte informações que já tem no histórico
    - NÃO seja muito formal - use um tom mais descontraído
    - NÃO mencione que o cliente "sumiu" ou esteve inativo de forma negativa

    ### 📞 ENCERRAMENTO DA REATIVAÇÃO:
    Se o cliente demonstrar interesse renovado, qualifique rapidamente e encaminhe:
    "Perfeito! Vou conectar você direto com nosso especialista em iPhones para te dar condições personalizadas. Um momento!"

    ---

    ## RESPOSTA PARA O CLIENTE (baseada no histórico e mensagem atual):
    """


def get_custom_prompt(query, history_str, intent ,nome_cliente):
    client_config = get_client_config()
    # Usar valores padrão se a configuração não for encontrada
    nome_do_agent = client_config.get('nome_do_agent', 'Eduardo')
    nome_da_loja = client_config.get('nome_da_loja', 'Não Informado')
    horario_atendimento = client_config.get('horario_atendimento', 'Não Informado')
    endereco_da_loja = client_config.get('endereco_da_loja', 'Não Informado')
    categorias_atendidas = client_config.get('categorias_atendidas', 'Iphone e Acessórios')
    forma_pagamento_iphone = client_config.get('forma_pagamento_iphone', 'à vista e cartão em até 21X')
    forma_pagamento_android = client_config.get('forma_pagamento_android', 'à vista, no cartão em até 21X ou boleto')
    
    # Buscar do banco de dados
    lista_iphone = client_config.get('lista_iphone', 'Iphone 11 até Iphone 16 Pro Max')
    lista_android = client_config.get('lista_android', 'Xiaomi, Redmi, Poco')
    msg_abertura_template  = client_config.get('msg_abertura', '')
    msg_fechamento_template  = client_config.get('msg_fechamento', '')
    
    
    if msg_abertura_template:
        msg_abertura = msg_abertura_template.format(
            nome_cliente=nome_cliente,
            nome_do_agent=nome_do_agent,
            nome_da_loja=nome_da_loja,
            categorias_atendidas=categorias_atendidas
        )
        
        
    if msg_fechamento_template:
        msg_fechamento = msg_fechamento_template.format(
            horario_atendimento=horario_atendimento
        )
    
    flow = client_config.get('prompt_text', False)
    
    if not flow:
        logging.info("Não foi possivel carregar o prompt")
        return JSONResponse(content={"status": "Problemas ao tentar carregar o prompt"}, status_code=200)
    
    qdrant_results = query_qdrant(query)

    # Preparar variáveis para formatação
    format_vars = {
        'nome_do_agent': nome_do_agent,
        'nome_da_loja': nome_da_loja,
        'horario_atendimento': horario_atendimento,
        'endereco_da_loja': endereco_da_loja,
        'categorias_atendidas': categorias_atendidas,
        'forma_pagamento_iphone': forma_pagamento_iphone,
        'forma_pagamento_android': forma_pagamento_android,
        'lista_iphone': lista_iphone,
        'lista_android': lista_android,
        'msg_abertura': msg_abertura,
        'msg_fechamento': msg_fechamento,
        'history_str': history_str,
        'qdrant_results': qdrant_results,
        'query': query,
        'nome_cliente': nome_cliente,
        'intent': intent
    }
    
    formatted_prompt = format_prompt(flow, format_vars)
    
    return f"""
    # 🤖 Agente Virtual: {nome_do_agent}

    ## 📌 Contexto da Conversa

    ### 🧠 Histórico da Conversa
    {history_str}

    ### 📚 Base de Conhecimento
    {qdrant_results}  

    ## 🧠 INSTRUÇÕES PARA O AGENTE
    {formatted_prompt}

    **Mensagem Atual do Cliente:** 
    {query}
    """

def make_answer(prompt):
    # Substituído Groq por OpenAI
    chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    res = chat.invoke(prompt)
    
    response_text = res.content
    # Removido tratamento específico do Groq
    response_text = response_text.strip()
    
    return AIMessage(content=response_text)

def detect_intent(text):
    keywords = {
        'compra': ['comprar', 'quero', 'preciso de'],
        'conserto': ['consertar', 'quebrou', 'arrumar'],
        'duvida': ['quanto custa', 'tem estoque', 'garantia']
    }
    for intent, terms in keywords.items():
        if any(term in text.lower() for term in terms):
            return intent
    return 'outros'

########################################################################## FIM LLM ###############################################################################################

# Função para montar a mensagem de texto
def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )

def send_whatsapp_message(number: str, text: str):
    #logging.info(f'resposta do bot -> {text}')
    url = f"https://saraevo-evolution-api.jntduz.easypanel.host/message/sendText/{cliente_evo}"
    payload = {
        "number": number,
        "text": text
    }
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    #logging.info(f'response do bot -> {response}')
    return response


@app.post("/messages-upsert")
async def messages_upsert(request: Request):
    data = await request.json()
    key = data['data']['key']
    full_jid = key.get('senderPn') or key.get('remoteJid')
    msg_type = data['data']['messageType']
    msg_id = data['data']['key']['id']
    from_me_flag = data['data']['key']['fromMe']
    
    ## Insere Dados CRM 
    #foto_url = obter_foto_perfil(EVOLUTION_SERVER_URL, cliente_evo, EVOLUTION_API_KEY, full_jid)
    #payload_wainbound = montar_payload_wa_inbound(data,foto_url)
    #inserir_dados_crm(payload_wainbound)

    sufixo = "@s.whatsapp.net"

    if full_jid.endswith(sufixo):
        numero = full_jid[:-len("@s.whatsapp.net")]
    else:
        numero = full_jid

    bot_sender = data['sender']
    bot_number = bot_sender.split('@')[0]

    # Verificar status do bot no Supabase  - profiles
    bot_active = is_bot_active(bot_number)
    
    if from_me_flag:
        sender_type = 'bot'
    else:
        sender_type = 'user'
    
    if msg_type not in ['audioMessage','imageMessage']:
        save_message_to_history(full_jid, sender_type, data['data']['message']['conversation'])
    
    if not bot_active_per_chat[full_jid]:
        logging.info(f"Bot Inativo para este número: {full_jid}, status: {bot_active_per_chat[full_jid]}")
        return JSONResponse(content={"status": "Bot Inativo"}, status_code=200)

    if bot_active is False:
        logging.info(f"Bot Inativado de forma manual, via aplicativo, {bot_number}: {bot_active}")
        return JSONResponse(content={"status": "Bot Inativo"}, status_code=200)

    if is_group_message(full_jid):
        group_name = IGNORED_GROUPS.get(full_jid, "Grupo Desconhecido")
        logger.info(f"🚫 Mensagem de grupo ignorada: {group_name}")
        return JSONResponse(content={"status": "group_message_ignored"}, status_code=200)

    valid_numbers = [num for num in AUTHORIZED_NUMBERS if num.strip()]

    logging.info(f'NUMEROS -> {valid_numbers}')
    logging.info(f"MSG RECEIVED: {data}")

    if valid_numbers:
        if numero not in valid_numbers:
            logging.info(f'Número {numero} não cadastrado na whitelist')
            return JSONResponse(content={"status": "number ignored"}, status_code=200)
        else:
            logging.info(f"MSG RECEIVED de número autorizado: {data}")
    else:
        logging.info("Whitelist vazia - permitindo todos os números")
        
    if full_jid.endswith('@s.whatsapp.net'):
        sender_number = full_jid.split('@')[0]
    else:
        sender_number = full_jid
    
    #valida se o lead foi qualificado recentemente
    if is_lead_qualified_recently(full_jid, CLIENT_ID) and verificar_lead_qualificado is True:
        logger.info(f"Ignorando mensagem de lead qualificado recentemente: {sender_number}")
        return JSONResponse(content={"status": "qualified_lead_ignored"}, status_code=200)

    with bot_state_lock:
        bot_status = bot_active_per_chat.get(sender_number, True)

    logging.info(f'STATUS ->>>>>>> {bot_status}')

    # Extrair a mensagem do usuário
    if msg_type == 'audioMessage' and from_me_flag is False:
        #if no_horario_inatividade():
        #    logger.info("Áudio recebido no horário de inatividade")
        #    return JSONResponse(content={"status": "inactive_time"}, status_code=200)
        
        # Processamento de áudio (mantido igual)
        message_data = data['data']['message']
        base64_audio = message_data.get("base64")

        if not base64_audio:
            logger.warning(f"⚠️ Webhook sem base64, buscando via API Evolution... {data}")
            instance = data.get("instance") or data.get("instance") or "default"
            logging.info(f'INSTANCE -> {instance}')
            message_id = data['data'].get("key", {}).get("id")
            logging.info(f'MESSAGE ID -> {message_id}')
            if instance and message_id:
                base64_audio = buscar_midia_por_id(instance, message_id)
            else:
                logger.error("❌ Não foi possível obter instance ou message_id para buscar mídia.")
        
        if base64_audio:
            logger.info("🎙️ Iniciando transcrição...")
            message = transcrever_audio_base64(base64_audio)
            if message:
                logger.info(f"📝 Transcrição: {message}")
            else:
                logger.warning("⚠️ Não foi possível transcrever o áudio.")
                send_whatsapp_message(full_jid, "Desculpe, estou tendo dificuldades com este audio. Se possivel envie sua mensagem em texto.")
                return JSONResponse(content={"status": "number ignored"}, status_code=200)
        else:
            logger.warning("⚠️ Nenhum áudio disponível para transcrição.")
            send_whatsapp_message(full_jid, "Desculpe, estou tendo dificuldades com este audio. Se possivel envie sua mensagem em texto.")
            return JSONResponse(content={"status": "number ignored"}, status_code=200)
    else:
        if msg_type == 'audioMessage' and from_me_flag is True:
            return JSONResponse(content={"status": "Msg de audio enviada pela loja"}, status_code=200)
        else:        
            message = data['data']['message']['conversation']   

    name = data['data']['pushName']

    # Verificar comandos #off/#on primeiro (sempre funcionam)
    if any(word in message.strip().lower() for word in ["#off", "off"]):
        deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[full_jid] = False
        
        return JSONResponse(content={"status": f"maintenance off for {sender_number}"}, status_code=200)

    elif message.strip().lower() == "#on":
        deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[full_jid] = True
        
        return JSONResponse(content={"status": f"maintenance on for {sender_number}"}, status_code=200)
    
    if from_me_flag:
        logging.info("Mensagem enviada pelo bot, ignorando...")
        return JSONResponse(content={"status": "message from me ignored"}, status_code=200)

    #Verificar se estamos no horário de inatividade
    #if no_horario_inatividade():
    #    logger.info(f"Mensagem recebida no horário de inatividade: {message}")
    #    # Não processar a mensagem, apenas registrar no log
    #    return JSONResponse(content={"status": "inactive_time"}, status_code=200)

    # Se chegou aqui, está fora do horário de inatividade, processar normalmente
    if msg_type == 'imageMessage' and bot_status:
        send_whatsapp_message(full_jid, "Desculpe, não consigo abrir imagens. Por favor, envie a mensagem em texto.")
        return JSONResponse(content={"status": "image ignored"}, status_code=200)
    elif msg_type == 'imageMessage' and not bot_status:
        logging.info(f'msg ignorada, imagem detectada e bot está off')
        return JSONResponse(content={"status": "image ignored"}, status_code=200)

    # Processamento normal das mensagens
    if not bot_active_per_chat[full_jid]:
        logging.info(f"Ignorando mensagem de {sender_number} - Bot inativo para este número")
    else:
        message_buffer.add_message(full_jid, message, name)

        try:
            supabase.table("conversation_states").delete().eq("phone", sender_number).eq("client_id", CLIENT_ID).execute()
        except Exception as e:
            logger.error(f"Erro ao resetar reativação: {str(e)}")

    return JSONResponse(content={"status": "received"}, status_code=200)

if __name__ == "__main__":
    cleanup_thread = threading.Thread(target=cleanup_expired_histories, daemon=True)
    cleanup_thread.start()

    # Iniciar thread de reativação
    reactivation_thread = threading.Thread(target=send_reactivation_message, daemon=True)
    reactivation_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)