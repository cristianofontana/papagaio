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
HISTORY_EXPIRATION_MINUTES = 20

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
                'group_id': config.get('id_grupo_cliente', '')
            }
        else:
            logger.error(f"Configuração não encontrada para cliente: {client_id}")
            return {}
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {str(e)}")
        return {}

# Carregar configurações do Supabase
CLIENT_ID = 'mr_shop'  # ID do cliente no Supabase

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
cliente_evo = 'papagaio'  #COLLECTION_NAME
AUTHORIZED_NUMBERS = client_config.get('authorized_numbers', [''])

id_grupo_cliente =  client_config.get('group_id', 'Não Informado')#'120363420079107628@g.us' #120363420079107628@g.us id grupo papagaio 

#for pattern in patterns:
#    matcher.add("TRANSFER_PATTERNS", [pattern])


# Adicione esta classe antes da definição do app
class MessageBuffer:
    def __init__(self, timeout=20):
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

# Sequência de reativação (tempo em minutos, mensagem)
REACTIVATION_SEQUENCE = [
    (480, 
f"""Eu não vou aceitar que você suma!
Aqui, na {nome_da_loja} a gente valoriza muito todas as pessoas que entram em contato com a gente!

Você tá precisando comprar o seu celular em um lugar que te entregue, qualidade e preço justo...
e isso nós temos de sobra!!!
a gente pode se ajudar!!!
me da 5 minutos da sua atenção que eu resolvo sua vida!"""),
    (960, 
"""Como eu te disse ontem... Eu não vou te abandonar... Ou voce me dá atenção
ou eu vou descobrir onde voce mora e ir ai na sua casa!!!
KKKKKKKKKKKK
me ajuda a te ajudar!!! Eu preciso bater a meta e você precisa de um novo CELULAR!!!"""),
    (960*2, 
"""Você tem dois caminhos:
Primeiro Caminho: Você vai ver essa mensagem, e vai me ignorar e a gente
nunca mais vai conversar... Provavelmente você vai comprar em outra loja,
essa loka, vai te prometer mundos e fundos, mas na hora que você precisar,
ELES VÃO SUMIR...

Segundo Caminho: Você me da 5 minutos da sua atenção, tempo suficiente 
pra eu provar que você está na loja certa... Te vendo um produto no preço 
justo, e com toda a qualidade do mundo, e você vira cliente fiel!
o segundo caminho é melhor não é ?"""),
    (960*4, 
"""
Uma vez me disseram que pessoas inteligentes são aquelas que estão 
sempre disponiveis pra conversar e escutar novas propostas...
eu sei que você precisa de um celular e eu tambem seu que você é uma pessoa inteligente não é ?"""),
    (960*8, 
"""Você é inteligente é ?""")
]

def save_conversation_state(sender_number: str, last_user_message: str, 
                           last_bot_message: str, stage: int, last_activity: datetime):
    qualified = stage >= 3

    data = {
        "phone": sender_number,
        "last_user_message": last_user_message,
        "last_bot_message": last_bot_message,
        "stage": stage,
        "last_activity": last_activity.isoformat(),
        "next_reminder": (last_activity + timedelta(minutes=REACTIVATION_SEQUENCE[0][0])).isoformat(),
        "reminder_step": 0,
        "qualified": qualified  # Agora calculado corretamente
    }
    
    
    try:
        # Upsert no Supabase
        supabase.table("conversation_states").upsert(data).execute()
    except Exception as e:
        logger.error(f"Erro ao salvar estado no Supabase: {str(e)}")

def update_reminder_step(phone: str, step: int):
    try:
        next_reminder_time = datetime.now(pytz.utc) + timedelta(minutes=REACTIVATION_SEQUENCE[step][0])
        supabase.table("conversation_states").update({
            "reminder_step": step,
            "next_reminder": next_reminder_time.isoformat(),
            "qualified": False
        }).eq("phone", phone).execute()
    except Exception as e:
        logger.error(f"Erro ao atualizar passo de lembrete: {str(e)}")

# Função para enviar mensagens de reativação
def send_reactivation_message():
    while True:
        try:
            now = datetime.now(pytz.utc)
            result = supabase.table("conversation_states").select("*").lte("next_reminder", now.isoformat()).eq("qualified", False).execute()
            
            for row in result.data:
                phone = row["phone"]
                step = row["reminder_step"]
                
                if step < len(REACTIVATION_SEQUENCE):
                    message = REACTIVATION_SEQUENCE[step][1]
                    send_whatsapp_message(phone, message)
                    
                    # Atualizar para o próximo passo
                    new_step = step + 1
                    if new_step < len(REACTIVATION_SEQUENCE):
                        update_reminder_step(phone, new_step)
                    else:
                        # Remover da lista de acompanhamento
                        supabase.table("conversation_states").delete().eq("phone", phone).execute()
        
        except Exception as e:
            logger.error(f"Erro no envio de reativação: {str(e)}")
        
        # Verificar a cada minuto
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

##################################################### FIM SUPABASE ##########################################################################################

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

# Variável global para o buffer
message_buffer = MessageBuffer(timeout=3)

def process_user_message(sender_number: str, message: str, name: str):

    # Gerar ID único para a conversa se for uma nova
    if sender_number not in conversation_history:
        conversation_id = str(uuid.uuid4())
    else:
        conversation_id = conversation_history[sender_number].get('conversation_id', str(uuid.uuid4()))
    
    save_message_to_history(sender_number, 'user', message, conversation_id)
    
    # Se nenhuma skill aplicável, continua com o fluxo normal
    current_intent = detect_intent(message)
    
    # Inicializa ou atualiza o histórico da conversa
    if sender_number not in conversation_history:
        conversation_history[sender_number] = {
            'messages': [],
            'stage': 0,
            'intent': current_intent,
            'bant': {'budget': None, 'authority': None, 'need': None, 'timing': None},
            'last_activity': time.time()
        }
    else:
        conversation_history[sender_number]['last_activity'] = time.time()
    
    # Adiciona a mensagem do usuário ao histórico
    conversation_history[sender_number]['messages'].append(HumanMessage(content=message))
    
    logging.info(f'Intenção detectada: {current_intent}')
    
    history = conversation_history[sender_number]['messages'][-20:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    
    prompt = get_custom_prompt(message, history_str, current_intent)
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
        

        interesse = infos.get('INTERESSE', "Produto não especificado")
        budget = infos.get('BUDGET', "Valor não especificado")
        
        msg_qualificacao = f"""
    Lead Qualificado 🔥:
    Nome: {name},
    Telefone: {numero},
    Interesse: {interesse},
    Budget: {budget},
    Compra urgente.
    Link: https://wa.me/{numero}
        """
        logging.info('enviando msg para grupode qualficacao')
        send_whatsapp_message(id_grupo_cliente, msg_qualificacao)
    
    logging.info(f'Resposta para o Usuario: {response_content}')
    if response_content.strip() != "#no-answer":
        send_whatsapp_message(sender_number, response_content)
        current_stage = conversation_history[sender_number]['stage']
        save_conversation_state(
            sender_number=sender_number,
            last_user_message=message,
            last_bot_message=response_content,
            stage=current_stage,
            last_activity=datetime.now(pytz.utc)
        )
        

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
    "INITIAL": 0,
    "NEED_IDENTIFIED": 1,
    "QUALIFICATION": 2,
    "HOT_LEAD": 3,
    "CLOSED": 4
}

##########################################################################  Transcrição de áudio ##########################################################################################
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

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
    
########################################################################## INICIO LLM ###############################################################################################

# Habilitar chave da OpenAI
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def get_info(history: list) -> str:

    prompt = f"""
    ## TAREFA
    Analise o histórico de conversa abaixo e extraia o INTERESSE principal do cliente e o BUDGET (valor total que ele tem para comprar o produto).

    ## INSTRUÇÕES

    ### INTERESSE
    1. Identifique o produto/serviço que o cliente demonstrou interesse.
    2. Seja específico com modelos quando possível (ex: "iPhone 15 Pro" em vez de apenas "iPhone").
    3. Se mencionar troca, inclua ambos os aparelhos (ex: "Troca de iPhone X por iPhone 12").
    4. Para consertos, especifique o problema (ex: "Conserto de tela quebrada").
    5. Priorize o interesse MAIS RECENTE.
    6. Se não encontrar interesse claro, retorne: "Produto não especificado".

    ### BUDGET
    2. Se não houver menção de valor, retorne: "Valor não especificado".

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


def get_custom_prompt(query, history_str, intent):
    client_config = get_client_config()
    # Usar valores padrão se a configuração não for encontrada
    nome_do_agent = client_config.get('nome_do_agent', 'Eduardo')
    nome_da_loja = client_config.get('nome_da_loja', 'Não Informado')
    horario_atendimento = client_config.get('horario_atendimento', 'Não Informado')
    endereco_da_loja = client_config.get('endereco_da_loja', 'Não Informado')
    categorias_atendidas = client_config.get('categorias_atendidas', 'Iphone e Acessórios')
    forma_pagamento_iphone = client_config.get('forma_pagamento_iphone', 'à vista e cartão em até 21X')
    forma_pagamento_android = client_config.get('forma_pagamento_android', 'à vista, no cartão em até 21X ou boleto')

    flow = f"""
    ## 🧭 Missão
    Você é  {nome_do_agent}, agente virtual da loja de celulares {nome_da_loja}. Sua função é **qualificar leads automaticamente usando o método abaixo** e, se estiverem qualificados, encaminhá-los para um especialista humano finalizar a venda.
    
    ### Etapas de qualificação
    > Para Celulares 
    > Sempre faça o item 4. Validação de Pagamento (APENAS CELULARES)
    1. Abertura 
    2. Identificação da Necessidade 
    3. Entrada de Aparelho (APENAS iPHONE)
    4. Validação de Pagamento (APENAS CELULARES)
    5. Urgência [APENAS CELULARES]
    6. Lead Qualificado

    > Outros
    2.5 Fluxo Especial para Outros
    
    Endereço da loja: {endereco_da_loja}

    ---

    ## 🎯 Fluxo de Conversa e Qualificação

    ### 1. 👋 Abertura
    Inicie a conversa se apresentando:
    > Oiii, tudo bem ? Quem ta falando é {nome_do_agent}, a IA da {nome_da_loja}! Eu estou pronto para te ajudar!!!

    > Me conta ai, o que está procucando hoje, aqui trabalhamos com: {categorias_atendidas}

    ---

    ### 2. 🧠 Identificação da Necessidade 
    - **Se o cliente mencionar acessórios** (capinha, carregador, fone, película, etc.):
    > "Entendi! Você pode me dizer qual tipo de acessório está buscando?"
    - Aguarde a especificação do acessório
    - **Pule direto para a Etapa 2.5**

    - Para celulares (iPhone/Android):
    - **NUNCA mostre preços na listagem**
    - **NUNCA mencione valores mesmo que o cliente peça explicitamente**
    - Use a Base de Conhecimento para listar os Produtos disponíveis

    - Caso o cliente não saiba exatamento o que quer ou pergunte o que tem:
    - Acesse a **Base de conhecimento** e liste até 5 opções com nome e armazenamento, exemplo:
    > "Olha, temos disponível:"
    > - iPhone 11 
    > - iPhone 13 
    ...
    > - iPhone 12 

    ---

    ### 2.5 🎧 Fluxo Especial para Outros
    - Após cliente especificar o acessório (ex: "capinha para iPhone 13"):
   
    - Qualquer resposta sobre o acessorio considera lead qualificado
    exemplos: 
    1. Capinha para iphone
    2. Carregador tipo C 
    ...
    - **Encaminhe imediatamente para o grupo de leads quentes**:
    > "Perfeito! Já adicionei você na nossa lista prioritária. Um especialista em acessórios vai entrar em contato ainda hoje, ok? Lembrando que atendemos das {horario_atendimento}!"

    - **FIM DO FLUXO PARA ACESSÓRIOS**

    ---

    ### 3. 🔁 Entrada de Aparelho (APENAS iPHONE)
    Se o cliente estiver interessado em um iPhone:
    > "Você pretende usar o seu iPhone atual como parte do pagamento?"
    *** Faça essa pergunta APENAS se o cliente estiver interessado em um iPhone

    - Se o cliente disser **sim**:
        - Pergunte:
        > "Qual o modelo exato do seu aparelho? (Ex: iPhone 11 Pro Max, iPhone SE 2020)"
        
        - **Validação rigorosa**:
            1. Consulte a **Base de Conhecimento** procurando pelo **modelo exato** informado
            2. Se não encontrar:
                - Verifique equivalências conhecidas:
                    - "iPhone X" → "iPhone 10" (e vice-versa)
                    - "iPhone 11 Pro" ≠ "iPhone 11" (modelos diferentes)
            3. Critérios de aceite:
                - ✅ APENAS se encontrar registro COM `aceita_como_entrada = "SIM"`
                - ❌ Caso contrário: rejeite

        - Responda **baseada estritamente nos resultados**:
            - Se encontrou modelo equivalente válido:
            > "Perfeito! Esse modelo é aceito como entrada sim."
            > "Você saberia me dizer como está a saúde da bateria? E se o aparelho já foi aberto, tem riscos ou trincados?"
            
            - Se **não encontrou** ou modelo inválido:
            > "No momento não estamos aceitando iPhone X/10 como entrada, mas posso te ajudar com outras formas de pagamento! Quer prosseguir?"
            > *[Aguarde resposta antes de avançar]*

    ---
    ### 4. 💳 Validação de Pagamento (APENAS CELULARES)
    Após confirmar urgência, pergunte sobre a forma de pagamento:

    #### Para iPhone:
    > "Para finalizar, você prefere pagar à vista ou no cartão?"
    - se o cliente perguntar sobre boleto, fale: "Para iPhones trabalhamos apenas com {forma_pagamento_iphone}. Qual dessas prefere?"


    - **Formas aceitas:** {forma_pagamento_iphone}
    - Se cliente sugerir outra forma:
    > "Para iPhones trabalhamos apenas com: {forma_pagamento_iphone}. Qual dessas prefere?"
    - Para parcelamentos, considere 1x, 2x ... 21x

    #### Para Android:
    > "Para finalizar, você prefere pagar {forma_pagamento_android}?"

    - **Formas aceitas:** {forma_pagamento_android}
    - Se cliente sugerir outra forma:
    > "Para Androids aceitamos {forma_pagamento_android}. Qual dessas formas se encaixa melhor?"

    #### Para outros produtos:
    - Não perguntar sobre forma de pagamento

    ---

    ### 5. ⏱️ Urgência [APENAS CELULARES]
    Depois:
    > "E você pretende comprar pra quando?"

    - Se o cliente disser algo como "hoje", "o quanto antes", "essa semana":
    - **Lead está qualificado** com urgência.
    - Se o cliente disser "sem pressa":
    - Use um **gatilho de urgência leve**:
        > "Boa! Só vale lembrar que os preços podem variar rápido por conta do dólar, tá?"

    ---

    ### 6. ✅ Lead Qualificado
    > Se o LEAD estiver qualificado, construa uma mensagem de resposta baseada no exemplo abaixo, mas personalize com as informações do lead, data e hora atual comparando com o horário de atendimento da loja.

    Exemplo de mensagem:
    > "Show! Já chamei um vendedor nosso aqui no WhatsApp. Ele vai cuidar de você com uma condição especial, beleza? Lembrando que nosso horário de atendimento é {horario_atendimento}, ele te chama logo mais!"

    ---

    ## 🧠 Regras e Lógica

    - **Para acessórios:**
    - Descubra apenas o tipo de acessório
    - Pergunte apenas sobre urgência
    - Encaminhe imediatamente após confirmar urgência
    - Não pergunte sobre orçamento ou entrada

    - Para celulares:
    - Sempre **pergunte uma coisa por vez**.
    - Nunca mencione **preço**. Apenas valide se “pode ser atendido”.
    - Se o cliente **não souber o modelo**, ofereça uma **lista curta**.
    - Não ofereça celulares que nao estiverem na Base de Conhecimento
    - Não repita uma pergunta se já foi feita anteriormente, verifique no ### 🧠 Histórico da Conversa, antes de formular sua pergunta.
    - Nunca aceite como entrada um modelo que não esteja na Base de Conhecimento.

    ---

    ## ⚠️ Ações Proibidas

    - Jamais revele valores específicos, mesmo se o cliente perguntar diretamente
    - Não fale valores diretamente.
    - Não invente modelos que não estão na Base de Conhecimento.
    - Não elogie aparelhos nem force entusiasmo.
    - Não retome o atendimento depois que encaminhar para o especialista.
    - Não aceite como entrada um modelo que não esteja na Base de Conhecimento.

    ---

    ## 📌 Exemplo de Conversa (Acessórios)

    **Bot:** Olá, sou {nome_do_agent}, da Mr Shop! Vou te ajudar hoje. Você está buscando algo específico?
    **Cliente:** Queria um carregador pra iPhone.

    **Bot:** Entendi! Você pode me dizer qual tipo de acessório está buscando?
    **Cliente:** Um carregador original pra iPhone 15.

    **Bot:** Anotado! Você precisa desse acessório para quando?
    **Cliente:** Se possível até amanhã.

    **Bot:** Perfeito! Já adicionei você na nossa lista prioritária. Um especialista em acessórios vai entrar em contato ainda hoje, ok? Lembrando que atendemos das 9h às 18h de Segunda a Sabado!
    """

    qdrant_results = query_qdrant(query)

    return f"""
    # 🤖 Agente Virtual: {nome_do_agent}

    ## 📌 Contexto da Conversa

    ### 🧠 Histórico da Conversa
    {history_str}

    ### 📚 Base de Conhecimento
    {qdrant_results}  

    ## 🧠 INSTRUÇÕES PARA O AGENTE
    {flow}

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
    data = await request.json() #body
    full_jid = data['data']['key']['remoteJid']
    msg_type = data['data']['messageType']
    msg_id = data['data']['key']['id']
    from_me_flag = data['data']['key']['fromMe'] 

    sufixo = "@s.whatsapp.net"

    if full_jid.endswith(sufixo):
        numero = full_jid[:-len("@s.whatsapp.net")]
    else:
        numero = full_jid  # Fallback se não tiver o sufixo

    bot_sender = data['sender']
    bot_number = bot_sender.split('@')[0]

    # Verificar status do bot no Supabase
    bot_active = is_bot_active(bot_number)

    if bot_active is False:
        logging.info(f"Bot Inativado de forma manual, via aplicativo, {bot_number}: {bot_active}")
        return JSONResponse(content={"status": "Bot Inativo"}, status_code=200)

    if is_group_message(full_jid):
        group_name = IGNORED_GROUPS.get(full_jid, "Grupo Desconhecido")
        logger.info(f"🚫 Mensagem de grupo ignorada: {group_name}")
        return JSONResponse(content={"status": "group_message_ignored"}, status_code=200)

    valid_numbers = [num for num in AUTHORIZED_NUMBERS if num.strip()]  # Remove strings vazias

    logging.info(f'NUMEROS -> {valid_numbers}')
    logging.info(f"MSG RECEIVED: {data}")

    if valid_numbers:  # Se houver números válidos na lista
        if numero not in valid_numbers:
            logging.info(f'Número {numero} não cadastrado na whitelist')
            return JSONResponse(content={"status": "number ignored"}, status_code=200)
        else:
            logging.info(f"MSG RECEIVED de número autorizado: {data}")
    else:
        logging.info("Whitelist vazia - permitindo todos os números")
        
    

    # Extrair sender_number ANTES de verificar o tipo de mensagem
    if full_jid.endswith('@s.whatsapp.net'):
        sender_number = full_jid.split('@')[0]
    else:
        sender_number = full_jid  # Fallback

    with bot_state_lock:
        bot_status = bot_active_per_chat.get(sender_number, True)
    
    logging.info(f'STATUS ->>>>>>> {bot_status}')

    if msg_type == 'imageMessage' and bot_status:
        send_whatsapp_message(full_jid, "Desculpe, não consigo abrir imagens. Por favor, envie a mensagem em texto.")
        return JSONResponse(content={"status": "image ignored"}, status_code=200)
    elif msg_type == 'imageMessage' and not bot_status:
        logging.info(f'msg ignorada, imagem detectada e bot está off')
        return JSONResponse(content={"status": "image ignored"}, status_code=200)
    elif msg_type == 'audioMessage':
        message = data['data']['message']
        base64_audio = message.get("base64")

        if not base64_audio:
            logger.warning(f"⚠️ Webhook sem base64, buscando via API Evolution... {data}")
            instance = data.get("instance") or data.get("instance") or "default"
            logging.info(f'INSTANCE -> {instance}')
            message_id = data.get("key", {}).get("id")
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
        message = data['data']['message']['conversation']   

    #logging.info(f"MSG RECEIVED: {data}")
    #bot_sender = data['sender']
    #bot_number = bot_sender.split('@')[0]
    
    name = data['data']['pushName']

    #logger.info(f"MSG RECEIVED FROM {sender_number}: {message}")

    if message.strip().lower() == "#off":
        deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[sender_number] = False
        
        return JSONResponse(content={"status": f"maintenance off for {sender_number}"}, status_code=200)

    elif message.strip().lower() == "#on":
        deletar_mensagem(msg_id, full_jid, from_me_flag)
        with bot_state_lock:
            bot_active_per_chat[sender_number] = True
        
        return JSONResponse(content={"status": f"maintenance on for {sender_number}"}, status_code=200)

        
    
    if not bot_active_per_chat[sender_number]:
        logging.info(f"Ignorando mensagem de {sender_number} - Bot inativo para este número")
    else:
        message_buffer.add_message(full_jid, message, name)

        try:
            supabase.table("conversation_states").delete().eq("phone", sender_number).execute()
        except Exception as e:
            logger.error(f"Erro ao resetar reativação: {str(e)}")

    return JSONResponse(content={"status": "received"}, status_code=200)

if __name__ == "__main__":
    cleanup_thread = threading.Thread(target=cleanup_expired_histories, daemon=True)
    cleanup_thread.start()

    # Iniciar thread de reativação
    #reactivation_thread = threading.Thread(target=send_reactivation_message, daemon=True)
    #reactivation_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)