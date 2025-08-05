from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi import FastAPI, HTTPException, Request, Query
import json
import aiohttp
import asyncio
import re

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
import spacy
from spacy.matcher import Matcher
from collections import defaultdict

import base64
import hashlib
from Crypto.Cipher import AES
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import tempfile
import openai

load_dotenv()
HISTORY_EXPIRATION_MINUTES = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVOLUTION_API_KEY = os.getenv("EVO_API_KEY")
EVOLUTION_SERVER_URL = 'https://saraevo-evolution-api.jntduz.easypanel.host/'  # Ex.: https://meu-servidor-evolution.com


bot_active_per_chat = defaultdict(lambda: True)  # Estado do bot por número do cliente
AUTHORIZED_NUMBERS = ['554108509968']
bot_state_lock = Lock()  # Lock para sincronização de estado

# ================== API FastAPI ================== #
app = FastAPI(title="WhatsApp Transcription API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load('pt_core_news_sm')

matcher = Matcher(nlp.vocab)

patterns = [
    [{"LOWER": {"IN": ["passar", "encaminhar", "transferir"]}}, {"LOWER": "para"}, {"LOWER": {"IN": ["gerente", "vendedor", "humano"]}}],
    [{"LOWER": "chamei"}, {"LOWER": "um"}, {"LOWER": {"IN": ["vendedor", "especialista"]}}],
    [{"LOWER": "finalizar"}, {"LOWER": "atendimento"}],
    [{"LOWER": "encaminhamento"}, {"LOWER": "para"}, {"LOWER": "humanos"}]
]

nome_do_agent = 'Papagaio'
nome_da_loja = 'Mr Shop'
horario_atendimento = '9h às 18h de Segunda a Sabado'
endereco_da_loja = 'Av. Pres. Carlos Luz - Pirapetinga, MG, 36730-000' 
metodo_de_pagamento = {
    'iphone': {'cartao'},
    'Android': {'cartao','boleto'},
    'Outros': {'cartao','boleto'}
}

for pattern in patterns:
    matcher.add("TRANSFER_PATTERNS", [pattern])


# Adicione esta classe antes da definição do app
class MessageBuffer:
    def __init__(self, timeout=10):
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
COLLECTION_NAME = "produtos"  # Nome da coleção que você criou
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


########################################################################## FIM RAG SYSTEM #######################################################################################

############################################################# INICIO SUPABASE ##########################################################################################
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Sequência de reativação (tempo em minutos, mensagem)
REACTIVATION_SEQUENCE = [
    (1, 
"""Eu não vou aceitar que voçê suma!
Aqui, na {nome_da_loja} a gente valoriza muito todas as pessoas que entram em contato com a gente!

Você tá precisando comprar o sey celylar em um lugar que te entregue, qualidade e preço justo...
e isso nós temos de sobra!!!
a gente pode se ajudar!!!
me da 5 minutos da sua atenção que eu resolvo sua vida!"""),
    (2, 
"""Como eu te disse ontem... Eu não vou te abandonar... Ou voce me dá atenção
ou eu vou descobrir onde voce mora e ir ai na sua casa!!!
KKKKKKKKKKKK
me ajuda a te ajudar!!! Eu preciso bater a meta e voce precisa de um novo CELULAR!!!"""),
    (3, 
"""Você tem dois caminhos:
primeiro: Você vai ver essa mensagem, e vai me ignorar e a gente
nunca mais vai conversar... Provavelmente você vai comprar em outra loja,
essa loka, vai te prometer mundos e fundos, mas na hora que você precisar,
ELES VÃO SUMIR...

Segundo Caminho: Você me da 5 minutos da sua atenção, tempo suficiente 
pra eu provar que você está na loja certa... Te vendo um produto no preço 
justo e com toda a qualidade do mundo e você vira cliente fiel!
o segundo caminho é melhor não é ?"""),
    (4, 
"""
Uma vez me disseram que pessoas iteligentes são aquelas que estão 
sempre disponiveis pra conversar e escutar novas propostas...
eu sei que você precisa de um celular e eu tambem seu que você é uma pessoa inteligente não é ?"""),
    (5, 
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
    # Primeiro tente usar uma skill especializada
    
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

    logging.info(f"BANT STATUS {conversation_history[sender_number]['bant']}")
    
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
        Telefone: {sender_number},
        Interesse: {interesse},
        Budget: {budget},
        Compra urgente.
        Link: https://wa.me/{sender_number}
        """
        
        send_whatsapp_message('120363420079107628@g.us', msg_qualificacao)
    
    logging.info(f'RESPONSE: {response_content}')
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
    if len(matcher(doc)) > 0:
        return True
    
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
    1. Extraia o valor que o cliente mencionou como orçamento.
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

    flow = f"""
    ## 🧭 Missão
    Você é o {nome_do_agent}, agente virtual da loja de celulares {nome_da_loja}. Sua função é **qualificar leads automaticamente usando o método BANT** e, se estiverem qualificados, encaminhá-los para um especialista humano finalizar a venda.
    Endereço da loja: {endereco_da_loja}

    ---

    ## 🎯 Fluxo de Conversa e Qualificação

    ### 1. 👋 Abertura
    Inicie a conversa se apresentando:
    > "Olá, sou o {nome_do_agent}, da loja de celulares {nome_da_loja}! Vou te ajudar hoje. Você está buscando algo específico?"

    ---

    ### 2. 🧠 Identificação da Necessidade (N – Need)
    - **NUNCA mostre preços na listagem**
    - **NUNCA mencione valores mesmo que o cliente peça explicitamente**
    - Use a Base de Conhecimento para listar os Produtos disponíveis

    - Caso o cliente não saiba exatamento o que quer ou pergunte o que tem:
    - Acesse a **Base de conhecimento** e liste até 5 opções com nome e armazenamento, exemplo:
    > "Olha, temos disponível:"
    > - iPhone 11 
    > - iPhone 13 
    > - iPhone 15 
    > - iPhone XR 
    > - iPhone 12 

    ---

    ### 3. 💰 Orçamento (B – Budget)
    Após o cliente indicar um modelo, pergunte:
    > "Legal! Quanto você está pensando em investir no [Modelo Escolhido]?"

    - Se o cliente perguntar diretamente pelo preço:
    > "Os valores variam conforme condições de pagamento. Pra te ajudar melhor, qual seria seu investimento máximo?"

    - Quando o cliente informar o valor:
    - Verifique na **Base de conhecimento** se o preço desejado está próximo do valor real (`preco_novo` ou `preco_semi_novo`).
    - **Responda de acordo:**
        - Se estiver próximo:
        > "Show! Esse valor pode sim ser atendido com o [Modelo Escolhido]."
        - Se estiver bem abaixo:
        > "Pode ser que esse valor fique um pouco abaixo do preço atual. Quer ver outras opções parecidas dentro dessa faixa?"

    ---

    ### 4. 🔁 Entrada de Aparelho (se for iPhone)

    Se o cliente estiver interessado em um **iPhone**, pergunte:
    > "Você pretende usar o seu iPhone atual como forma de entrada no pagamento?"

    - Se o cliente disser **sim**:
        - Pergunte:
        > "Qual o modelo do seu aparelho atual?"

        - Verifique no **Vector DB** se o modelo informado é aceito como entrada (`aceita_como_entreda = "SIM"`).

        - Responda de acordo:
            - ✅ Se for aceito:
            > "Perfeito! Esse modelo é aceito como entrada sim."

            - ❌ Se **não** for aceito:
            > "Esse modelo infelizmente não conseguimos aceitar como entrada, mas posso te ajudar com outras formas de pagamento, pode ser?"

        - Depois, pergunte:
        > "Você saberia me dizer como está a saúde da bateria? E se o aparelho já foi aberto, tem riscos ou trincados?"

    - Se o cliente disser **não**:
        > "Sem problemas! Podemos continuar com outras formas de pagamento."
    
    Se o cliente estiver interessado em um Android ou acessorio:
        - Não pergunte sobre usar o celular atual como forma de entrada no pagamento, e de continuidade no fluxo de conversas.

    ---

    ### 5. ⏱️ Urgência (T – Timeline)
    Depois de entender o orçamento, pergunte:
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

    - Sempre **pergunte uma coisa por vez**.
    - Nunca mencione **preço**. Apenas valide se “pode ser atendido”.
    - Se o cliente **não souber o modelo**, ofereça uma **lista curta**.
    - Se o cliente **fugir do fluxo** (ex: pedir capinha, assistência, etc.), redirecione:
    > "Esse atendimento é focado em venda de celulares. Posso te ajudar com isso?"

    ---

    ## ⚠️ Ações Proibidas

    - Jamais revele valores específicos, mesmo se o cliente perguntar diretamente
    - Não fale valores diretamente.
    - Não invente modelos que não estão na Base de Conhecimento.
    - Não elogie aparelhos nem force entusiasmo.
    - Não retome o atendimento depois que encaminhar para o especialista.

    ---

    ## 📌 Exemplo de Conversa Ideal (Simulação)

    ### Conversa 1 - Cliente querendo um iphone 
    **Bot:** Olá, sou o Papagaio, da {nome_da_loja}. Vou te ajudar hoje! Você tem algo em mente?
    **Cliente:** Tava querendo um iPhone.

    **Bot:** Claro! Hoje temos:
    - iPhone 11 
    - iPhone 13 
    - iPhone 15 

    **Cliente:** Gostei do iPhone 13.
    **Bot:** Legal! Qual faixa de valor você pensa pra ele?=
    **Cliente:** Até R$ 3.500.
    **Bot:** Esse valor pode sim ser atendido com esse modelo.
    **Bot:** Você pretende usar o seu iPhone atual como entrada?
    **Cliente:** Sim, é um iPhone XR.
    **Bot:** Perfeito! Esse modelo é aceito como entrada sim. Você saberia me dizer como está a saúde da bateria? Já foi aberto ou tem algum risco?
    **Cliente:** Tá com bateria 85% e sem riscos.
    **Bot:** Show! E pretende comprar pra quando?
    **Cliente:** Queria hoje ainda.
    **Bot:** Show! Já chamei um vendedor nosso aqui no WhatsApp. Ele vai cuidar de você com uma condição especial, beleza? Lembrando que nosso horário de atendimento é das 09h às 18h, ele te chama logo mais!

    ### Conversa 2 - Cliente querendo um Android 
    **Bot:** Olá, sou o Papagaio, da {nome_da_loja}. Vou te ajudar hoje! Você tem algo em mente?
    **Cliente:** Tava querendo um Celular.

    **Bot:** Claro! Hoje temos:
    - iPhone 11  
    - iPhone 13 
    - iPhone 15 
    - Iphone 14 
    ...

    **Cliente:** tem xiaomi ? 
    **Bot:** Claro! Hoje temos:
    - Redmi Note 13 pro   
    - Poco X7  
    ...
    - POCO F6 PRO 
    ***Cliente: Gostei do POCO F6 PRO 

    **Bot:** Legal! Qual faixa de valor você pensa pra ele?=
    **Cliente:** Até R$ 3.500.=
    **Bot:** Esse valor pode sim ser atendido com esse modelo.
    **Bot:** Você pretende usar o seu iPhone atual como entrada?
    **Cliente:** Sim, é um Redmi note 10 PRO.
    **Bot:** Esse modelo infelizmente não conseguimos aceitar como entrada, mas posso te ajudar com outras formas de pagamento, pode ser?
    **Cliente:** ok
    **Bot:** Show! E pretende comprar pra quando?
    **Cliente:** Queria hoje ainda.
    **Bot:** Show! Já chamei um vendedor nosso aqui no WhatsApp. Ele vai cuidar de você com uma condição especial, beleza? Lembrando que nosso horário de atendimento é das 09h às 18h, ele te chama logo mais!

    ### Conversa 3 - Cliente perguntando preço diretamente
    **Cliente:** Qual é o valor do iPhone 14?
    **Bot:** Os valores variam conforme condições. Pra te ajudar, qual seria seu investimento máximo?
    **Cliente:** Até R$ 3.500.
    **Bot:** Seu investimento é compatível! 
    **Bot:** Você pretende usar seu aparelho atual como entrada?

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
    
    ## 🧠 Objetivo Agora

    Com base nas instruções acima, responda de forma natural, curta e com uma pergunta por vez. Siga o fluxo conforme a etapa atual da conversa:

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
    url = "https://saraevo-evolution-api.jntduz.easypanel.host/message/sendText/papagaio"
    payload = {
        "number": number,
        "text": text
    }
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response


@app.post("/messages-upsert")
async def messages_upsert(request: Request):
    data = await request.json()
    full_jid = data['data']['key']['remoteJid']
    msg_type = data['data']['messageType']

    logging.info(f"MSG RECEIVED: {data}")

    if msg_type == 'imageMessage':
        send_whatsapp_message(full_jid, "Desculpe, não consigo abrir imagens. Por favor, envie a mensagem em texto.")
        return JSONResponse(content={"status": "image ignored"}, status_code=200)
    elif msg_type == 'audioMessage':
        message = data['data']['message']
        base64_audio = message.get("base64")

        if not base64_audio:
            logger.warning("⚠️ Webhook sem base64, buscando via API Evolution...")
            instance = data.get("instance") or data.get("instance") or "default"
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
        else:
            logger.warning("⚠️ Nenhum áudio disponível para transcrição.")
    else:        
        
        sender_number = full_jid.split('@')[0]
        message = data['data']['message']['conversation']   

    bot_sender = data['sender']
    bot_number = bot_sender.split('@')[0]
    
    name = data['data']['pushName']

    #logger.info(f"MSG RECEIVED FROM {sender_number}: {message}")
    
    if message.strip().lower() == "#off":
        with bot_state_lock:
            bot_active_per_chat[sender_number] = False
        send_whatsapp_message(bot_number, "🤖 Bot desativado para conversa com {sender_number}. Não responderei novas mensagens até ser reativado com #on")
        return JSONResponse(content={"status": f"maintenance off for {sender_number}"}, status_code=200)

    elif message.strip().lower() == "#on":
        with bot_state_lock:
            bot_active_per_chat[sender_number] = True
        send_whatsapp_message(bot_number, "🤖 Bot reativado para conversa com {sender_number}! Agora estou respondendo normalmente")
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
    reactivation_thread = threading.Thread(target=send_reactivation_message, daemon=True)
    reactivation_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)