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
import threading
from threading import Lock
from typing import Dict, Any, List, Optional, Union

from fastapi.responses import JSONResponse
import spacy
from spacy.matcher import Matcher
from collections import defaultdict

load_dotenv()
HISTORY_EXPIRATION_MINUTES = 5 

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

for pattern in patterns:
    matcher.add("TRANSFER_PATTERNS", [pattern])



class IPhoneSalesAssistant:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.skills = {
            'list_models': self.list_available_models,
            'compare': self.compare_models,
            'model_details': self.get_model_details,
            'price_info': self.get_price_info,
            'recommend': self.recommend_model
        }
    
    def handle_query(self, query: str) -> str:
        """Roteia a consulta para a função apropriada"""
        # Detecção de intenção aprimorada
        query_lower = query.lower()
        
        if "modelos" in query_lower or "tipos" in query_lower or "opções" in query_lower:
            return self.skills['list_models']()
        
        elif "comparar" in query_lower or "diferença" in query_lower or "mais" in query_lower or "melhor" in query_lower:
            models = self.extract_model_names(query)
            if len(models) >= 2:
                return self.skills['compare'](models)
            else:
                return "Por favor, especifique pelo menos dois modelos para comparar."
        
        elif "detalhes" in query_lower or "especificações" in query_lower or "fale sobre" in query_lower:
            models = self.extract_model_names(query)
            if models:
                return self.skills['model_details'](models[0])
            else:
                return "Qual modelo você gostaria de conhecer?"
        
        elif "preço" in query_lower or "quanto custa" in query_lower or "valor" in query_lower:
            models = self.extract_model_names(query)
            if models:
                return self.skills['price_info'](models[0])
            else:
                return "Para qual modelo você gostaria de saber o preço?"
        
        elif "recomendar" in query_lower or "sugerir" in query_lower or "indicar" in query_lower:
            criteria = self.extract_recommendation_criteria(query)
            return self.skills['recommend'](criteria)
        
        return None  # Nenhuma skill aplicável
    
    def extract_model_names(self, query: str) -> list:
        """Extrai nomes de modelos da consulta"""
        # Padrão para identificar modelos de iPhone
        pattern = r'\b(iPhone\s*\d{1,2}\s*(?:Pro\s*Max|Pro|Plus|mini|e)?)\b'
        matches = re.findall(pattern, query, re.IGNORECASE)
        
        # Normalização dos nomes
        normalized = []
        for match in matches:
            # Padroniza o nome: primeira letra maiúscula e remove espaços extras
            parts = [p.capitalize() for p in match.split()]
            if parts[0].lower() == 'iphone':
                parts[0] = 'iPhone'
            normalized.append(' '.join(parts))
        
        return list(set(normalized))  # Remove duplicatas
    
    def extract_recommendation_criteria(self, query: str) -> dict:
        """Extrai critérios de recomendação da consulta"""
        criteria = {
            'budget': None,
            'use_case': None,
            'screen_size': None,
            'priority': 'Custo-benefício'
        }
        
        # Detecção de orçamento
        money_match = re.search(r'R\$\s*(\d+[\.,]?\d*)', query)
        if money_match:
            criteria['budget'] = money_match.group(0)
        
        # Detecção de casos de uso
        use_cases = {
            'fotos': 'Fotografia',
            'filmes': 'Vídeos',
            'jogos': 'Jogos',
            'trabalho': 'Produtividade',
            'social': 'Redes Sociais'
        }
        for key, value in use_cases.items():
            if key in query.lower():
                criteria['use_case'] = value
                break
        
        # Detecção de tamanho de tela
        size_match = re.search(r'(\d+[\.,]?\d*)\s*["”]|polegadas', query)
        if size_match:
            criteria['screen_size'] = size_match.group(1) + '"'
        
        return criteria
    
    def list_available_models(self) -> str:
        """Lista modelos disponíveis agrupados por série"""
        grouped_models = self.rag.get_all_models()
        if not grouped_models:
            return "Desculpe, não consegui recuperar a lista de modelos no momento."
        
        response = "Temos várias opções disponíveis:\n\n"
        for series, variants in grouped_models.items():
            variants_str = ", ".join([v for v in variants if v])
            response += f"*{series}*: {variants_str}\n"
        
        response += "\nQual série te interessa mais?"
        return response
    
    def compare_models(self, model_names: list) -> str:
        """Compara modelos e retorna diferenças principais"""
        comparison = self.rag.compare_models(model_names)
        if not comparison:
            return "Desculpe, não consegui comparar esses modelos."
        
        # Selecionar características mais relevantes para comparação
        features = [
            'Ano_Lancamento', 
            'Tamanho_Tela(polegadas)', 
            'Processador',
            'RAM(GB)',
            'Camera_Traseira(MP)',
            'Bateria(mAh)'
        ]
        
        response = "Aqui estão as principais diferenças:\n\n"
        for feature in features:
            values = []
            for model, specs in comparison.items():
                value = specs.get(feature, "N/A")
                # Simplificar valores complexos
                if "Camera_Traseira" in feature and "+" in str(value):
                    value = f"{value.split('+')[0]}MP + outras"
                values.append(f"{model}: {value}")
            
            feature_name = feature.replace('_', ' ').split('(')[0]
            response += f"• *{feature_name}*: {', '.join(values)}\n"
        
        return response
    
    def get_model_details(self, model_name: str) -> str:
        """Retorna detalhes completos de um modelo específico"""
        specs = self.rag.get_model_specs(model_name)
        if not specs:
            return f"Desculpe, não encontrei informações sobre o {model_name}."
        
        # Selecionar campos mais relevantes
        relevant_fields = [
            'Ano_Lancamento',
            'Tamanho_Tela(polegadas)',
            'Resolucao_Tela',
            'Processador',
            'RAM(GB)',
            'Armazenamento(GB)',
            'Camera_Traseira(MP)',
            'Camera_Frontal(MP)',
            'Bateria(mAh)',
            'Recursos_Especiais'
        ]
        
        response = f"📊 *Especificações do {model_name}:*\n\n"
        for field in relevant_fields:
            value = specs.get(field, "N/A")
            if value != "N/A":
                field_name = field.replace('_', ' ').split('(')[0]
                response += f"• *{field_name}*: {value}\n"
        
        return response
    
    def get_price_info(self, model_name: str) -> str:
        """Retorna informações de preço (simulado)"""
        # Em uma implementação real, isso viria de uma base de dados
        price_ranges = {
            "iPhone 16 Pro Max": "R$ 9.000 - R$ 12.000",
            "iPhone 16 Pro": "R$ 8.000 - R$ 10.000",
            "iPhone 16": "R$ 6.000 - R$ 8.000",
            "iPhone 15 Pro Max": "R$ 7.000 - R$ 9.500",
            "iPhone 15 Pro": "R$ 6.500 - R$ 8.500",
            "iPhone 15": "R$ 5.000 - R$ 7.000",
            "iPhone 14": "R$ 4.000 - R$ 5.500",
            "iPhone 13": "R$ 3.500 - R$ 4.500",
            "iPhone 12": "R$ 2.800 - R$ 3.800",
            "iPhone 11": "R$ 2.000 - R$ 2.800",
            "iPhone SE": "R$ 1.800 - R$ 2.500"
        }
        
        # Encontrar melhor correspondência
        for model_pattern, price in price_ranges.items():
            if model_name.lower() in model_pattern.lower():
                return f"O *{model_pattern}* está na faixa de preço: {price}\n\nValores variam conforme a capacidade de armazenamento."
        
        return f"Preço para o {model_name} não disponível no momento. Posso verificar com nosso time?"
    
    def recommend_model(self, criteria: dict) -> str:
        """Recomenda um modelo com base nos critérios"""
        return self.rag.recommend_model(criteria)


# Adicione esta classe antes da definição do app
class MessageBuffer:
    def __init__(self, timeout=2):
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

class IPhoneRAGSystem:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.3
        )
        
        self.embedder = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        self.qdrant = qdrant_client.QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.model_cache = None
        self.last_cache_update = 0
    
    def get_all_models(self) -> dict:
        """Retorna todos os modelos agrupados por série"""
        if self.model_cache and time.time() - self.last_cache_update < 3600:  # Cache por 1 hora
            return self.model_cache
            
        try:
            results = self.qdrant.scroll(
                collection_name="iphones",
                limit=100
            )
            
            models = set()
            for result in results[0]:
                model_name = result.payload.get("metadata", {}).get("Modelo", "")
                if model_name:
                    models.add(model_name)
            
            # Agrupar por série
            grouped_models = defaultdict(list)
            for model in sorted(models, reverse=True):
                if "iPhone" not in model:
                    continue
                parts = model.split()
                series = f"{parts[0]} {parts[1]}"
                variant = " ".join(parts[2:]) if len(parts) > 2 else "Standard"
                grouped_models[series].append(variant)
            
            self.model_cache = grouped_models
            self.last_cache_update = time.time()
            return grouped_models
            
        except Exception as e:
            logging.error(f"Erro ao obter modelos: {str(e)}")
            return {}
    
    def get_model_specs(self, model_name: str) -> dict:
        """Obtém especificações completas de um modelo"""
        try:
            results = self.qdrant.scroll(
                collection_name="iphones",
                scroll_filter=qdrant_client.models.Filter(
                    must=[qdrant_client.models.FieldCondition(
                        key="metadata.Modelo", 
                        match=qdrant_client.models.MatchValue(value=model_name)
                    )]
                ),
                limit=1
            )
            if results[0]:
                return results[0][0].payload.get("metadata", {})
            return {}
        except Exception as e:
            logging.error(f"Erro ao obter specs para {model_name}: {str(e)}")
            return {}
    
    def compare_models(self, model_names: list, features: list = None) -> dict:
        """Compara múltiplos modelos em características específicas"""
        comparison = {}
        for model in model_names:
            specs = self.get_model_specs(model)
            if specs:
                comparison[model] = specs
        
        if features:
            return {model: {feature: specs.get(feature, "N/A") for feature in features} 
                    for model, specs in comparison.items()}
        return comparison
    
    def recommend_model(self, criteria: dict) -> str:
        """Recomenda um modelo com base em critérios"""
        try:
            # Converter critérios para prompt
            prompt = f"""Recomende um iPhone com base nestes critérios:
            - Orçamento: {criteria.get('budget', 'Qualquer')}
            - Uso principal: {criteria.get('use_case', 'Geral')}
            - Tamanho de tela: {criteria.get('screen_size', 'Qualquer')}
            - Prioridade: {criteria.get('priority', 'Custo-benefício')}
            
            Escolha entre os modelos disponíveis e justifique brevemente."""
            
            return self.llm.invoke(prompt).content
            
        except Exception as e:
            logging.error(f"Erro na recomendação: {str(e)}")
            return "Não consegui fazer uma recomendação no momento."
    
    # Adicionar o método retrieve_relevant_docs na classe principal
    def retrieve_relevant_docs(self, query: str, k=6) -> str:
        """Recupera documentos relevantes do Qdrant"""
        try:
            query_embedding = self.embedder.embed_query(query)
            
            results = self.qdrant.search(
                collection_name="iphones",
                query_vector=query_embedding,
                limit=k
            )
            
            context = "## Informações Técnicas Relevantes:\n"
            for i, result in enumerate(results):
                content = result.payload.get("content", "")
                metadata = result.payload.get("metadata", {})
                
                # Formata as informações técnicas
                tech_info = "\n".join([f"- {key}: {value}" for key, value in metadata.items()])
                context += f"\n### Documento {i+1}:\n{tech_info}\n"
            
            return context
        
        except Exception as e:
            logging.error(f"Erro na recuperação RAG: {str(e)}")
            return ""

# Instância global do sistema RAG
rag_system = IPhoneRAGSystem()

def extract_entities(text: str) -> dict:
    """Extrai entidades relevantes usando spaCy"""
    doc = nlp(text)
    entities = {
        "modelos": [],
        "caracteristicas": [],
        "precos": []
    }
    
    # Padrões para modelos de iPhone
    iphone_pattern = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]
    matcher.add("IPHONE_MODEL", [iphone_pattern])
    matches = matcher(doc)
    
    for match_id, start, end in matches:
        span = doc[start:end]
        entities["modelos"].append(span.text)
    
    # Características técnicas
    tech_terms = ["tela", "câmera", "memória", "bateria", "processador"]
    for token in doc:
        if token.lemma_ in tech_terms:
            entities["caracteristicas"].append(token.lemma_)
    
    # Menções a preços
    money_pattern = [{"LIKE_NUM": True}, {"TEXT": {"REGEX": "r\$|reais"}}]
    matcher.add("MONEY", [money_pattern])
    money_matches = matcher(doc)
    
    for match_id, start, end in money_matches:
        span = doc[start:end]
        entities["precos"].append(span.text)
    
    return entities

def is_technical_question(text: str) -> bool:
    """Determina se a pergunta é técnica e requer consulta RAG"""
    technical_keywords = [
        'especificação', 'tela', 'câmera', 'processador', 'memória', 'armazenamento', 
        'bateria', 'carregamento', 'ios', 'resolução', 'peso', 'dimensão', 'tamanho',
        'modelo', 'iphone', 'comparar', 'diferença', 'qual é o', 'quanto custa'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in technical_keywords)

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

########################################################################## FIM RAG SYSTEM #######################################################################################

# Variável global para o buffer
message_buffer = MessageBuffer(timeout=3)
sales_assistant = IPhoneSalesAssistant(rag_system)

def process_user_message(sender_number: str, message: str, name: str):
    # Primeiro tente usar uma skill especializada
    skill_response = sales_assistant.handle_query(message)
    
    if skill_response:
        # Se uma skill respondeu, use essa resposta
        response_content = skill_response
        # Adiciona ao histórico como AIMessage
        if sender_number in conversation_history:
            conversation_history[sender_number]['messages'].append(AIMessage(content=response_content))
        
        # Envia a resposta
        if response_content.strip() != "#no-answer":
            send_whatsapp_message(sender_number, response_content)
        return
    
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

########################################################################## INICIO LLM ###############################################################################################

client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

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
    # Guidelines for the Virtual Agent "Felipe"

    ## 🎯 Role and Mission
    You are *Felipe, the virtual agent of the store serving customers via **WhatsApp***. Your mission is:
    - Welcome customers with enthusiasm and naturalness;
    - Act as an *excited friend*, without sounding salesy or robotic;
    - Qualify customers subtly using the *BANT framework*;
    - Forward qualified leads to the human team;
    - Redirect other requests (like repairs or Android) to the correct group;
    - *Never* resume the conversation after forwarding.
    - Use the history to avoid repeating questions or information already provided by the customer.

    **Intent Detected:** {intent.upper()}
    
    **Specific Instructions:**
    {f"Focus on quick budgeting" if intent == 'buy' else ""}
    {f"Prioritize technical details" if intent == 'repair' else ""}
    {f"Offer model comparison" if intent == 'question' else ""}

    ## 🗣️ Tone and Style
    - Use *natural, direct, and mature language* **in Portuguese**;
    - **ALWAYS respond in Portuguese, even if the customer uses another language**;
    - Convey enthusiasm with *moderation* (e.g., “Heyyy!”, “Maaan!”);
    - Never use *emojis, gifs, or stickers*;
    - Do not compliment devices or force jokes;
    - Never reveal you’re a bot or AI;
    - Ask *one question at a time* and keep messages *short*;
    - Avoid repetitive phrases like “tell me,” “share something”;
    - Use *“my friend” at most once per conversation*;
    - *Never send long messages* — split into short blocks.

    ## 📌 Context
    - Customers typically seek:
    - iPhones (new or pre-owned);
    - Trading their own device;
    - Cases or accessories;
    - Phone repairs;
    - All customers already own a phone;
    - The store has a database called *products* containing:
    - iPhones available in stock;
    - Models accepted as trade-ins;
    - Anything outside this database is *another request*.

    ## ✅ Conversation Flow

    ### 1. Opening
    Never use "good morning," "good afternoon," "good evening," or assume gender.  
    Example:  
    > Heyy! How can I help you today?

    ### 2. Need Identification  
    *Ask what the customer is looking for.*  
    If purchasing a phone:  
    - Confirm:  
    > You’re looking for an iPhone, right?  
    If *not an iPhone*:  
    > Make clear the products sold, and answer the question

    ### 3. Cross-Authority  
    > How did you hear about us? Was it a referral? I ask because 80% of our sales today come from referrals.

    ### 4. Qualification (BANT)

    *A. Trade-in Intent*  
    > Would you like to give your cell phone as a deposit?
    - If *no*:  
    - Skip device questions;  
    - Proceed:  
        > Have you researched elsewhere?  
    - If *yes*:  
    1. What’s your current device?  
        - If *not accepted*, say:  
        > Unfortunately we can’t accept this model as a trade-in, but let’s explore other options for you?  
        - Never revisit the device topic.  
    2. Want to use it as partial payment?  
    3. What’s its battery health?  
    4. Did you buy it new or pre-owned?  
    5. Has it been opened or shows signs of use?  
    6. Have you researched elsewhere?  
    > All used iPhones must be evaluated before we accept them in a trade, alright?

    *B. Budget*  
    > What price range do you have in mind for this device?

    *C. Urgency*  
    > When are you planning to buy?  
    - If *no rush*, say:  
    > The dollar is rising, so prices might increase in the coming weeks.

    ### 5. Stock Check  
    Never say “I’ll check.”  
    Example:  
    > I see we have 256GB available for this model.  
    If the exact model is unavailable, suggest similar options within budget.

    ### 6. Price Request  
    Never quote prices.  
    > I’ll tell you shortly. Just one thing...  
    and follow the flow.

    ### 7. Justification  
    Before forwarding to a manager:  
    > I asked these questions because we’re very selective about what we sell. We need to know exactly what we’re getting because if issues arise later, we bear the responsibility. And with our profit margins... we can’t slip up.

    ### 8. Forwarding to *Hot Lead*  
    Say:  
    > I’ve notified a salesperson here on WhatsApp. They’ll assist you with a special deal, okay?  
    Use the *Send to Hot Leads Group* tool with:  

    Qualified Lead 🔥:  
    Name: Fulano,  
    Phone: 551999000000,  
    Interest: iPhone 13 128GB,  
    Trade-in: No,  
    Budget: R$3,500,  
    Urgent purchase.  
    Link: https://wa.me/551999000000

    ### 9. Forwarding to *Other Requests*  
    Say:  
    > Great! I’ve notified a specialist here on WhatsApp. They’ll handle your request, okay?  
    Use the *Send to Other Requests Group* tool with:  

    Other requests:  
    Name: Fulano,  
    Phone: 551999000000,  
    Interest: Buy Macbook,  
    Link: https://wa.me/551999000000

    ## 🛑 Closing and Post-Forwarding Rules  
    After sending to any group:  
    - Never message again;  
    - Never return to the support flow;  
    - Never recheck stock;  
    - Never react or continue the conversation;  
    - If the client replies with “ok,” “thanks,” etc., respond only with:  

    #no-answer

    ## 🧠 Special Situations  

    *Receiving Audio or Photos*  
    > I can’t open photos or audio here. Can you send it in writing, please?  

    *Photo Request*  
    > I can’t send photos here. The salesperson will send them once they contact you.  
    Continue the conversation.

    ***Customer in Another Language***  
    > I'll keep assisting in Portuguese for better support. You can speak your language and I'll understand, but my responses will be in PT-BR, okay?

    ## 🚫 Prohibited Actions  
    - Never awnser in english or any other language;
    - Never quote prices;  
    - Never say “I can’t help”;  
    - Never say "only in person";  
    - Never redirect clients to another store;  
    - Never reveal you’re an AI;  
    - Never activate the forwarding tool more than once;  
    - Never compliment the client’s device;  
    - Never use emojis;  
    - Never use forced language or caricatures;  
    - Never ask multiple questions together;  
    - Never leave clients waiting for a response that won’t come.  

    ---  

    By following these guidelines, you’ll be a reliable, consultative, and efficient agent. Good luck out there, Felipe!  

    """

    # Adicionar contexto RAG se for uma pergunta técnica
    additional_context = ""
    if is_technical_question(query):
        rag_context = rag_system.retrieve_relevant_docs(query)
        logging.info(f"RAG Context Retrieved: {rag_context}")
        if rag_context:
            additional_context = f"""
            ## 📚 Base de Conhecimento iPhone:
            {rag_context}
            
            IMPORTANTE: Use estas informações para responder perguntas técnicas sobre modelos de iPhone.
            Não invente especificações - se não encontrar na base, diga que não sabe.
            """
    
    skill_section = """
    ## 🛠 Specialized Skills Available
    
    You have access to specialized functions for:
    - Listing available models
    - Comparing technical specifications
    - Detailing features of a specific model
    - Providing price ranges
    - Recommending models based on needs
    
    **Always prefer using these functions when appropriate, as they provide accurate and structured responses.
    """
    
    return f"""
    {flow}

    {skill_section}

    {additional_context}
    
    **Histórico Recente:**
    {history_str}

    **Mensagem Atual:**
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
        "apikey": os.getenv("EVO_API_KEY"),
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response


@app.post("/messages-upsert")
async def messages_upsert(request: Request):
    data = await request.json()
    full_jid = data['data']['key']['remoteJid']
    msg_type = data['data']['messageType']

    if msg_type == 'imageMessage':
        send_whatsapp_message(full_jid, "Desculpe, não consigo abrir imagens. Por favor, envie a mensagem em texto.")
        return JSONResponse(content={"status": "image ignored"}, status_code=200)

    name = data['data']['pushName']
    
    sender_number = full_jid.split('@')[0]
    message = data['data']['message']['conversation']

    bot_sender = data['sender']
    bot_number = bot_sender.split('@')[0]

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
    
    # Se o bot estiver inativo, ignorar mensagens
    with bot_state_lock:
        if not bot_active_per_chat[sender_number]:
            logger.info(f"Ignorando mensagem de {sender_number} - Bot inativo para este número")
            return JSONResponse(content={"status": f"ignored - bot inactive for {sender_number}"}, status_code=200)
    
    # Adiciona mensagem ao buffer
    #message_buffer.add_message(full_jid, message, name)  # Alterado para usar full_jid

    #return JSONResponse(content={"status": "received"}, status_code=200)

    logging.info(f"Received message from {full_jid}: {data['data']['message']}")
    if 'imageMessage' in data['data']['message']:
        send_whatsapp_message(full_jid, "Desculpe, não consigo abrir imagens. Por favor, envie a mensagem em texto.")
    else:
        message = data['data']['message']['conversation']
        # Adiciona mensagem ao buffer em vez de processar diretamente
        message_buffer.add_message(full_jid, message, name)

    return JSONResponse(content={"status": "received"}, status_code=200)

if __name__ == "__main__":
    cleanup_thread = threading.Thread(target=cleanup_expired_histories, daemon=True)
    cleanup_thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)