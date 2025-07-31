from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi import FastAPI, HTTPException, Request, Query
import json
import aiohttp
import asyncio

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
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging
import requests 

from supabase import create_client, Client
import os
from datetime import datetime
import threading
from threading import Lock
from typing import Dict, Any, List, Optional, Union

from fastapi.responses import JSONResponse
import spacy
from spacy.matcher import Matcher
from collections import defaultdict

load_dotenv()
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

# Variável global para o buffer
message_buffer = MessageBuffer(timeout=3)

def process_user_message(sender_number: str, message: str, name: str):
    current_intent = detect_intent(message)
    logging.info(f'Intenção detectada: {current_intent}')

    if sender_number not in conversation_history:
        conversation_history[sender_number] = {
            'messages': [],
            'stage': 0,
            'intent': current_intent,
            'bant': {'budget': None, 'authority': None, 'need': None, 'timing': None}
        }
    else:
        conversation_history[sender_number]['messages'].append(HumanMessage(content=message))
    
    history = conversation_history[sender_number]['messages'][-10:]
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    
    prompt = get_custom_prompt(message, history_str, current_intent)
    response = make_answer([SystemMessage(content=prompt)] + history)
    
    conversation_history[sender_number]['messages'].append(response)
    
    if "orcamento" in response.content.lower() or "orçamento" in response.content.lower():
        conversation_history[sender_number]['stage'] = 2
    elif is_qualification_detected(response.content, conversation_history[sender_number]['stage']):
        logging.info(f"Qualificação detectada para {sender_number}")
        interesse = get_interesse(history_str)
        conversation_history[sender_number]['stage'] = 3
        logging.info(f"Lead qualificado: {sender_number} - Intent: {conversation_history[sender_number]['intent']}")
        msg_qualificacao = f"Lead qualificado 🔥: Nome: {name}, Telefone: {sender_number}, Interesse: {interesse}, Compra urgente. Link: https://wa.me/{sender_number}"

        # Consulta no Supabase se já existe
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        query = supabase.table("leadsqualificados").select("telefone").eq("telefone", sender_number)
        result = query.execute()
        ja_qualificado = result.data and len(result.data) > 0

        if not ja_qualificado:
            send_whatsapp_message('120363420079107628@g.us', msg_qualificacao)
            # Grava no Supabase
            supabase.table("leadsqualificados").insert({
                "telefone": sender_number,
                "nome": name,
                "interesse": interesse,
                "data_qualificacao": datetime.now().isoformat()
            }).execute()
    
    logging.info(f'RESPONSE: {response}')
    if response.content.strip() != "`#no-answer`":
        send_whatsapp_message(sender_number, response.content)

def is_qualification_detected(response_text: str, conversation_stage: int) -> bool:
    doc = nlp(response_text.lower())
    
    # 1. Verificação com spaCy Matcher
    if len(matcher(doc)) > 0:
        return True
    
    # 2. Verificação contextual com palavras-chave
    keywords = {
        "lead quente": ["condição especial", "vendedor vai cuidar"],
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

#os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def get_interesse(history: list) -> str:

    prompt = f"""
    ## TASK
    Analise o histórico de conversa abaixo e extraia APENAS o interesse principal do cliente em UMA FRASE CURTA.
    
    ## INSTRUCTIONS
    1. Identifique o produto/serviço que o cliente demonstrou interesse
    2. Seja específico com modelos quando possível (ex: "iPhone 15 Pro" em vez de "iPhone")
    3. Se mencionar troca, inclua ambos aparelhos (ex: "Troca de iPhone X por iPhone 12")
    4. Para consertos, especifique o problema (ex: "Conserto de tela quebrada")
    5. Priorize o interesse MAIS RECENTE
    6. Se não encontrar interesse claro, retorne "Produto não especificado"
    
    ## FORMATO DE RESPOSTA
    A resposta deve conter APENAS UMA DAS OPÇÕES:
    - Nome exato do produto (ex: "iPhone 15 Pro Max 256GB")
    - Tipo de serviço (ex: "Troca de bateria")
    - "Produto não especificado"
    
    ## HISTÓRICO
    {history}
    
    ## RESPOSTA:
    """

    chat = ChatGroq(temperature=0, model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
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
    - iPhones (new or refurbished);
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
    > I can’t help directly with this, but I’ve passed it to our team. They’ll contact you shortly, no worries.  
    - Forward to the *other requests* group and end.

    ### 3. Cross-Authority  
    > How did you hear about us? Was it a referral? I ask because 80% of our sales today come from referrals.

    ### 4. Qualification (BANT)

    *A. Trade-in Intent*  
    > Would you like to trade in your current device?  
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
    4. Did you buy it new or refurbished?  
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
    > Awesome! I’ve notified a salesperson here on WhatsApp. They’ll assist you with a special deal, okay?  
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

    return f"""
    {flow}
    **Histórico Recente:**
    {history_str}

    **Mensagem Atual:**
    {query}
    """

def make_answer(prompt):
    # Criação do modelo de linguagem
    chat = ChatGroq(temperature=0, model="deepseek-r1-distill-llama-70b", api_key=os.getenv("GROQ_API_KEY"))
    res = chat.invoke(prompt)
    
    response_text = res.content
    if '<think>' in response_text:
        try:
            # Extrai apenas o texto após o último </think>
            response_text = response_text.split('</think>')[-1]
        except:
            # Fallback: remove as tags manualmente
            response_text = response_text.replace('<think>', '').replace('</think>', '')
    # Remove quebras de linha no início/fim e espaços extras
    response_text = response_text.strip()
    
    return AIMessage(content=response_text)  # Mantemos o tipo AIMessage

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
    url = "https://saraevo-evolution-api.jntduz.easypanel.host/message/sendText/Cris"
    payload = {
        "number": number,
        "text": text
    }
    headers = {
        "apikey": "D67E3CC26167-473D-947E-8DD27A7807BD",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response


@app.post("/messages-upsert")
async def messages_upsert(request: Request):
    data = await request.json()
    name = data['data']['pushName']
    sender_number = data['data']['key']['remoteJid'].split('@')[0]
    #logging.info(f"MSG RECEIVED: {data}")
    
    
    #if '554196137682' == sender_number:
    #if '120363420079107628@g.us' == sender_number:

    if '120363420079107628' == sender_number:
        logging.info(f"Mensagem de grupo recebida, ignorando")
        return JSONResponse(content={"status": "ignored"}, status_code=200)
    
    elif '554196137682' == sender_number:
        logging.info(f"Received message from {sender_number}: {data['data']['message']['conversation']}")
        message = data['data']['message']['conversation']
        # Adiciona mensagem ao buffer em vez de processar diretamente
        message_buffer.add_message(sender_number, message, name)
    
        return JSONResponse(content={"status": "received"}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)