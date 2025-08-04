from qdrant_client import QdrantClient
from qdrant_client.http import models
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import os 
from dotenv import load_dotenv

load_dotenv()

# Configurações do Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "produtos"  # Nome da coleção que você criou
EMBEDDING_MODEL = "text-embedding-3-small"  # Modelo usado para embeddings

# Inicializar cliente Qdrant
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def gerar_embedding(texto: str) -> list:
    """
    Gera embeddings vetoriais para texto em português
    Retorna uma lista de floats (vetor de 384 dimensões)
    """
    embedding = model.encode(texto, convert_to_tensor=False)
    return embedding.tolist()

def query_celulares(pergunta: str, filtros: dict):
    # Gere o embedding da pergunta (ex: usando Sentence-BERT)
    pergunta_embedding = gerar_embedding(pergunta)  # Implemente essa função
    
    # Construa o filtro dinâmico
    query_filter = models.Filter(**filtros) if filtros else None

    # Execute a busca
    resultados = client.search(
        collection_name="produtos",
        query_vector=pergunta_embedding,
        query_filter=query_filter,
        limit=10
    )
    return resultados

def parse_pergunta(pergunta: str) -> dict:
    pergunta = pergunta.lower()
    filtros = {"must": []}
    
    # Filtro por marca (ex: "xiaomi" ou "samsung")
    marcas = ["apple", "samsung", "xiaomi", "motorola"]
    for marca in marcas:
        if marca in pergunta:
            filtros["must"].append(
                models.FieldCondition(
                    key="marca",
                    match=models.MatchValue(value=marca)
            )
            )
    
    # Filtro por faixa de preço (ex: "até 2000 reais")
    if "preço" in pergunta or "reais" in pergunta:
        valores = re.findall(r'R?\$?\s*(\d+[\.,]?\d*)', pergunta)
        valores = [float(val.replace(',', '.')) for val in valores]
        
        if "até" in pergunta and valores:
            filtros["must"].append(
                models.FieldCondition(
                    key="preco_novo",  # Ou preco_semi_novo
                    range=models.Range(lte=valores[0])
            )
            )
    
    # Filtro por modelo específico (ex: "iPhone 12")
    modelos = ["iphone 12", "redmi note 13 pro"]
    for modelo in modelos:
        if modelo in pergunta:
            filtros["must"].append(
                models.FieldCondition(
                    key="modelo",
                    match=models.MatchValue(value=modelo)
            )
            )
    
    return filtros

def chatbot(pergunta: str):
    # Passo 1: Extrair filtros da pergunta
    filtros = parse_pergunta(pergunta)
    
    # Passo 2: Buscar no Qdrant
    resultados = query_celulares(pergunta, filtros)
    
    # Passo 3: Gerar resposta
    resposta = []
    for item in resultados:
        payload = item.payload
        resposta.append(
            f"{payload['modelo']} ({'Novo' if payload['preco_novo'] else 'Semi-novo'}): "
            f"R${payload.get('preco_novo') or payload.get('preco_semi_novo')}"
        )
    
    return resposta[:3]  # Retorna top 3 resultados

resposta = chatbot("Quais celulares da Xiaomi estão disponíveis até 2000 reais?")  # Exemplo de uso
print(resposta)