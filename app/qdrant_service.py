import logging
import os
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from .settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

_qdrant_client: QdrantClient | None = None
if QDRANT_URL and QDRANT_API_KEY:
    _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    logger.warning("Qdrant credentials are not fully configured.")


def query_qdrant(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Consulta o Qdrant e retorna os documentos mais relevantes."""
    if not _qdrant_client:
        logger.warning("Qdrant client not initialised; skipping query.")
        return []

    logger.info("Consultando Qdrant com a query: %s", query)
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        query_embedding = embeddings.embed_query(query)

        results = _qdrant_client.search(
            collection_name=settings.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
        )

        context: List[Dict[str, Any]] = []
        for result in results:
            payload = result.payload or {}
            metadata = payload.get("metadata", {})
            item = metadata.get("Item", "")
            descricao = payload.get("content", "")
            if not descricao and item:
                descricao = f"Produto: {item}"
            context.append(
                {
                    "content": descricao,
                    "item": item,
                    "aceita_como_entrada": metadata.get("aceita_como_entreda", ""),
                    "preco_novo": metadata.get("preco_novo", ""),
                    "preco_semi_novo": metadata.get("preco_semi_novo", ""),
                }
            )
        logger.info("Resultados encontrados: %s", context)
        return context
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro ao consultar Qdrant: %s", exc)
        return []


def is_technical_question(text: str) -> bool:
    """Determina se a pergunta requer consulta ao Qdrant."""
    technical_keywords = [
        "especificacao",
        "tela",
        "camera",
        "processador",
        "memoria",
        "armazenamento",
        "bateria",
        "carregamento",
        "ios",
        "resolucao",
        "peso",
        "dimensao",
        "tamanho",
        "modelo",
        "iphone",
        "comparar",
        "diferenca",
        "qual e o",
        "quanto custa",
        "quais sao os modelos",
        "quais modelos",
        "voces tem",
        "vcis tem",
        "entrada",
        "troca",
        "aceita troca",
        "aceita como entrada",
        "mais novo",
        "novo ou usado",
        "mais memoria",
        "modelos de celular",
        "acessorios",
        "nao sejam iphone",
        "outros modelos",
        "qual iphone",
        "tem estoque",
        "estoque",
        "disponivel",
        "caracteristicas",
        "especificacoes",
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in technical_keywords)
