import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Conectar ao Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
    ,timeout=120
)

client.create_payload_index(
        collection_name="produtos",
        field_name="metadata.marca",
        field_schema="text"
    )