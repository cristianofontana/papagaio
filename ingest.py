import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def create_iphone_documents(df):
    """Transforma o CSV em documentos para embedding"""
    documents = []
    for _, row in df.iterrows():
        metadata = row.to_dict()
        content = f"""
        Modelo: {row['Modelo']}
        Ano: {row['Ano_Lancamento']}
        Tela: {row['Tamanho_Tela(polegadas)']} polegadas
        Processador: {row['Processador']}
        Câmera: {row['Camera_Traseira(MP)']} MP
        Recursos: {row['Recursos_Especiais']}
        """
        documents.append((content, metadata))
    return documents

def main():
    # Carregar dados
    df = pd.read_csv("iphones.csv")
    documents = create_iphone_documents(df)
    
    # Conectar ao Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Criar collection se não existir
    collection_name = "iphones"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # Tamanho para text-embedding-3-small
                distance=Distance.COSINE
            )
        )
    
    # Gerar embeddings e carregar
    embedder = OpenAIEmbeddings(model='text-embedding-3-small')
    
    points = []
    for idx, (content, metadata) in enumerate(documents):
        embedding = embedder.embed_query(content)
        points.append({
            "id": idx,
            "vector": embedding,
            "payload": {
                "content": content,
                "metadata": metadata
            }
        })
    
    # Upload para Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"✅ {len(points)} documentos carregados no Qdrant")

if __name__ == "__main__":
    main()