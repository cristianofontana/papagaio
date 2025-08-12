import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings

load_dotenv()

collection_name = 'ibuy'

def create_documents(df):
    """Transforma o CSV em documentos para embedding"""
    documents = []
    for _, row in df.iterrows():
        # Se a descrição estiver vazia, usar o item como conteúdo
        content = row['Descricao'] if pd.notna(row['Descricao']) and row['Descricao'] != '' else row['Item']
        
        metadata = {
            'Item': row['Item'],
            'marca': row['marca'],
            'aceita_como_entreda': row['aceita_como_entreda'],
            'preco_novo': row['preco_novo'],
            'preco_semi_novo': row['preco_semi_novo']
        }
        
        documents.append((content, metadata))
    return documents

def main():
    # Carregar dados com delimitador correto
    df = pd.read_csv(f"{collection_name}.csv", delimiter=';', encoding='utf-8-sig')
    
    # Converter preços para string
    df['preco_novo'] = df['preco_novo'].astype(str)
    df['preco_semi_novo'] = df['preco_semi_novo'].astype(str)
    
    # Preencher valores NaN
    df.fillna('', inplace=True)
    
    documents = create_documents(df)
    
    # Conectar ao Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
        ,timeout=120
    )
    
    # Criar collection se não existir
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


    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.Item",
        field_schema="text"
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.preco_semi_novo",
        field_schema="text"
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.marca",
        field_schema="text"
    )

    print("✅ Índice de payload 'Item' criado com sucesso.")

if __name__ == "__main__":
    main()