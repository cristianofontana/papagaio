from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import os
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou especifique seu domínio, ex: ["https://preview--post-comment-insight.lovable.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração da API Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Modelo de dados para entrada
class Comentario(BaseModel):
    nome: str
    empresa: str
    descricao: str
    comentario: str
    post: str

# Modelo de dados para saída
class MensagemFormatada(BaseModel):
    nome: str
    msg: str

def humanizar_post_slug(slug: str) -> str:
    """Converte slugs de post em títulos legíveis"""
    words = [word for word in slug.split('-') if word not in ["activity", "estamos", "nosso"]]
    return " ".join(word.capitalize() for word in words)

def extrair_cargo(descricao: str) -> str:
    """Extrai o cargo principal da descrição do perfil"""
    # Remove conteúdos entre parênteses
    descricao = re.sub(r'\([^)]*\)', '', descricao)
    
    # Divide por vírgulas, barras ou '&' e pega a primeira parte
    partes = re.split(r'[,/&]| at | na | no ', descricao)
    cargo = partes[0].strip()
    
    # Remove menções a empresas (palavras com @)
    cargo = re.sub(r'\S*@\S*\s?', '', cargo).strip()
    
    return cargo.split('@')[0].strip()

def gerar_mensagem_groq(comentario: Comentario) -> str:
    """Gera mensagem personalizada usando LLM da Groq"""
    # Prepara o contexto para o modelo
    titulo_post = humanizar_post_slug(comentario.post)
    contexto = f"""
    ## Contexto:
    Você é um especialista em marketing digital respondendo comentários no LinkedIn.
    Precisa gerar uma mensagem personalizada para engajar o usuário baseado em:
    - Nome: {comentario.nome}
    - Empresa: {comentario.empresa}
    - Descrição do perfil: {comentario.descricao}
    - Comentário feito: "{comentario.comentario}"
    - Post onde comentou: "{titulo_post}"

    ## Instruções:
    1. Mensagem deve ser direta e pessoal (começar com "Oi [Nome]")
    2. Demonstrar que leu o comentário e o perfil
    3. Oferecer material relevante baseado no perfil (ex: founder, CTO, gerente)
    4. Usar no máximo 3 emojis
    5. Manter tom profissional mas amigável
    6. Terminar com call-to-action para envio do material
    7. Limite de 2 parágrafos
    """

    # Chama a API Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente de marketing especializado em geração de leads no LinkedIn"
            },
            {
                "role": "user",
                "content": contexto
            }
        ],
        model="llama3-70b-8192",
        temperature=0.7,
        max_tokens=300,
    )

    return chat_completion.choices[0].message.content

@app.post("/formatar-mensagens", response_model=List[MensagemFormatada])
async def formatar_mensagens(comentarios: List[Comentario]):
    mensagens_formatadas = []
    for comentario in comentarios:
        msg = gerar_mensagem_groq(comentario)
        mensagens_formatadas.append({
            "nome": comentario.nome,
            "msg": msg
        })
    return mensagens_formatadas

@app.post("/formatar-mensagens", response_model=List[MensagemFormatada])
async def formatar_mensagens(comentarios: List[Comentario]):
    mensagens_formatadas = []
    for comentario in comentarios:
        msg = gerar_mensagem_groq(comentario)
        mensagens_formatadas.append({
            "nome": comentario.nome,
            "msg": msg
        })
    return mensagens_formatadas

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)