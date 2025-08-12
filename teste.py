import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Carregue as variáveis de ambiente
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Inicialize o cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def listar_profiles():
    try:
        response = supabase.table("profiles").select("*").execute()
        if response.data:
            for row in response.data:
                print(row)
        else:
            print("Nenhum registro encontrado na tabela profiles.")
    except Exception as e:
        print(f"Erro ao consultar a tabela profiles: {e}")

if __name__ == "__main__":
    listar_profiles()