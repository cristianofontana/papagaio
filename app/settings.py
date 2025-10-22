import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables and configure logging as early as possible
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application-level settings and client configuration."""

    openai_api_key: str = field(init=False, default="")
    evolution_api_key: str = field(init=False, default="")
    evolution_server_url: str = field(init=False, default="")
    cliente_evo: str = field(init=False, default="")
    client_id: str = field(init=False, default="")
    verificar_lead_qualificado: bool = field(init=False, default=True)
    history_expiration_minutes: int = field(init=False, default=180)
    supabase: Client = field(init=False)
    client_config: Dict[str, Any] = field(init=False, default_factory=dict)
    collection_name: str = field(init=False, default="")
    authorized_numbers: List[str] = field(init=False, default_factory=list)
    ignored_groups: Dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._load_env()
        self._init_supabase()
        self.reload_client_config()
        self._init_runtime_defaults()

    def _load_env(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.evolution_api_key = os.getenv("EVO_API_KEY", "")
        self.evolution_server_url = os.getenv(
            "EVOLUTION_SERVER_URL",
            "https://saraevo-evolution-api.jntduz.easypanel.host/",
        )
        self.cliente_evo = os.getenv("EVOLUTION_INSTANCE", "Five")
        self.client_id = os.getenv("CLIENT_ID", "five_store")
        self.verificar_lead_qualificado = (
            os.getenv("VERIFY_LEAD_QUALIFIED", "true").lower() == "true"
        )
        self.history_expiration_minutes = int(
            os.getenv("HISTORY_EXPIRATION_MINUTES", "180")
        )
        # Ensure downstream libraries see the OpenAI key
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def _init_supabase(self) -> None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise RuntimeError("Supabase credentials are not configured.")
        self.supabase = create_client(supabase_url, supabase_key)

    def _load_client_config(self) -> Dict[str, Any]:
        try:
            response = (
                self.supabase.table("client_config")
                .select("*")
                .eq("client_id", self.client_id)
                .limit(1)
                .execute()
            )
            if not response.data:
                logger.error("Configuração não encontrada para cliente: %s", self.client_id)
                return {}

            config = response.data[0]
            prompt_id = config.get("prompt_id")
            prompt_text = None
            if prompt_id:
                prompt_response = (
                    self.supabase.table("prompts")
                    .select("prompt_text")
                    .eq("id", prompt_id)
                    .limit(1)
                    .execute()
                )
                if prompt_response.data:
                    prompt_text = prompt_response.data[0].get("prompt_text")

            return {
                "nome_do_agent": config.get("nome_do_agent", "Agente"),
                "nome_da_loja": config.get("nome_da_loja", "Loja"),
                "horario_atendimento": config.get("horario_atendimento", "Seg a Sex 9:00-18:00"),
                "endereco_da_loja": config.get("endereco_da_loja", "Endereco nao especificado"),
                "categorias_atendidas": config.get("categorias_atendidas", "Produtos em geral"),
                "lugares_que_faz_entrega": config.get("lugares_que_faz_entrega", ""),
                "forma_pagamento_iphone": config.get("forma_pagamento_iphone", "À vista ou parcelado"),
                "forma_pagamento_android": config.get("forma_pagamento_android", "À vista ou parcelado"),
                "collection_name": config.get("collection_name", "default_collection"),
                "authorized_numbers": config.get("authorized_numbers", []),
                "group_id": config.get("id_grupo_cliente", ""),
                "lista_iphone": config.get("lista_iphone", "iPhone 11 até iPhone 16 Pro Max"),
                "lista_android": config.get("lista_android", "Xiaomi, Redmi, Poco"),
                "msg_abertura": config.get("msg_abertura", ""),
                "msg_fechamento": config.get("msg_fechamento", ""),
                "prompt_text": prompt_text,
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Erro ao carregar configuração do cliente: %s", exc)
            return {}

    def reload_client_config(self) -> Dict[str, Any]:
        self.client_config = self._load_client_config()
        self.collection_name = self.client_config.get("collection_name", "default_collection")
        self.authorized_numbers = [
            number for number in self.client_config.get("authorized_numbers", []) if number
        ]
        self.ignored_groups = {
            "120363420079107628@g.us": "Grupo Admin",
        }
        return self.client_config

    def get_client_config(self) -> Dict[str, Any]:
        if not self.client_config:
            return self.reload_client_config()
        return self.client_config


settings = Settings()
