import logging
import re
from typing import Any, Dict, List

from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .qdrant_service import query_qdrant
from .settings import settings

logger = logging.getLogger(__name__)


def get_info(history: List[str]) -> str:
    prompt = f"""
    ## TAREFA
    Analise o historico de conversa abaixo e extraia 
    - o *INTERESSE* principal do cliente
    - a *DEMANDA* (se o interesse do cliente e comprar um celular ou outro produto/servico)
    - o *BUDGET/FORMA PAGAMENTO* (valor total que ele tem para comprar o produto, e a forma de pagamento escolhida)
    - a *URGENCIA* (Quando o cliente pretende comprar o produto)
    - *ESTA-PESQUISANDO* (Quando o cliente esta fazendo o orcamento ou pesquisando em outras lojas)

    ## INSTRUCOES
    
    ### INTERESSE
    1. Identifique o produto/servico que o cliente demonstrou interesse. Os servicos incluem: compra, venda, troca, conserto e impressao de documentos.
    2. Seja especifico com modelos quando possivel (ex: "iPhone 15 Pro" em vez de apenas "iPhone").
    3. Se mencionar troca, inclua ambos os aparelhos (ex: "Troca de iPhone X por iPhone 12").
    4. Para consertos, especifique o problema (ex: "Conserto de tela quebrada").
    5. Priorize o interesse MAIS RECENTE.
    6. Para impressao de documentos, especifique o tipo (ex: "Impressao de documentos").
    
    ### DEMANDA
    - Caso o Interesse do cliente seja a COMPRA de um celular retorne o valor "Compra"
    - Caso o Interesse do cliente nao seja a COMPRA de um celular retorne o valor "Outro"

    ### BUDGET/FORMA PAGAMENTO
    - Exemplo com budget e forma de pagamento: "Budget/Forma Pagamento": "5000,00 - Pix" 
    - Exemplo com budget e sem forma de pagamento: "Budget/Forma Pagamento": "5000,00"
    - Exemplo sem budget e sem forma de pagamento : "Budget/Forma Pagamento": "Nao Informado"

    ### URGENCIA
    - Identifique a urgencia do cliente, exemplo: hoje, amanha, semana que vem, mes que vem
    - Se nao houver mençao de valor, retorne: "Nao especificado".
    
    ### ESTA-PESQUISANDO
    - Identifique se o cliente esta pesquisando ou orcando em outro estabelecimento 
    - Exemplo: "ESTA-PESQUISANDO": "Tem orcamento de outra loja, valor: 5200,00"

    ## IMPORTANTE
    - A resposta deve conter apenas o JSON.
    - Nao adicione comentarios, explicacoes ou qualquer outro texto fora do JSON.
    - Certifique-se de que o JSON esta formatado corretamente sem ``` e sem a palavra "json" escrito, apenas as keys, valores e chaves.

    ## HISTORICO
    {history}
    """

    chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    response = chat.invoke(prompt)
    return response.content.strip()


def format_prompt(template: str, format_vars: Dict[str, Any]) -> str:
    formatted = template
    for key, value in format_vars.items():
        placeholder = "{" + key + "}"
        formatted = formatted.replace(placeholder, str(value))
    return formatted


def get_reativacao_prompt(history_str: str, current_message: str) -> str:
    return f"""
    ## CONTEXTO DE REATIVACAO

    Voce esta retomando uma conversa com um lead que estava inativo. O cliente demonstrou interesse anteriormente em produtos Apple/iPhone mas nao finalizou a compra. Agora ele respondeu a uma de suas mensagens de reativacao.

    ## HISTORICO ANTERIOR DA CONVERSA (contexto importante):
    {history_str}

    ## MENSAGEM DE REATIVACAO DO CLIENTE:
    "{current_message}"

    ## INSTRUCOES ESPECIFICAS PARA REATIVACAO:

    ### COMPORTAMENTO NA REATIVACAO:
    1. Reconheca o retorno: "Que bom ver voce de volta!", "Obrigado por retornar!"
    2. Relembre rapidamente o contexto anterior: "Voce estava interessado em [produto mencionado anteriormente]"
    3. Seja mais direto e objetivo: O cliente ja conhece a loja, nao precisa se reapresentar completamente
    4. Foque em resolver obvecoes: Pergunte se ainda tem interesse ou se surgiu alguma duvida
    5. Mantenha o entusiasmo: Mostre que esta feliz com o retorno dele

    ### FLUXO DE QUALIFICACAO NA REATIVACAO:
    1. Confirme o interesse atual: "Voce ainda esta interessado em algum iPhone especifico?"
    2. Verifique mudancas: "Alguma coisa mudou desde nossa ultima conversa?"
    3. Repita qualificacao rapida: 
       - Interesse em modelos especificos
       - Forma de pagamento preferida
       - Urgencia na compra

    ### GATILHOS ESPECIAIS PARA REATIVACAO:
    - "Lembro que voce tinha interesse em [produto], posso verificar as condicoes atuais?"
    - "Temos algumas novidades desde nossa ultima conversa que podem te interessar"
    - "Vou te passar direto para nosso especialista para condicoes exclusivas"

    ### RESTRICOES:
    - NAO repita toda a apresentacao inicial
    - NAO pergunte informacoes que ja tem no historico
    - NAO seja muito formal - use um tom mais descontraido
    - NAO mencione que o cliente "sumiu" ou esteve inativo de forma negativa

    ### ENCERRAMENTO DA REATIVACAO:
    Se o cliente demonstrar interesse renovado, qualifique rapidamente e encaminhe:
    "Perfeito! Vou conectar voce direto com nosso especialista em iPhones para te dar condicoes personalizadas. Um momento!"

    ---

    ## RESPOSTA PARA O CLIENTE (baseada no historico e mensagem atual):
    """


def get_custom_prompt(query: str, history_str: str, intent: str, nome_cliente: str) -> str:
    client_config = settings.get_client_config()
    nome_do_agent = client_config.get("nome_do_agent", "Eduardo")
    nome_da_loja = client_config.get("nome_da_loja", "Nao Informado")
    horario_atendimento = client_config.get("horario_atendimento", "Nao Informado")
    endereco_da_loja = client_config.get("endereco_da_loja", "Nao Informado")
    categorias_atendidas = client_config.get("categorias_atendidas", "Iphone e Acessorios")
    forma_pagamento_iphone = client_config.get(
        "forma_pagamento_iphone", "A vista e cartao em ate 21X"
    )
    forma_pagamento_android = client_config.get(
        "forma_pagamento_android", "A vista, no cartao em ate 21X ou boleto"
    )
    lista_iphone = client_config.get("lista_iphone", "Iphone 11 ate Iphone 16 Pro Max")
    lista_android = client_config.get("lista_android", "Xiaomi, Redmi, Poco")
    msg_abertura_template = client_config.get("msg_abertura", "")
    msg_fechamento_template = client_config.get("msg_fechamento", "")

    msg_abertura = ""
    if msg_abertura_template:
        msg_abertura = msg_abertura_template.format(
            nome_cliente=nome_cliente,
            nome_do_agent=nome_do_agent,
            nome_da_loja=nome_da_loja,
            categorias_atendidas=categorias_atendidas,
        )

    msg_fechamento = ""
    if msg_fechamento_template:
        msg_fechamento = msg_fechamento_template.format(
            horario_atendimento=horario_atendimento
        )

    flow = client_config.get("prompt_text")
    if not flow:
        raise HTTPException(status_code=500, detail="Prompt base não configurado.")

    qdrant_results = query_qdrant(query)

    format_vars: Dict[str, Any] = {
        "nome_do_agent": nome_do_agent,
        "nome_da_loja": nome_da_loja,
        "horario_atendimento": horario_atendimento,
        "endereco_da_loja": endereco_da_loja,
        "categorias_atendidas": categorias_atendidas,
        "forma_pagamento_iphone": forma_pagamento_iphone,
        "forma_pagamento_android": forma_pagamento_android,
        "lista_iphone": lista_iphone,
        "lista_android": lista_android,
        "msg_abertura": msg_abertura,
        "msg_fechamento": msg_fechamento,
        "history_str": history_str,
        "qdrant_results": qdrant_results,
        "query": query,
        "nome_cliente": nome_cliente,
        "intent": intent,
    }

    formatted_prompt = format_prompt(flow, format_vars)
    return f"""
    # Agente Virtual: {nome_do_agent}

    ## Contexto da Conversa

    ### Historico da Conversa
    {history_str}

    ### Base de Conhecimento
    {qdrant_results}

    ## INSTRUCOES PARA O AGENTE
    {formatted_prompt}

    **Mensagem Atual do Cliente:** 
    {query}
    """


def make_answer(prompt: List[Any]) -> AIMessage:
    chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    res = chat.invoke(prompt)
    response_text = res.content.strip()
    return AIMessage(content=response_text)


def detect_intent(text: str) -> str:
    keywords = {
        "compra": ["comprar", "quero", "preciso de"],
        "conserto": ["consertar", "quebrou", "arrumar"],
        "duvida": ["quanto custa", "tem estoque", "garantia"],
    }
    for intent, terms in keywords.items():
        if any(term in text.lower() for term in terms):
            return intent
    return "outros"


def is_stop_request(message: str) -> bool:
    try:
        prompt = f"""
        ## ANALISE DE SOLICITACAO DE INTERRUPCAO

        Analise a mensagem do usuario e determine se ele esta solicitando PARAR 
        de receber mensagens de reativacao ou promover encerramento.

        ## CRITERIOS PARA CONSIDERAR COMO "STOP REQUEST":
        - Pedidos explicitos para parar/envios ("pare", "chega", "stop")
        - Mencao de que ja comprou em outro lugar
        - Solicitacao para nao receber mais mensagens
        - Expressoes de desinteresse final ("nao quero mais", "ja resolvi")
        - Pedidos para ser removido da lista
        - Frases indicando que o assunto esta encerrado

        ## CRITERIOS PARA IGNORAR (NAO E STOP REQUEST):
        - Duvidas sobre produtos
        - Solicitacao de informacoes
        - Mensagens de saudacao/despedida normais
        - Perguntas sobre precos/estoque
        - Expressoes de interesse temporario ("mais tarde", "depois")

        ## MENSAGEM DO USUARIO:
        "{message}"

        ## RESPOSTA:
        Responda APENAS com "true" se for uma solicitacao de parada OU "false" se nao for.
        Nao inclua explicacoes, apenas "true" ou "false".
        """
        chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = chat.invoke(prompt)
        response_content = response.content.strip().lower()
        logger.info("LLM stop request analysis for '%s' -> %s", message, response_content)
        if response_content == "true":
            return True
        if response_content == "false":
            return False
        logger.warning("Resposta inesperada da LLM: %s, usando fallback", response_content)
        return fallback_stop_detection(message)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro na analise LLM de stop request: %s", exc)
        return fallback_stop_detection(message)


def fallback_stop_detection(message: str) -> bool:
    stop_keywords = [
        "pare de enviar",
        "para de mandar",
        "chega de mensagem",
        "nao quero receber",
        "ja comprei",
        "comprei em outro",
        "stop",
        "cancelar",
        "interromper",
        "nao me envie",
        "nao quero mais",
        "basta",
        "suficiente",
        "chega",
        "remover da lista",
        "nao enviar mais",
        "parar mensagens",
        "cancelar envio",
        "nao estou interessado",
        "ja resolvi",
        "nao preciso mais",
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in stop_keywords)


def is_qualification_message(message: str) -> bool:
    chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    prompt = f"""
    ## INSTRUCOES
    Analise a mensagem abaixo e responda APENAS com:
    - "true" se ela indicar que o cliente sera transferido para um humano/vendedor
    - "false" caso contrario

    ## CRITERIOS
    - Termos como "vou chamar um vendedor", "vou te conectar com um especialista"
    - Mensagens que informem que um humano continuara o atendimento
    - Encerramentos que confirmem encaminhamento para equipe humana

    ## MENSAGEM
    {message}

    ## RESPOSTA:
    """
    try:
        response = chat.invoke(prompt)
        return response.content.strip().lower() == "true"
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erro na verificacao de qualificacao: %s", exc)
        return fallback_qualification_check(message)


def fallback_qualification_check(message: str) -> bool:
    patterns = [
        r"vou (notificar|passar|encaminhar|transferir) (para |o )?(vendedor|especialista|humano|gerente|equipe)",
        r"vou (notificar|passar|encaminhar|transferir) (seu contato|o contato)",
        r"vou (chamar|solicitar) (um|o) (vendedor|especialista|humano|gerente)",
        r"transferindo (para|o) (vendedor|especialista|humano|gerente|equipe)",
        r"passando (para|o) (vendedor|especialista|humano|gerente|equipe)",
        r"vamos te conectar",
        r"vou repassar (seu contato|para o time)",
        r"nosso time (vai|ira) entrar em contato",
        r"um (vendedor|especialista|consultor) (vai|ira) entrar em contato",
        r"aguarde um momento (que|enquanto) (vou|irei) (conectar|transferir|encaminhar)",
        r"encaminhamento (para|ao) (vendedor|especialista|humano|gerente|equipe)",
    ]
    for pattern in patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False
