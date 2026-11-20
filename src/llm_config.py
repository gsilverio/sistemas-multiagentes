"""Configuração central dos LLMs (Ollama) usados pelos agentes GOBR.

Variáveis de ambiente: GOBR_LLM_MODEL (padrão gemma3:12b), GOBR_LLM_NUM_CTX
(padrão 4096) e OLLAMA_BASE_URL (padrão http://localhost:11434).
"""

import os


def modelo_padrao() -> str:
    return os.getenv("GOBR_LLM_MODEL", "gemma3:12b")


def contexto_padrao() -> int:
    return int(os.getenv("GOBR_LLM_NUM_CTX", "4096"))


def create_llm(model: str | None = None, *, num_ctx: int | None = None, temperature: float = 0):
    from langchain_ollama import ChatOllama

    nome = model or modelo_padrao()
    extras = {}
    if nome.startswith("qwen3"):
        extras["reasoning"] = False  # Sem bloco <think>: resposta direta.
    return ChatOllama(
        model=nome,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        num_ctx=num_ctx or contexto_padrao(),
        **extras,
    )
