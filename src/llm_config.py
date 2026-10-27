"""Configuração central dos LLMs utilizados nos agentes GOBR."""

import os

from langchain_ollama import ChatOllama


def create_llm() -> ChatOllama:
    """Instancia o mesmo modelo e parâmetros para os dois agentes."""
    return ChatOllama(
        model=os.getenv("GOBR_LLM_MODEL", "gemma3:12b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        num_ctx=int(os.getenv("GOBR_LLM_NUM_CTX", "4096")),
    )
