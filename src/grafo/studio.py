"""Grafo exposto ao LangGraph Studio (`langgraph.json` aponta para `grafo`).

O servidor do Studio (`langgraph dev`) tem a própria memória por thread, então
o grafo é compilado sem checkpointer. A entrada é um formulário simples:
`pedido` (texto) e `anexo` (caminho de um PDF/imagem, opcional).

Variáveis de ambiente:
    GOBR_ORQUESTRADOR_LLM=0   orquestrador só com as regras (padrão: 1)
    GOBR_EXPLICAR_LLM=1       explicações com o LLM (padrão: 0, rascunho)
    OLLAMA_BASE_URL, GOBR_LLM_MODEL, GOBR_MODELO_*   como no resto do projeto
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:  # o servidor carrega este arquivo pelo caminho
    sys.path.insert(0, str(RAIZ))

from src.grafo.estado import Agentes  # noqa: E402
from src.grafo.grafo import construir_grafo  # noqa: E402


def _flag(nome: str, padrao: bool) -> bool:
    return os.getenv(nome, "1" if padrao else "0").strip().lower() in ("1", "true", "sim", "yes")


grafo = construir_grafo(
    Agentes.padrao(orquestrador_com_llm=_flag("GOBR_ORQUESTRADOR_LLM", True),
                   explicar_com_llm=_flag("GOBR_EXPLICAR_LLM", False), interativo=True),
    entrada_simples=True,
)
