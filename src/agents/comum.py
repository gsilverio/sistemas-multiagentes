"""Utilidades compartilhadas pelos agentes com LLM."""

from __future__ import annotations

import json
import re
from typing import Any


def texto_da_resposta(resposta: Any) -> str:
    """Conteúdo textual de uma resposta do LangChain (ou de uma string), sem `<think>`."""
    texto = getattr(resposta, "content", resposta)
    if isinstance(texto, list):
        texto = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in texto)
    return re.sub(r"<think>.*?</think>", "", str(texto), flags=re.DOTALL).strip()


def extrair_json(texto: str) -> Any:
    """Primeiro objeto JSON do texto (aceita bloco ```json ... ```)."""
    bloco = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, flags=re.DOTALL)
    if bloco:
        return json.loads(bloco.group(1))
    inicio = texto.find("{")
    if inicio < 0:
        raise ValueError("Nenhum objeto JSON na resposta.")
    objeto, _ = json.JSONDecoder().raw_decode(texto[inicio:])
    return objeto
