"""Chaves que agrupam blocos `parametros` parecidos.

- `chave_corpo`: mesmo corpo, só o nome no cabeçalho muda (1.700 blocos do
  acervo → 1.204 corpos distintos).
- `assinatura_estrutural`: mesma *forma* de regra, com números e valores
  entre aspas trocados por marcadores (1.204 corpos → ~470 formas). Serve
  para deduplicar exemplos no RAG e para amostrar o acervo.
"""

from __future__ import annotations

import hashlib
import re

from src.gobr.ast import GobrParsedRule
from src.gobr.grammar import normalizar
from src.gobr.vocabulary import modelo_condicao


def chave_corpo(gobr: str) -> str:
    corpo = gobr.strip().splitlines()[1:]
    return hashlib.sha256(normalizar(" ".join(corpo)).encode("utf-8")).hexdigest()


def _modelo(texto: str) -> str:
    texto = normalizar(texto)
    texto = re.sub(r'"[^"]*"', '"X"', texto)
    return re.sub(r"\d+(?:[.,]\d+)?", "N", texto)


def assinatura_estrutural(regra: GobrParsedRule) -> str:
    partes = [
        "escopo:" + ",".join(f"{e.tipo}{'/' + e.operador if e.operador else ''}" for e in regra.escopos),
        "quando:" + " | ".join(f"{c.operador or ''} {modelo_condicao(c.texto)}".strip() for c in regra.condicoes),
        "acao:" + ("nao agendar" if regra.nao_agendar else _modelo(regra.frequencia or "")),
        "mods:" + " | ".join(_modelo(m) for m in regra.modificadores),
        "filtros:" + " | ".join(_modelo(f) for f in regra.filtros_temporais),
    ]
    return "\n".join(partes)


def construcoes(regra: GobrParsedRule) -> list[str]:
    """Rótulos das construções da linguagem usadas pelo bloco (para filtros e métricas)."""
    rotulos = [f"escopo_{e.tipo}" for e in regra.escopos]
    if regra.condicoes:
        rotulos.append("condicao")
    if regra.nao_agendar:
        rotulos.append("nao_agendar")
    elif regra.frequencia:
        freq = normalizar(regra.frequencia)
        rotulos.append("frequencia_dinamica" if re.search(r"\b(?:do|de qualquer|govendas|historico)\b", freq)
                       else "frequencia_fixa")
    for modificador in regra.modificadores:
        palavra = normalizar(modificador).split()[0]
        rotulos.append(f"modificador_{palavra}")
    if regra.filtros_temporais:
        rotulos.append("filtro_temporal")
    return list(dict.fromkeys(rotulos))
