"""Orquestrador supervisor: decide o próximo agente.

Duas camadas:

1. **Guardrails** (`destinos_permitidos`): pré-condições em código. Ex.: só
   gera GOBR com contexto recuperado e sem pergunta bloqueante aberta; só
   explica depois da validação; só finaliza com explicação pronta, erro ou
   impasse. Também dão a sugestão determinística (`sugestao`).
2. **LLM** (opcional): escolhe entre os destinos permitidos a partir de um
   resumo do estado (contagens e flags, nunca o conteúdo dos agentes). Só é
   chamado quando há mais de uma opção válida, por exemplo: perguntar ao
   usuário ou seguir com as suposições. Escolha inválida ou resposta
   ilegível → vale a sugestão, e o `trace` registra a correção.
"""

from __future__ import annotations

from typing import Any

from src.agents.comum import extrair_json, texto_da_resposta
from src.grafo.estado import Agentes, Estado
from src.prompts import orquestrador_prompt

AGENTES = ("percepcao", "analista", "perguntar_usuario", "recuperador", "gerador", "validador", "explicador",
           "finalizar")


def perguntas_abertas(e: Estado) -> list[dict]:
    return list((e.get("analise") or {}).get("perguntas") or [])


def bloqueantes_abertas(e: Estado) -> list[dict]:
    return [p for p in perguntas_abertas(e) if p.get("bloqueante")]


def a_perguntar(e: Estado) -> list[dict]:
    """Bloqueantes podem ser repetidas; as demais são feitas uma vez só (depois viram suposição)."""
    feitas = set(e.get("perguntas_feitas") or [])
    return [p for p in perguntas_abertas(e) if p.get("bloqueante") or p["id"] not in feitas]


def pode_perguntar(e: Estado, agentes: Agentes) -> bool:
    return agentes.interativo and bool(a_perguntar(e)) and e.get("rodadas_perguntas", 0) < agentes.max_rodadas


def validacao_ok(e: Estado) -> bool:
    return bool((e.get("conferencia") or {}).get("ok"))


def destinos_permitidos(e: Estado, agentes: Agentes) -> list[str]:
    if e.get("erro") or e.get("resposta"):
        return ["finalizar"]
    if e.get("spec") is None:
        return ["percepcao"]
    if e.get("analise") is None:
        return ["analista"]
    permitidos: list[str] = []
    if e.get("contextos") is None:
        if pode_perguntar(e, agentes):
            permitidos.append("perguntar_usuario")
        if not bloqueantes_abertas(e):
            permitidos.append("recuperador")
        return permitidos or ["finalizar"]          # bloqueante sem poder perguntar: impasse
    if e.get("gobr") is None:
        return ["gerador"]
    if e.get("conferencia") is None:
        return ["validador"]
    tentativas_restantes = e.get("tentativas", 0) < agentes.max_tentativas
    if not validacao_ok(e) and tentativas_restantes:
        return ["gerador"]
    if e.get("explicacoes") is None:
        return ["explicador"]
    return ["finalizar"]


def sugestao(e: Estado, agentes: Agentes) -> str:
    """Escolha determinística: a primeira opção permitida (perguntar vem antes de seguir)."""
    return destinos_permitidos(e, agentes)[0]


def resumo(e: Estado, agentes: Agentes) -> dict[str, Any]:
    spec = e.get("spec") or {}
    conferencia = e.get("conferencia") or {}
    return {
        "mensagens_do_usuario": len(e.get("mensagens") or []),
        "segmentos_extraidos": len(spec.get("segmentos") or []) if e.get("spec") else None,
        "analise_feita": e.get("analise") is not None,
        "perguntas_abertas": len(perguntas_abertas(e)),
        "perguntas_bloqueantes": len(bloqueantes_abertas(e)),
        "rodadas_de_perguntas": e.get("rodadas_perguntas", 0),
        "modo_interativo": agentes.interativo,
        "contexto_recuperado": e.get("contextos") is not None,
        "gobr_gerado": e.get("gobr") is not None,
        "tentativas_de_geracao": e.get("tentativas", 0),
        "max_tentativas": agentes.max_tentativas,
        "validacao": None if not conferencia else ("ok" if conferencia.get("ok") else "com erros"),
        "blocos_com_erro": len(conferencia.get("erros_por_bloco") or {}),
        "explicado": e.get("explicacoes") is not None,
    }


def decidir(e: Estado, agentes: Agentes) -> dict[str, Any]:
    permitidos = destinos_permitidos(e, agentes)
    padrao = permitidos[0]
    if len(permitidos) == 1 or agentes.orquestrador_llm is None:
        return {"proximo": padrao, "motivo": "única opção válida" if len(permitidos) == 1 else "regra",
                "origem": "regra", "permitidos": permitidos}
    try:
        resposta = texto_da_resposta(agentes.orquestrador_llm.invoke(
            orquestrador_prompt.mensagens(resumo(e, agentes), permitidos)))
        escolha = extrair_json(resposta)
        proximo = str(escolha.get("proximo", "")).strip()
        motivo = str(escolha.get("motivo", "")).strip()[:300]
    except Exception as exc:  # noqa: BLE001 - qualquer falha do LLM cai na regra
        return {"proximo": padrao, "motivo": f"LLM falhou ({type(exc).__name__}); usei a regra",
                "origem": "guardrail", "permitidos": permitidos}
    if proximo not in permitidos:
        return {"proximo": padrao, "motivo": f'LLM escolheu "{proximo}", que não é permitido agora; usei a regra',
                "origem": "guardrail", "permitidos": permitidos}
    return {"proximo": proximo, "motivo": motivo, "origem": "llm", "permitidos": permitidos}
