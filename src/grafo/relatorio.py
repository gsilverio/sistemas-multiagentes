"""Resposta final ao usuário (markdown), montada sem LLM a partir do estado."""

from __future__ import annotations

from src.grafo.estado import Estado

ICONES = {"erro": "❌", "aviso": "⚠️", "info": "💡"}


def montar_resposta(e: Estado, *, max_tentativas: int) -> str:
    if e.get("erro"):
        return f"Não consegui concluir: {e['erro']}"
    analise = e.get("analise") or {}
    partes: list[str] = []

    bloqueantes = [p for p in analise.get("perguntas") or [] if p.get("bloqueante")]
    if bloqueantes and not e.get("gobr"):
        partes.append("## Preciso de mais informações\n\n" + "\n".join(f"- {p['texto']}" for p in bloqueantes))

    achados = analise.get("achados") or []
    if achados:
        ordem = {"erro": 0, "aviso": 1, "info": 2}
        linhas = [f"- {ICONES[a['severidade']]} {a['mensagem']}"
                  for a in sorted(achados, key=lambda a: ordem[a["severidade"]])]
        partes.append("## Falhas e pontos de atenção na modelagem\n\n" + "\n".join(linhas))

    if analise.get("ajustes"):
        partes.append("## Ajustes feitos automaticamente\n\n" + "\n".join(f"- {a}" for a in analise["ajustes"]))

    suposicoes = [p for p in [*(analise.get("perguntas") or []), *(analise.get("perguntas_sem_resposta") or [])]
                  if not p.get("bloqueante") and p.get("suposicao")]
    if suposicoes and e.get("gobr"):
        partes.append("## Suposições (perguntas sem resposta)\n\n"
                      + "\n".join(f"- {p['texto']} **Assumido:** {p['suposicao']}." for p in suposicoes))

    if e.get("gobr"):
        conferencia = e.get("conferencia") or {}
        if conferencia.get("ok"):
            status = "✅ GOBR válido e conferido com as regras descritas."
        else:
            erros = [f"- `{bloco}`: {m}" for bloco, ms in (conferencia.get("erros_por_bloco") or {}).items()
                     for m in ms] + [f"- {m}" for m in conferencia.get("erros_gerais") or []]
            status = (f"⚠️ Restaram erros depois de {e.get('tentativas', 0)} tentativa(s) "
                      f"(máximo {max_tentativas}); revise antes de usar:\n" + "\n".join(erros))
        partes.append(f"## Regras em GOBR\n\n```gobr\n{e['gobr'].rstrip()}\n```\n\n{status}")

    explicacoes = e.get("explicacoes") or []
    if explicacoes:
        partes.append("## O que cada regra faz\n\n"
                      + "\n".join(f"- **{x['nome']}**: {x['texto']}" for x in explicacoes))

    return "\n\n".join(partes) or "Nada a responder."
