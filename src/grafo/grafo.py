"""Grafo multiagente (LangGraph): supervisor em estrela.

    START → inicio → orquestrador ─Command(goto)─► percepcao | analista | perguntar_usuario |
                         ▲                         recuperador | gerador | validador |
                         └──────── todos voltam ───┘ explicador
                                                   finalizar → END

- `inicio` zera os campos derivados quando chega uma mensagem nova (uma
  conversa pode pedir ajustes depois da primeira resposta; o checkpointer
  guarda o histórico por `thread_id`).
- Cada nó de agente recebe uma projeção tipada do estado e devolve só os
  campos que produz. Verificação cega: o Analista vê só o RuleSpec, o
  Explicador só o GOBR.
- `perguntar_usuario` pausa com `interrupt()`; retome com
  `grafo.invoke(Command(resume="respostas"), config)`.

Uso:
    grafo = construir_grafo(Agentes.padrao(), checkpointer=InMemorySaver())
    saida = grafo.invoke(entrada("descrição..."), config_thread("t1"))
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents.contratos import Analise, Conferencia, ContextoSegmento, RuleSpec
from src.agents.gerador import montar_arquivo
from src.grafo import orquestrador
from src.grafo.estado import CAMPOS_DERIVADOS, Agentes, Estado
from src.grafo.relatorio import montar_resposta

log = logging.getLogger(__name__)
LIMITE_PASSOS = 80


def entrada(texto: str = "", imagens: list[str] | None = None, origem: str = "usuario", *,
            exibir: str | None = None, anexos: list[str] | None = None) -> dict[str, Any]:
    """Mensagem do usuário. `exibir`/`anexos`: o que aparece no histórico (sem o texto extraído do PDF)."""
    mensagem = {"texto": texto, "imagens": list(imagens or []), "origem": origem}
    if exibir is not None:
        mensagem["exibir"] = exibir
    if anexos:
        mensagem["anexos"] = list(anexos)
    return {"mensagens": [mensagem]}


def config_thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": LIMITE_PASSOS}


def _passo(no: str, **dados: Any) -> dict[str, Any]:
    return {"trace": [{"no": no, **dados}]}


def _falha(no: str, exc: Exception) -> dict[str, Any]:
    log.exception("Falha no nó %s", no)
    return {"erro": f"{no}: {type(exc).__name__}: {exc}", **_passo(no, erro=str(exc)[:300])}


def construir_grafo(agentes: Agentes, *, checkpointer: Any = None):
    def inicio(e: Estado) -> dict[str, Any]:
        mensagens = e.get("mensagens") or []
        total = len(mensagens)
        if total > e.get("processadas", 0):
            novas = [{"papel": "usuario", "texto": m.get("exibir") or m["texto"],
                      "anexos": m.get("anexos") or []} for m in mensagens[e.get("processadas", 0):]]
            return {**CAMPOS_DERIVADOS, "processadas": total, "rodadas_perguntas": 0, "perguntas_feitas": [],
                    "conversa": novas, **_passo("inicio", nova_mensagem=True)}
        return _passo("inicio", nova_mensagem=False)

    def no_orquestrador(e: Estado) -> Command:
        decisao = orquestrador.decidir(e, agentes)
        return Command(goto=decisao["proximo"], update=_passo("orquestrador", **decisao))

    def no_percepcao(e: Estado) -> dict[str, Any]:
        mensagens = e.get("mensagens") or []
        textos = [m["texto"] if m.get("origem") != "respostas" else f"Respostas do usuário às perguntas:\n{m['texto']}"
                  for m in mensagens]
        imagens = [img for m in mensagens for img in m.get("imagens") or []]
        try:
            spec = agentes.percepcao.extrair(textos, imagens)
        except Exception as exc:  # noqa: BLE001
            return _falha("percepcao", exc)
        return {"spec": spec.model_dump(), **_passo("percepcao", segmentos=len(spec.segmentos))}

    def no_analista(e: Estado) -> dict[str, Any]:
        analise = agentes.analista.analisar(RuleSpec.model_validate(e["spec"]))
        return {"analise": analise.model_dump(),
                **_passo("analista", achados=len(analise.achados), perguntas=len(analise.perguntas),
                         ajustes=len(analise.ajustes))}

    def no_perguntar(e: Estado) -> dict[str, Any]:
        perguntas = orquestrador.a_perguntar(e)
        texto = "Antes de gerar as regras, preciso confirmar:\n" + "\n".join(
            f"{i}. {p['texto']}" + (f" (Se não responder, assumo que {p['suposicao']}.)" if p.get("suposicao") else "")
            for i, p in enumerate(perguntas, start=1))
        resposta = interrupt({"perguntas": perguntas, "mensagem": texto})
        resposta = str(resposta or "").strip()
        update: dict[str, Any] = {"rodadas_perguntas": e.get("rodadas_perguntas", 0) + 1,
                                  "conversa": [{"papel": "agente", "texto": texto},
                                               {"papel": "usuario",
                                                "texto": resposta or "(seguir com as suposições)"}],
                                  "perguntas_feitas": [*(e.get("perguntas_feitas") or []), *(p["id"] for p in perguntas)],
                                  **_passo("perguntar_usuario", perguntas=len(perguntas), respondeu=bool(resposta))}
        if resposta:
            # A Percepção refaz o RuleSpec com a descrição original + as respostas.
            update.update({"mensagens": [{"texto": resposta, "imagens": [], "origem": "respostas"}],
                           "processadas": len(e.get("mensagens") or []) + 1, "spec": None, "analise": None})
        else:
            # Sem resposta: segue com as suposições (perguntas não bloqueantes).
            analise = dict(e["analise"])
            analise["perguntas"] = [p for p in analise.get("perguntas") or [] if p.get("bloqueante")]
            analise["perguntas_sem_resposta"] = [p for p in perguntas if not p.get("bloqueante")]
            update["analise"] = analise
        return update

    def no_recuperador(e: Estado) -> dict[str, Any]:
        spec = Analise.model_validate(e["analise"]).spec
        try:
            contextos = agentes.recuperador.recuperar(spec)
        except Exception as exc:  # noqa: BLE001 - sem índice o gerador ainda funciona, só sem exemplos
            log.warning("Recuperador falhou: %s", exc)
            contextos = {}
        return {"contextos": {n: c.model_dump() for n, c in contextos.items()},
                **_passo("recuperador", segmentos=len(contextos),
                         exemplos=sum(len(c.exemplos) for c in contextos.values()))}

    def no_gerador(e: Estado) -> dict[str, Any]:
        spec = Analise.model_validate(e["analise"]).spec
        contextos = {n: ContextoSegmento.model_validate(c) for n, c in (e.get("contextos") or {}).items()}
        conferencia = e.get("conferencia") or {}
        anteriores = e.get("blocos") or {}
        erros = dict(conferencia.get("erros_por_bloco") or {})
        if conferencia and conferencia.get("erros_gerais") and not erros:
            erros = {s.nome: conferencia["erros_gerais"] for s in spec.segmentos}  # erro sem bloco: refaz tudo
        try:
            blocos = agentes.gerador.gerar(spec, contextos, anteriores=anteriores, erros_por_bloco=erros)
        except Exception as exc:  # noqa: BLE001
            return _falha("gerador", exc)
        return {"blocos": blocos, "gobr": montar_arquivo(spec, blocos), "conferencia": None,
                "tentativas": e.get("tentativas", 0) + 1,
                **_passo("gerador", tentativa=e.get("tentativas", 0) + 1,
                         refeitos=len(erros) if anteriores else len(blocos))}

    def no_validador(e: Estado) -> dict[str, Any]:
        spec = Analise.model_validate(e["analise"]).spec
        conferencia: Conferencia = agentes.validador.validar(e["gobr"], spec)
        return {"conferencia": conferencia.model_dump(),
                **_passo("validador", ok=conferencia.ok, blocos_com_erro=len(conferencia.erros_por_bloco),
                         erros_gerais=len(conferencia.erros_gerais))}

    def no_explicador(e: Estado) -> dict[str, Any]:
        com_erro = set((e.get("conferencia") or {}).get("erros_por_bloco") or {})
        explicacoes = []
        for nome, bloco in (e.get("blocos") or {}).items():
            if nome in com_erro:
                continue
            try:
                x = agentes.explicador.explicar(bloco)  # só o GOBR: verificação cega
                explicacoes.append({"nome": nome, "texto": x.texto, "metodo": x.metodo, "aprovada": x.aprovada})
            except Exception as exc:  # noqa: BLE001
                log.warning("Explicador falhou em %s: %s", nome, exc)
        return {"explicacoes": explicacoes, **_passo("explicador", explicados=len(explicacoes))}

    def no_finalizar(e: Estado) -> dict[str, Any]:
        resposta = montar_resposta(e, max_tentativas=agentes.max_tentativas)
        return {"resposta": resposta, "conversa": [{"papel": "agente", "texto": resposta}],
                **_passo("finalizar", caracteres=len(resposta))}

    g = StateGraph(Estado)
    g.add_node("inicio", inicio)
    g.add_node("orquestrador", no_orquestrador, destinations=orquestrador.AGENTES)
    trabalhadores = {
        "percepcao": no_percepcao, "analista": no_analista, "perguntar_usuario": no_perguntar,
        "recuperador": no_recuperador, "gerador": no_gerador, "validador": no_validador,
        "explicador": no_explicador,
    }
    for nome, funcao in trabalhadores.items():
        g.add_node(nome, funcao)
        g.add_edge(nome, "orquestrador")
    g.add_node("finalizar", no_finalizar)
    g.add_edge(START, "inicio")
    g.add_edge("inicio", "orquestrador")
    g.add_edge("finalizar", END)
    return g.compile(checkpointer=checkpointer)
