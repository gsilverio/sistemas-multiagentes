"""Avaliação ponta a ponta do agente: descrição do cliente → GOBR, comparado com o esperado.

Roda o grafo completo (Percepção → Analista → Recuperador → Gerador ⇄
Validador → Explicador) sem pausar para perguntas, em cada caso de
`data/avaliacao/casos_agente.json`, e mede:

- `valido`: o GOBR final passa no validador e na conferência com o RuleSpec;
- `blocos`: quantos blocos `parametros` foram gerados × esperados;
- `cobertura_ast`: fração dos blocos esperados com um bloco gerado de mesma
  estrutura (condições, frequência, `nao agendar`, modificadores; o nome do
  time e da regra não contam);
- `cobertura_acao`: fração dos blocos esperados com um gerado de mesma ação
  (mesma frequência ou `nao agendar`) e mesmo limite de dias nas condições,
  métrica tolerante a formas equivalentes (ex.: `recencia em 90` depois da
  regra de 60 × `recencia em 90 e fora da recencia em 60`);
- `achados`: falhas de modelagem esperadas que o Analista apontou (e as que
  não deveria apontar);
- tempo, tentativas do Gerador e passos do grafo.

    python -m src.pipelines.avaliar_agente --rotulo ollama
    GOBR_MODELO_GERADOR=hf:Qwen/Qwen3-4B GOBR_GERADOR_ADAPTADOR=<dir> python -m src.pipelines.avaliar_agente --rotulo treinado
    python -m src.pipelines.avaliar_agente --comparar outputs/avaliacao_agente
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from src.agents.validador import dias_da_frequencia
from src.gobr.ast import GobrParsedRule
from src.gobr.grammar import normalizar
from src.validation.gobr_validator import GobrValidator

RAIZ = Path(__file__).resolve().parents[2]
CASOS = RAIZ / "data" / "avaliacao" / "casos_agente.json"
METRICAS = ("valido", "cobertura_ast", "cobertura_acao", "achados_ok")


def _estrutura(regra: GobrParsedRule) -> tuple:
    return (tuple((c.operador or "", normalizar(c.texto)) for c in regra.condicoes),
            normalizar(regra.frequencia or ""), regra.nao_agendar,
            tuple(sorted(normalizar(m) for m in regra.modificadores)),
            tuple(sorted(normalizar(f) for f in regra.filtros_temporais)))


def _acao(regra: GobrParsedRule) -> tuple:
    numeros = [int(n) for c in regra.condicoes for n in re.findall(r"\d+", c.texto)]
    frequencia = dias_da_frequencia(regra.frequencia) if regra.frequencia else None
    return (regra.nao_agendar, frequencia if frequencia is not None else normalizar(regra.frequencia or ""),
            max(numeros) if numeros else None)


def _cobertura(esperadas: list[GobrParsedRule], geradas: list[GobrParsedRule], chave) -> float:
    if not esperadas:
        return 1.0
    livres = [chave(g) for g in geradas]
    acertos = 0
    for regra in esperadas:
        alvo = chave(regra)
        if alvo in livres:
            livres.remove(alvo)
            acertos += 1
    return round(acertos / len(esperadas), 3)


def comparar(gobr_gerado: str | None, gobr_esperado: str | None) -> dict[str, Any]:
    validador = GobrValidator()
    geradas = list(validador.validar(gobr_gerado).parametros.values()) if gobr_gerado else []
    if not gobr_esperado:
        return {"blocos_gerados": len(geradas)}
    esperadas = list(validador.validar(gobr_esperado).parametros.values())
    return {"blocos_gerados": len(geradas), "blocos_esperados": len(esperadas),
            "cobertura_ast": _cobertura(esperadas, geradas, _estrutura),
            "cobertura_acao": _cobertura(esperadas, geradas, _acao)}


def conferir_achados(codigos: list[str], caso: dict[str, Any]) -> dict[str, Any]:
    esperados, ausentes = caso.get("achados_esperados") or [], caso.get("achados_ausentes") or []
    if not esperados and not ausentes:
        return {}
    encontrados = [c for c in esperados if c in codigos]
    indevidos = [c for c in ausentes if c in codigos]
    total = len(esperados) + len(ausentes)
    return {"achados_ok": round((len(encontrados) + len(ausentes) - len(indevidos)) / total, 3),
            "achados_faltando": [c for c in esperados if c not in codigos], "achados_indevidos": indevidos}


def entrada_do_caso(caso: dict[str, Any]) -> tuple[str, list[str]]:
    from src.agents.percepcao import ler_anexo

    if caso.get("arquivo"):
        texto, imagens = ler_anexo(RAIZ / caso["arquivo"])
        return f"{caso.get('texto', '')}\n\n{texto}".strip(), imagens
    return caso["texto"], []


def gobr_esperado(caso: dict[str, Any]) -> str | None:
    if caso.get("gobr_esperado_texto"):
        return caso["gobr_esperado_texto"]
    if caso.get("gobr_esperado"):
        return (RAIZ / caso["gobr_esperado"]).read_text(encoding="utf-8")
    return None


def avaliar_caso(grafo: Any, caso: dict[str, Any]) -> dict[str, Any]:
    from src.grafo.grafo import config_thread, entrada

    texto, imagens = entrada_do_caso(caso)
    inicio = time.time()
    try:
        estado = grafo.invoke(entrada(texto, imagens), config_thread(f"avaliacao-{caso['id']}-{uuid.uuid4().hex[:6]}"))
    except Exception as exc:  # noqa: BLE001 - um caso que quebra não derruba a avaliação
        return {"id": caso["id"], "erro": f"{type(exc).__name__}: {exc}"[:300], "valido": False,
                "segundos": round(time.time() - inicio, 1)}
    analise = estado.get("analise") or {}
    codigos = [a["codigo"] for a in analise.get("achados") or []]
    resultado = {
        "id": caso["id"],
        "segundos": round(time.time() - inicio, 1),
        "valido": bool((estado.get("conferencia") or {}).get("ok")),
        "tentativas": estado.get("tentativas", 0),
        "passos": len([p for p in estado.get("trace") or [] if p["no"] not in ("inicio", "orquestrador")]),
        "decisoes_llm": sum(1 for p in estado.get("trace") or [] if p.get("origem") == "llm"),
        "guardrails": sum(1 for p in estado.get("trace") or [] if p.get("origem") == "guardrail"),
        "segmentos": len(((analise.get("spec") or {}).get("segmentos")) or []),
        "achados": codigos,
        "perguntas": [p["id"] for p in analise.get("perguntas") or []],
        "erro": estado.get("erro"),
        "gobr": estado.get("gobr"),
        **comparar(estado.get("gobr"), gobr_esperado(caso)),
        **conferir_achados(codigos, caso),
    }
    return resultado


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    resumo: dict[str, Any] = {"casos": len(resultados)}
    for m in METRICAS:
        valores = [float(r[m]) for r in resultados if m in r]
        if valores:
            resumo[m] = round(sum(valores) / len(valores), 3)
    resumo["segundos_medio"] = round(sum(r["segundos"] for r in resultados) / max(1, len(resultados)), 1)
    return resumo


def avaliar(grafo: Any, casos: list[dict[str, Any]], saida: Path, *, rotulo: str) -> dict[str, Any]:
    saida.mkdir(parents=True, exist_ok=True)
    resultados = []
    for caso in casos:
        r = avaliar_caso(grafo, caso)
        resultados.append(r)
        metricas = "  ".join(f"{m}={r[m]}" for m in ("valido", "cobertura_ast", "cobertura_acao", "achados_ok") if m in r)
        print(f"[{rotulo}] {caso['id']}: {metricas}  ({r['segundos']}s)", flush=True)
    resumo = resumir(resultados)
    (saida / f"{rotulo}.json").write_text(json.dumps({"resumo": resumo, "casos": resultados}, ensure_ascii=False,
                                                     indent=2) + "\n", encoding="utf-8")
    return resumo


def tabela(diretorio: Path) -> str:
    linhas = ["| execução | casos | " + " | ".join(METRICAS) + " | s/caso |", "|---|---|" + "---|" * len(METRICAS) + "---|"]
    detalhes = []
    for arquivo in sorted(diretorio.glob("*.json")):
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        r = dados["resumo"]
        linhas.append(f"| {arquivo.stem} | {r['casos']} | " + " | ".join(f"{r[m]:.3f}" if m in r else "-"
                                                                        for m in METRICAS)
                      + f" | {r['segundos_medio']} |")
        for c in dados["casos"]:
            detalhes.append(f"| {arquivo.stem} | {c['id']} | {c.get('valido')} | {c.get('blocos_gerados', '-')}"
                            f"/{c.get('blocos_esperados', '-')} | {c.get('cobertura_ast', '-')} | "
                            f"{c.get('cobertura_acao', '-')} | {', '.join(c.get('achados_faltando') or []) or '-'} |")
    cab = ["", "| execução | caso | válido | blocos (gerados/esperados) | cobertura AST | cobertura ação | "
               "achados faltando |", "|---|---|---|---|---|---|---|"]
    return "\n".join(linhas + cab + detalhes)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--rotulo", help="nome desta execução (ex.: ollama, treinado)")
    args.add_argument("--casos", type=Path, default=CASOS)
    args.add_argument("--somente", help="ids separados por vírgula")
    args.add_argument("--saida", type=Path, default=RAIZ / "outputs" / "avaliacao_agente")
    args.add_argument("--orquestrador-sem-llm", action="store_true")
    args.add_argument("--comparar", type=Path, help="só imprime a tabela das execuções salvas neste diretório")
    a = args.parse_args()
    if a.comparar:
        print(tabela(a.comparar))
        return
    if not a.rotulo:
        args.error("--rotulo é obrigatório (ou use --comparar)")
    from langgraph.checkpoint.memory import InMemorySaver

    from src.grafo.estado import Agentes
    from src.grafo.grafo import construir_grafo

    casos = json.loads(a.casos.read_text(encoding="utf-8"))["casos"]
    if a.somente:
        ids = set(a.somente.split(","))
        casos = [c for c in casos if c["id"] in ids]
    agentes = Agentes.padrao(orquestrador_com_llm=not a.orquestrador_sem_llm, interativo=False)
    grafo = construir_grafo(agentes, checkpointer=InMemorySaver())
    print(json.dumps(avaliar(grafo, casos, a.saida, rotulo=a.rotulo), ensure_ascii=False, indent=2))
    print(tabela(a.saida))


if __name__ == "__main__":
    main()
