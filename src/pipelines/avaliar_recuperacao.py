"""Mede a recuperação com as consultas de `data/avaliacao/recuperacao.json`.

Compara BM25, denso (embeddings) e híbrido (RRF) com:
- acerto@1 e acerto@k: alguma resposta certa na 1ª posição / no top-k;
- MRR: média de 1/posição da primeira resposta certa (0 se não houver).

    python -m src.pipelines.avaliar_recuperacao
    python -m src.pipelines.avaliar_recuperacao --erros   # lista as consultas que falharam
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from src.rag.busca_hibrida import normalizar_busca
from src.rag.recuperador import buscar_exemplos, buscar_manual, carregar_indice, DIR_INDICES

RAIZ = Path(__file__).resolve().parents[2]
CONSULTAS = RAIZ / "data" / "avaliacao" / "recuperacao.json"


def _posicao(resultados: list[Any], certo: Callable[[Any], bool]) -> int | None:
    return next((i for i, r in enumerate(resultados, start=1) if certo(r)), None)


def _resumo(posicoes: list[int | None], k: int) -> dict[str, float]:
    n = len(posicoes) or 1
    return {
        "acerto@1": round(sum(p == 1 for p in posicoes) / n, 3),
        f"acerto@{k}": round(sum(p is not None and p <= k for p in posicoes) / n, 3),
        "mrr": round(sum(1 / p for p in posicoes if p) / n, 3),
    }


def avaliar(consultas: dict[str, Any], *, modos: list[str], k_manual: int = 4, k_exemplos: int = 3,
            diretorio: Path = DIR_INDICES) -> dict[str, Any]:
    relatorio: dict[str, Any] = {"manual": {}, "exemplos": {}, "erros": []}
    for modo in modos:
        posicoes = []
        for item in consultas["manual"]:
            res = buscar_manual(item["consulta"], k=k_manual, modo=modo, diretorio=diretorio / "manual")
            p = _posicao(res, lambda r: r.secao in item["secoes"])
            posicoes.append(p)
            if p is None:
                relatorio["erros"].append({"indice": "manual", "modo": modo, "consulta": item["consulta"],
                                           "esperado": item["secoes"], "obtido": [r.secao for r in res]})
        relatorio["manual"][modo] = _resumo(posicoes, k_manual) | {"modo_efetivo": res[0].modo if res else None}

        posicoes = []
        for item in consultas["exemplos"]:
            res = buscar_exemplos(item["consulta"], k=k_exemplos, modo=modo, diretorio=diretorio / "exemplos")
            padroes = [re.compile(p) for p in item["padroes"]]
            p = _posicao(res, lambda r: all(x.search(normalizar_busca(r.gobr)) for x in padroes))
            posicoes.append(p)
            if p is None:
                relatorio["erros"].append({"indice": "exemplos", "modo": modo, "consulta": item["consulta"],
                                           "esperado": item["padroes"], "obtido": [r.nome for r in res]})
        relatorio["exemplos"][modo] = _resumo(posicoes, k_exemplos) | {"modo_efetivo": res[0].modo if res else None}
    return relatorio


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--consultas", type=Path, default=CONSULTAS)
    args.add_argument("--indices", type=Path, default=DIR_INDICES)
    args.add_argument("--erros", action="store_true")
    args.add_argument("--json", action="store_true")
    a = args.parse_args()
    consultas = json.loads(a.consultas.read_text(encoding="utf-8"))
    tem_vetores = carregar_indice(str(a.indices / "manual")).vetores is not None
    modos = ["bm25", "denso", "hibrido"] if tem_vetores else ["bm25"]
    relatorio = avaliar(consultas, modos=modos, diretorio=a.indices)
    if a.json:
        print(json.dumps(relatorio, ensure_ascii=False, indent=2))
        return
    for indice in ("manual", "exemplos"):
        n = len(consultas[indice])
        print(f"\n{indice} ({n} consultas)")
        for modo, r in relatorio[indice].items():
            metricas = "  ".join(f"{m}={v:.3f}" for m, v in r.items() if m != "modo_efetivo")
            print(f"  {modo:8s} {metricas}")
    if not tem_vetores:
        print("\n(sem vetores.npy: só BM25; gere os embeddings para comparar os três modos)")
    if a.erros:
        for e in relatorio["erros"]:
            print(f"\nFALHOU [{e['indice']}/{e['modo']}] {e['consulta']}\n  esperado: {e['esperado']}\n  obtido:   {e['obtido']}")


if __name__ == "__main__":
    main()
