"""Confere um JSONL de explicações com o guard da etapa 2.

Cada linha precisa de `id`, `gobr` e `explicacao` (texto). Serve tanto para a
saída de `generate_explanations` (`explicacoes.jsonl`) quanto para a linha
de base antiga (`data/avaliacao/explicacoes_baseline_v1.jsonl`).

    python -m src.pipelines.avaliar_explicacoes data/avaliacao/explicacoes_baseline_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from src.explicacao.guarda import verificar
from src.explicacao.rascunho import rascunho_de_gobr


def avaliar(caminho: Path) -> dict[str, Any]:
    total, aprovadas = 0, 0
    por_metodo: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    motivos: collections.Counter = collections.Counter()
    reprovadas: list[dict[str, Any]] = []
    with caminho.open(encoding="utf-8") as origem:
        for linha in origem:
            if not linha.strip():
                continue
            item = json.loads(linha)
            texto = item["explicacao"] if isinstance(item["explicacao"], str) else item["explicacao"]["texto"]
            _, rascunho = rascunho_de_gobr(item["gobr"])
            resultado = verificar(rascunho, texto, gobr=item["gobr"])
            metodo = item.get("metodo") or item.get("revisao_metodo") or "?"
            total += 1
            por_metodo[metodo]["total"] += 1
            if resultado.aprovado:
                aprovadas += 1
                por_metodo[metodo]["aprovadas"] += 1
                continue
            for problema in resultado.problemas:
                motivos[problema.split(":")[0]] += 1
            reprovadas.append({"id": item["id"], "metodo": metodo, "explicacao": texto,
                               "problemas": resultado.problemas})
    return {"arquivo": str(caminho), "total": total, "aprovadas": aprovadas,
            "por_metodo": {m: dict(c) for m, c in por_metodo.items()},
            "motivos": dict(motivos.most_common()), "reprovadas": reprovadas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arquivo", type=Path)
    parser.add_argument("--exemplos", type=int, default=5, help="Quantas reprovações mostrar.")
    parser.add_argument("--json", type=Path, help="Grava o relatório completo neste arquivo.")
    args = parser.parse_args()
    relatorio = avaliar(args.arquivo)
    print(f"{relatorio['aprovadas']}/{relatorio['total']} explicações aprovadas pelo guard.")
    for metodo, contagem in relatorio["por_metodo"].items():
        print(f"  {metodo}: {contagem.get('aprovadas', 0)}/{contagem['total']}")
    if relatorio["motivos"]:
        print("Motivos de reprovação:")
        for motivo, quantidade in relatorio["motivos"].items():
            print(f"  {quantidade:4d}  {motivo}")
    for item in relatorio["reprovadas"][:args.exemplos]:
        print(f"\n{item['id']} ({item['metodo']}): {item['explicacao']}")
        for problema in item["problemas"]:
            print(f"  - {problema}")
    if args.json:
        args.json.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
