"""Prepara um JSONL de regras extraidas. Nao requer Ollama nem Colab."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.parsers.gobr_rule_extractor import GobrRuleExtractor
from src.parsers.gobr_parser import GobrParser


def main() -> None:
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--input", type=Path, default=Path("data/referencias"))
    argumentos.add_argument("--output", type=Path,
                            default=Path("outputs/generated/regras_extraidas.jsonl"))
    args = argumentos.parse_args()

    extractor = GobrRuleExtractor()
    arquivos = sorted(args.input.glob("*.gobr"))
    regras = extractor.extract_directory(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destino:
        for regra in regras:
            destino.write(json.dumps(asdict(regra), ensure_ascii=False) + "\n")

    # Diagnostico separado: nao transforma casos nao suportados em explicacoes.
    parser = GobrParser()
    diagnosticos = []
    sem_suporte = 0
    erros = 0
    for regra in regras:
        try:
            resultado = parser.parse(regra.gobr)
        except ValueError as erro:
            erros += 1
            diagnosticos.append({"id": regra.id, "erro": str(erro)})
            continue
        if resultado.desconhecidos:
            sem_suporte += 1
            diagnosticos.append({
                "id": regra.id,
                "clausulas_desconhecidas": resultado.desconhecidos,
            })
    diagnostico_path = args.output.with_name("diagnostico_parser.json")
    diagnostico_path.write_text(
        json.dumps(diagnosticos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    com_contexto = sum(bool(r.contexto) for r in regras)
    faltando = sum(bool(r.referencias_nao_resolvidas) for r in regras)
    print(f"Arquivos lidos: {len(arquivos)}")
    print(f"Blocos parametros extraidos: {len(regras)}")
    print(f"Blocos com contexto local: {com_contexto}")
    print(f"Blocos com referencias nao resolvidas localmente: {faltando}")
    print(f"Blocos com clausulas desconhecidas pelo parser: {sem_suporte}")
    print(f"Blocos com erro de parsing: {erros}")
    print(f"Arquivo JSONL gerado: {args.output.resolve()}")
    print(f"Diagnostico gerado: {diagnostico_path.resolve()}")


if __name__ == "__main__":
    main()
