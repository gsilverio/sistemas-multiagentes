"""Lote de descrições no estilo das tabelas dos clientes (Colab, Ollama).

Uma chamada por corpo distinto de bloco (1.202 no acervo); blocos repetidos
recebem a mesma descrição. Cada resultado vira `items/<id>.json` assim que
termina; reexecutar pula o que já foi feito com o mesmo prompt e modelo. Se o
Ollama parar de responder (3 falhas seguidas), o lote para com código 3 e a
célula do notebook reinicia o Ollama. No fim, `descricoes_cliente.jsonl`
consolida tudo (uma linha por bloco, inclusive os repetidos).

    python -m src.pipelines.gerar_descricoes_cliente --output-dir /content/drive/.../descricoes --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from src.gobr.assinatura import chave_corpo
from src.gobr.extractor import GobrRuleExtractor
from src.pipelines.generate_explanations import (CODIGO_OLLAMA_INDISPONIVEL, MAX_FALHAS_INFRA, OllamaIndisponivel,
                                                 atomic_write_json, falha_de_infra, read_previous, verificar_ollama)

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVOS_DO_PROMPT = ("src/prompts/descricao_cliente_prompt.py", "src/explicacao/descricao_cliente.py")


def versao_prompt() -> str:
    h = hashlib.sha256()
    for nome in ARQUIVOS_DO_PROMPT:
        h.update((RAIZ / nome).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def consolidar(saida: Path, grupos: dict[str, list[dict[str, Any]]]) -> Path:
    destino = saida / "descricoes_cliente.jsonl"
    total = 0
    with destino.open("w", encoding="utf-8") as arquivo:
        for blocos in grupos.values():
            item = read_previous(saida / "items" / f"{blocos[0]['id']}.json")
            if not item or item.get("status") != "completed":
                continue
            for bloco in blocos:
                arquivo.write(json.dumps({"id": bloco["id"], "nome": bloco["nome"], "descricao": item["descricao"],
                                          "metodo": item["metodo"], "model": item["model"],
                                          "reaproveitada_de": None if bloco is blocos[0] else blocos[0]["id"]},
                                         ensure_ascii=False) + "\n")
                total += 1
    print(f"{total} descrições consolidadas em {destino}", flush=True)
    return destino


def run(referencias: Path, saida: Path, *, limite: int = 0, max_tentativas: int = 2, modelo: str | None = None,
        agente_factory: Callable[[], Any] | None = None, verificar_servico: Callable[[], None] | None = None,
        dry_run: bool = False) -> dict[str, Any]:
    from src.llm_config import modelo_padrao

    modelo = modelo or modelo_padrao()
    grupos: dict[str, list[dict[str, Any]]] = {}
    for b in GobrRuleExtractor().extract_directory(referencias):
        grupos.setdefault(chave_corpo(b.gobr), []).append({"id": b.id, "nome": b.nome, "gobr": b.gobr})
    assinatura = f"{versao_prompt()}:{modelo}"
    pendentes = []
    for blocos in grupos.values():
        anterior = read_previous(saida / "items" / f"{blocos[0]['id']}.json")
        if anterior and anterior.get("assinatura") == assinatura:
            if anterior["status"] in ("completed", "blocked"):
                continue
            if anterior["status"] == "failed" and anterior.get("tentativas", 0) >= max_tentativas:
                continue
        pendentes.append((blocos[0], anterior if anterior and anterior.get("assinatura") == assinatura else None))
    if limite:
        pendentes = pendentes[:limite]
    print(f"Corpos distintos: {len(grupos)} | a processar agora: {len(pendentes)}", flush=True)
    if dry_run:
        for bloco, _ in pendentes[:10]:
            print(" -", bloco["id"], bloco["nome"])
        return {"pendentes": len(pendentes)}

    (verificar_servico or verificar_ollama)()
    if agente_factory is None:
        from src.explicacao.descricao_cliente import AgenteDescricaoCliente
        from src.llm_config import create_llm

        agente_factory = lambda: AgenteDescricaoCliente(create_llm(modelo))  # noqa: E731
    agente = agente_factory()
    contagem: dict[str, int] = {}
    falhas_infra = 0
    for i, (bloco, anterior) in enumerate(pendentes, start=1):
        inicio = time.time()
        registro: dict[str, Any] = {"id": bloco["id"], "gobr": bloco["gobr"], "assinatura": assinatura,
                                    "model": modelo, "tentativas": (anterior or {}).get("tentativas", 0) + 1}
        try:
            resultado = agente.descrever(bloco["gobr"])
            registro.update(status="completed" if resultado.texto else "failed", descricao=resultado.texto,
                            metodo=resultado.metodo, problemas=resultado.problemas, respostas=resultado.tentativas)
            falhas_infra = 0
        except ValueError as exc:  # bloco que o parser não entende
            registro.update(status="blocked", erro=str(exc))
        except Exception as exc:  # noqa: BLE001
            infra = falha_de_infra(type(exc).__name__, str(exc))
            registro.update(status="failed", erro=f"{type(exc).__name__}: {exc}"[:500])
            if infra:
                registro["tentativas"] -= 1  # não é culpa do bloco
                falhas_infra += 1
        registro["segundos"] = round(time.time() - inicio, 2)
        atomic_write_json(saida / "items" / f"{bloco['id']}.json", registro)
        contagem[registro.get("metodo") or registro["status"]] = contagem.get(registro.get("metodo") or
                                                                             registro["status"], 0) + 1
        if falhas_infra >= MAX_FALHAS_INFRA:
            consolidar(saida, grupos)
            raise OllamaIndisponivel("O Ollama parou de responder no meio do lote.")
        if i % 25 == 0:
            print(f"{i}/{len(pendentes)} {contagem}", flush=True)
    consolidar(saida, grupos)
    print("Resultado desta execução:", contagem, flush=True)
    return contagem


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--referencias", type=Path, default=RAIZ / "data" / "referencias")
    args.add_argument("--output-dir", type=Path, default=RAIZ / "outputs" / "descricoes_cliente")
    args.add_argument("--limit", type=int, default=0, help="0 = tudo o que falta")
    args.add_argument("--max-attempts", type=int, default=2)
    args.add_argument("--dry-run", action="store_true")
    a = args.parse_args()
    try:
        run(a.referencias, a.output_dir, limite=a.limit, max_tentativas=a.max_attempts, dry_run=a.dry_run)
    except OllamaIndisponivel as exc:
        print(f"LOTE INTERROMPIDO: {exc}", flush=True)
        sys.exit(CODIGO_OLLAMA_INDISPONIVEL)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
