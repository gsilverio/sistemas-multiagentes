"""Geração incremental de explicações GOBR para revisão, sem indexação.

Uso local (sem GPU):
    python -m src.prepare_references
    python -m src.batch_reference_explanations --dry-run --limit 3

No Colab, configure OLLAMA_BASE_URL, GOBR_LLM_MODEL e GOBR_LLM_NUM_CTX
antes de executar, apontando --output-dir para o Drive montado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.parsers.gobr_parser import GobrParser

PIPELINE_VERSION = "gobr-reference-batch-v1"
SAFE_ID = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_revision() -> str:
    """Commit para impedir reutilização silenciosa de resultados de código antigo."""
    explicit = os.getenv("GOBR_SOURCE_REVISION", "").strip()
    if explicit:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return "unversioned"
    revision = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=False,
    )
    return revision + ("-dirty" if dirty.stdout.strip() else "")


def load_rules(path: Path) -> list[dict[str, Any]]:
    """Valida a estrutura mínima do JSONL antes de iniciar uma inferência."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Arquivo {path} ausente. Execute python -m src.prepare_references."
        )
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if not isinstance(rid, str) or not SAFE_ID.fullmatch(rid):
                raise ValueError(f"ID inválido na linha {line_number}: {rid!r}")
            if rid in seen:
                raise ValueError(f"ID duplicado no JSONL: {rid}")
            if not isinstance(obj.get("gobr"), str) or not obj["gobr"].strip():
                raise ValueError(f"GOBR ausente na linha {line_number}")
            if not isinstance(obj.get("contexto", []), list):
                raise ValueError(f"Contexto inválido na linha {line_number}")
            seen.add(rid)
            rules.append(obj)
    return rules


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Escreve cada item em duas etapas, para evitar JSON parcial em interrupções."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as dest:
        json.dump(obj, dest, ensure_ascii=False, indent=2)
        dest.write("\n")
        dest.flush()
        os.fsync(dest.fileno())
    os.replace(temp, path)


def read_previous(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        return previous if isinstance(previous, dict) else None
    except (OSError, json.JSONDecodeError):
        return None  # O item será reprocessado em vez de marcado como concluído.


def fingerprint(rule: dict[str, Any], *, model: str, num_ctx: int,
                revision: str, max_context_chars: int) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "id": rule["id"],
        "gobr": rule["gobr"],
        "contexto": rule.get("contexto", []),
        "model": model,
        "num_ctx": num_ctx,
        "code_revision": revision,
        "max_context_chars": max_context_chars,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def context_text(context: list[dict[str, Any]]) -> str:
    """Todas as definições locais como dados, sem alterar a GOBR original."""
    return json.dumps(
        [
            {"tipo": d["tipo"], "nome": d["nome"], "texto": d["texto"]}
            for d in context
        ],
        ensure_ascii=False,
    )


class ReferenceBatchRunner:
    def __init__(
        self, *,
        interpreter_factory: Callable[[], Any] | None = None,
        explainer_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.parser = GobrParser()
        self.interpreter_factory = interpreter_factory
        self.explainer_factory = explainer_factory
        self.interpreter: Any = None
        self.explainer: Any = None

    def _load_agents(self) -> None:
        if self.interpreter is not None and self.explainer is not None:
            return
        if self.interpreter_factory is None:
            from src.agents.gobr_interpreter import GobrInterpreterAgent
            self.interpreter_factory = GobrInterpreterAgent
        if self.explainer_factory is None:
            from src.agents.business_explainer import BusinessExplainerAgent
            self.explainer_factory = BusinessExplainerAgent
        if self.interpreter is None:
            self.interpreter = self.interpreter_factory()
        if self.explainer is None:
            self.explainer = self.explainer_factory()

    def run(
        self, *, input_path: Path, output_dir: Path, limit: int = 3,
        max_attempts: int = 2, max_context_chars: int = 4000,
        model: str = "gemma3:12b", num_ctx: int = 4096,
        revision: str = "unversioned", dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        if limit < 0 or max_attempts < 1 or max_context_chars < 1 or num_ctx < 1:
            raise ValueError("limit >= 0, max_attempts >= 1 e tamanhos positivos.")
        rules = load_rules(input_path)
        items_dir = output_dir / "items"
        stats: dict[str, Any] = {
            "total_regras": len(rules), "simuladas": 0, "concluidas_nesta_execucao": 0,
            "falhas_nesta_execucao": 0, "bloqueadas_nesta_execucao": 0,
            "ignoradas_concluidas": 0, "ignoradas_tentativas": 0,
            "ignoradas_contexto": 0, "dry_run": dry_run,
            "model": model, "num_ctx": num_ctx, "revision": revision,
            "executed_at_utc": utc_now(), "output_dir": str(output_dir),
        }
        executed = 0
        for rule in rules:
            rule_id = rule["id"]
            output = items_dir / f"{rule_id}.json"
            signature = fingerprint(rule, model=model, num_ctx=num_ctx,
                                    revision=revision,
                                    max_context_chars=max_context_chars)
            previous = read_previous(output)
            matches = previous is not None and previous.get("fingerprint") == signature
            if not force and matches:
                if previous.get("status") == "completed":
                    stats["ignoradas_concluidas"] += 1
                    continue
                if previous.get("status") == "blocked_context":
                    stats["ignoradas_contexto"] += 1
                    continue
                if previous.get("status") == "failed" and previous.get("attempt", 0) >= max_attempts:
                    stats["ignoradas_tentativas"] += 1
                    continue
            if limit and executed >= limit:
                break
            executed += 1
            if dry_run:
                stats["simuladas"] += 1
                print(f"[DRY RUN] {rule_id}: {rule.get('nome', '')}")
                continue
            attempts = (
                int(previous.get("attempt", 0)) + 1
                if matches and not force and previous else 1
            )
            started = time.monotonic()
            record: dict[str, Any] = {
                "id": rule_id,
                "nome": rule.get("nome"),
                "arquivo_origem": rule.get("arquivo_origem"),
                "indice": rule.get("indice"),
                "linha_inicio": rule.get("linha_inicio"),
                "linha_fim": rule.get("linha_fim"),
                "gobr": rule["gobr"],
                "contexto": rule.get("contexto", []),
                "referencias_nao_resolvidas": rule.get("referencias_nao_resolvidas", []),
                "fingerprint": signature,
                "pipeline_version": PIPELINE_VERSION,
                "model": model,
                "num_ctx": num_ctx,
                "code_revision": revision,
                "attempt": attempts,
                "created_at_utc": utc_now(),
                "indexable": False,  # Nenhum item entra automaticamente no RAG.
            }
            try:
                ctx = record["contexto"]
                if not all(isinstance(d, dict) and all(k in d for k in ("tipo", "nome", "texto")) for d in ctx):
                    raise ValueError("Definições de contexto malformadas.")
                context_json = context_text(ctx)
                if len(context_json) > max_context_chars:
                    record.update(
                        status="blocked_context",
                        reason=f"Contexto de {len(context_json)} caracteres ultrapassa o limite de {max_context_chars}; nenhuma definição foi omitida.",
                    )
                    stats["bloqueadas_nesta_execucao"] += 1
                elif record["referencias_nao_resolvidas"]:
                    record.update(
                        status="blocked_context",
                        reason="Referências de contexto não resolvidas: " + ", ".join(record["referencias_nao_resolvidas"]),
                    )
                    stats["bloqueadas_nesta_execucao"] += 1
                else:
                    parsed = self.parser.parse(rule["gobr"])
                    record["parsed"] = parsed.model_dump(mode="json")
                    if parsed.desconhecidos:
                        record.update(status="blocked_context",
                                      reason="Cláusulas não reconhecidas pelo parser.")
                        stats["bloqueadas_nesta_execucao"] += 1
                    else:
                        self._load_agents()
                        interpretation = self.interpreter.interpret(parsed, contexto=ctx)
                        explanation = self.explainer.explain(
                            interpretation, parsed_rule=parsed,
                        )
                        record["interpretation"] = interpretation.model_dump(mode="json")
                        record["explanation"] = explanation.model_dump(mode="json")
                        record["status"] = "completed"
                        stats["concluidas_nesta_execucao"] += 1
            except Exception as exc:
                record.update(status="failed", error={
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                })
                stats["falhas_nesta_execucao"] += 1
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            atomic_write_json(output, record)
            print(f"[{record['status']}] {rule_id} | tentativa {attempts} | {record['elapsed_seconds']}s", flush=True)
        if not dry_run:
            atomic_write_json(output_dir / "batch_summary.json", stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/generated/regras_extraidas.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference_explanations"))
    parser.add_argument("--limit", type=int, default=3,
                        help="Número máximo de novas tentativas nesta execução; 0 = sem limite.")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Refaz regras mesmo que já tenham sido concluídas.")
    args = parser.parse_args()
    runner = ReferenceBatchRunner()
    runner.run(
        input_path=args.input, output_dir=args.output_dir,
        limit=args.limit, max_attempts=args.max_attempts,
        max_context_chars=args.max_context_chars,
        model=os.getenv("GOBR_LLM_MODEL", "gemma3:12b"),
        num_ctx=int(os.getenv("GOBR_LLM_NUM_CTX", "4096")),
        revision=code_revision(), dry_run=args.dry_run, force=args.force,
    )


if __name__ == "__main__":
    main()
