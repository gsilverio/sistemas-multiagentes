"""Geração incremental de explicações para os blocos `parametros` do acervo.

Cada bloco vira `items/<id>.json`, gravado assim que termina; reexecutar
pula o que já foi feito com a mesma versão do explicador e o mesmo modelo.
Ao final, `explicacoes.jsonl` consolida todos os itens concluídos (entrada
do índice RAG da etapa 3 e dos pares de fine-tuning da etapa 6).

Local, sem LLM (só o rascunho determinístico, ~1 min para o acervo todo):
    python -m src.pipelines.prepare_references
    python -m src.pipelines.generate_explanations --sem-llm --limit 0

No Colab (Ollama + gemma3:12b), apontando --output-dir para o Drive:
    python -m src.pipelines.generate_explanations --output-dir /content/drive/... --limit 50
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

PIPELINE_VERSION = "gobr-explicador-v2"
SAFE_ID = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$")
RAIZ = Path(__file__).resolve().parents[2]
# Arquivos que mudam o resultado da explicação; mudar qualquer um deles
# invalida os itens já gerados (mudanças em outras partes do código, não).
ARQUIVOS_DO_EXPLICADOR = (
    "src/agents/explicador.py",
    "src/prompts/explicador_prompt.py",
    "src/explicacao/glossario.py",
    "src/explicacao/rascunho.py",
    "src/explicacao/guarda.py",
    "src/gobr/parser.py",
    "src/gobr/scope_text.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def versao_explicador() -> str:
    h = hashlib.sha256(PIPELINE_VERSION.encode())
    for relativo in ARQUIVOS_DO_EXPLICADOR:
        h.update(relativo.encode())
        h.update((RAIZ / relativo).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def code_revision() -> str:
    """Commit atual, só para registro (não entra na assinatura)."""
    explicit = os.getenv("GOBR_SOURCE_REVISION", "").strip()
    if explicit:
        return explicit
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return "unversioned"
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=False)
    return result.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def load_rules(path: Path) -> list[dict[str, Any]]:
    """Valida a estrutura mínima do JSONL antes de iniciar uma inferência."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo {path} ausente. Execute python -m src.pipelines.prepare_references.")
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
            seen.add(rid)
            rules.append(obj)
    return rules


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Grava em arquivo temporário e renomeia: nunca deixa JSON parcial."""
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
        return None  # O item será reprocessado.


def fingerprint(rule: dict[str, Any], *, model: str, num_ctx: int, versao: str) -> str:
    payload = {"pipeline_version": PIPELINE_VERSION, "id": rule["id"], "gobr": rule["gobr"],
               "model": model, "num_ctx": num_ctx, "versao_explicador": versao}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chave_corpo(gobr: str) -> str:
    """Identifica blocos com o mesmo corpo (só o nome no cabeçalho muda).

    O acervo tem 1.700 blocos e só 1.204 corpos distintos; a explicação não
    cita o nome da regra, então blocos repetidos reaproveitam a mesma.
    """
    from src.gobr.grammar import normalizar
    corpo = gobr.strip().splitlines()[1:]
    return hashlib.sha256(normalizar(" ".join(corpo)).encode("utf-8")).hexdigest()


def consolidar(output_dir: Path) -> Path:
    """Junta os itens concluídos em `explicacoes.jsonl`, em ordem de id."""
    destino = output_dir / "explicacoes.jsonl"
    temp = destino.with_suffix(".jsonl.tmp")
    total = 0
    with temp.open("w", encoding="utf-8") as saida:
        for arquivo in sorted((output_dir / "items").glob("*.json")):
            item = read_previous(arquivo)
            if not item or item.get("status") != "completed":
                continue
            explicacao = item["explicacao"]
            saida.write(json.dumps({
                "id": item["id"], "nome": item.get("nome"), "arquivo_origem": item.get("arquivo_origem"),
                "gobr": item["gobr"], "explicacao": explicacao["texto"], "metodo": explicacao["metodo"],
                "aprovada": explicacao["aprovada"], "model": item.get("model"),
            }, ensure_ascii=False) + "\n")
            total += 1
    os.replace(temp, destino)
    print(f"{total} explicações consolidadas em {destino}", flush=True)
    return destino


class ReferenceBatchRunner:
    def __init__(self, *, explicador_factory: Callable[[], Any] | None = None) -> None:
        self.explicador_factory = explicador_factory
        self.explicador: Any = None

    def _carregar(self, sem_llm: bool) -> Any:
        if self.explicador is None:
            if self.explicador_factory is None:
                from src.agents.explicador import AgenteExplicador
                self.explicador_factory = lambda: AgenteExplicador(usar_llm=not sem_llm)
            self.explicador = self.explicador_factory()
        return self.explicador

    @staticmethod
    def _registro(rule: dict[str, Any], signature: str, versao: str, model: str, num_ctx: int,
                  revision: str, attempts: int) -> dict[str, Any]:
        return {
            "id": rule["id"], "nome": rule.get("nome"), "arquivo_origem": rule.get("arquivo_origem"),
            "indice": rule.get("indice"), "linha_inicio": rule.get("linha_inicio"),
            "linha_fim": rule.get("linha_fim"), "gobr": rule["gobr"],
            "referencias_nao_resolvidas": rule.get("referencias_nao_resolvidas", []),
            "fingerprint": signature, "pipeline_version": PIPELINE_VERSION,
            "versao_explicador": versao, "model": model, "num_ctx": num_ctx,
            "code_revision": revision, "attempt": attempts, "created_at_utc": utc_now(),
        }

    def run(
        self, *, input_path: Path, output_dir: Path, limit: int = 3, max_attempts: int = 2,
        model: str = "gemma3:12b", num_ctx: int = 4096, versao: str | None = None,
        revision: str = "unversioned", dry_run: bool = False, force: bool = False, sem_llm: bool = False,
    ) -> dict[str, Any]:
        if limit < 0 or max_attempts < 1 or num_ctx < 1:
            raise ValueError("limit >= 0, max_attempts >= 1 e num_ctx positivo.")
        rules = load_rules(input_path)
        versao = versao or versao_explicador()
        if sem_llm:
            model, num_ctx = "sem-llm", 0
        items_dir = output_dir / "items"
        stats: dict[str, Any] = {
            "total_regras": len(rules), "simuladas": 0, "concluidas_nesta_execucao": 0,
            "falhas_nesta_execucao": 0, "bloqueadas_nesta_execucao": 0,
            "ignoradas_concluidas": 0, "ignoradas_tentativas": 0, "ignoradas_bloqueadas": 0,
            "reaproveitadas_nesta_execucao": 0,
            "metodos_nesta_execucao": {}, "dry_run": dry_run, "model": model, "num_ctx": num_ctx,
            "versao_explicador": versao, "revision": revision,
            "executed_at_utc": utc_now(), "output_dir": str(output_dir),
        }
        executed = 0
        # corpo do bloco → (id, explicação) já concluída nesta pasta
        por_corpo: dict[str, tuple[str, dict[str, Any]]] = {}
        for rule in rules:
            rule_id = rule["id"]
            output = items_dir / f"{rule_id}.json"
            signature = fingerprint(rule, model=model, num_ctx=num_ctx, versao=versao)
            previous = read_previous(output)
            matches = previous is not None and previous.get("fingerprint") == signature
            corpo = chave_corpo(rule["gobr"])
            if not force and matches:
                status = previous.get("status")
                if status == "completed":
                    stats["ignoradas_concluidas"] += 1
                    por_corpo.setdefault(corpo, (rule_id, previous["explicacao"]))
                    continue
                if status == "blocked":
                    stats["ignoradas_bloqueadas"] += 1
                    continue
                if status == "failed" and previous.get("attempt", 0) >= max_attempts:
                    stats["ignoradas_tentativas"] += 1
                    continue
            if corpo in por_corpo and not dry_run:
                # Mesmo corpo de um bloco já explicado: copia, sem chamar o LLM
                # e sem contar no limite do lote.
                origem, explicacao = por_corpo[corpo]
                record = self._registro(rule, signature, versao, model, num_ctx, revision, 1)
                record.update(status="completed", reaproveitada_de=origem,
                              explicacao={**explicacao, "nome": rule.get("nome") or explicacao.get("nome")},
                              elapsed_seconds=0.0)
                atomic_write_json(output, record)
                stats["reaproveitadas_nesta_execucao"] += 1
                print(f"[completed] {rule_id} | reaproveitada de {origem}", flush=True)
                continue
            if limit and executed >= limit:
                break
            executed += 1
            if dry_run:
                stats["simuladas"] += 1
                print(f"[DRY RUN] {rule_id}: {rule.get('nome', '')}")
                continue
            attempts = int(previous.get("attempt", 0)) + 1 if matches and not force and previous else 1
            started = time.monotonic()
            record = self._registro(rule, signature, versao, model, num_ctx, revision, attempts)
            try:
                explicacao = self._carregar(sem_llm).explicar(rule["gobr"])
                dados = explicacao.model_dump(mode="json")
                record["explicacao"] = dados
                if any(not f["conhecido"] for f in dados.get("fatos", [])):
                    record["status"] = "blocked"
                    record["reason"] = "Há trechos sem tradução no glossário; explicação fica para revisão."
                    stats["bloqueadas_nesta_execucao"] += 1
                else:
                    record["status"] = "completed"
                    por_corpo.setdefault(corpo, (rule_id, dados))
                    stats["concluidas_nesta_execucao"] += 1
                    metodo = dados["metodo"]
                    stats["metodos_nesta_execucao"][metodo] = stats["metodos_nesta_execucao"].get(metodo, 0) + 1
            except Exception as exc:
                record.update(status="failed", error={
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                })
                stats["falhas_nesta_execucao"] += 1
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            atomic_write_json(output, record)
            metodo = (record.get("explicacao") or {}).get("metodo", "-")
            print(f"[{record['status']}] {rule_id} | {metodo} | tentativa {attempts} | "
                  f"{record['elapsed_seconds']}s", flush=True)
        if not dry_run:
            atomic_write_json(output_dir / "batch_summary.json", stats)
            if items_dir.exists():
                consolidar(output_dir)
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("outputs/generated/regras_extraidas.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Padrão: outputs/explicacoes (LLM) ou outputs/explicacoes_rascunho (--sem-llm).")
    parser.add_argument("--limit", type=int, default=3,
                        help="Máximo de blocos processados nesta execução; 0 = sem limite.")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refaz blocos já concluídos.")
    parser.add_argument("--sem-llm", action="store_true",
                        help="Só o rascunho determinístico (não precisa de Ollama).")
    args = parser.parse_args()
    from src.llm_config import contexto_padrao, modelo_padrao
    output_dir = args.output_dir or Path("outputs/explicacoes_rascunho" if args.sem_llm else "outputs/explicacoes")
    ReferenceBatchRunner().run(
        input_path=args.input, output_dir=output_dir, limit=args.limit, max_attempts=args.max_attempts,
        model=modelo_padrao(), num_ctx=contexto_padrao(), revision=code_revision(),
        dry_run=args.dry_run, force=args.force, sem_llm=args.sem_llm,
    )


if __name__ == "__main__":
    main()
