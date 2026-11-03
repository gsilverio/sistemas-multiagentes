"""Build a persistent FAISS index from generated GOBR explanations.

The Colab notebook runs this after ``batch_reference_explanations``.  By
default only explicitly approved items are indexed.  ``--include-pending`` is
an intentional fast path for the coursework smoke test; it keeps the review
status in metadata so the index can be rebuilt later after editorial review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_documents(items_dir: Path, *, include_pending: bool = False) -> list[dict[str, Any]]:
    """Load completed explanation records as embedding documents."""
    documents: list[dict[str, Any]] = []
    for path in sorted(items_dir.glob("*.json")):
        if path.name == "batch_summary.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        explanation = record.get("explanation") or {}
        text = (explanation.get("explicacao") or "").strip()
        if not text:
            continue
        review = explanation.get("revisao") or {}
        approved = bool(record.get("indexable"))
        if not approved and not include_pending:
            continue
        documents.append(
            {
                "id": record.get("id", path.stem),
                "text": text,
                "metadata": {
                    "id": record.get("id", path.stem),
                    "nome": record.get("nome"),
                    "arquivo_origem": record.get("arquivo_origem"),
                    "linha_inicio": record.get("linha_inicio"),
                    "linha_fim": record.get("linha_fim"),
                    "gobr": record.get("gobr"),
                    "review_status": review.get("status"),
                    "review_method": review.get("metodo"),
                    "approved": approved,
                    "provisional": not approved,
                },
            }
        )
    return documents


def build_index(documents: list[dict[str, Any]], output_dir: Path, *, model_name: str) -> dict[str, Any]:
    """Embed documents and persist FAISS index plus JSON metadata."""
    if not documents:
        raise ValueError("Nenhuma explicação elegível para indexação.")
    from sentence_transformers import SentenceTransformer
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _cuda_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        [doc["text"] for doc in documents],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "explanations.faiss"))
    (output_dir / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "kind": "gobr_explanation_index",
        "count": len(documents),
        "dimension": int(vectors.shape[1]),
        "embedding_model": model_name,
        "device": device,
        "provisional_count": sum(d["metadata"]["provisional"] for d in documents),
        "approved_count": sum(d["metadata"]["approved"] for d in documents),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def search_index(index_dir: Path, query: str, *, k: int = 3) -> list[dict[str, Any]]:
    """Search a persisted explanation index."""
    from sentence_transformers import SentenceTransformer
    import faiss

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(manifest["embedding_model"], device="cuda" if _cuda_available() else "cpu")
    vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, positions = faiss.read_index(str(index_dir / "explanations.faiss")).search(vector, k)
    documents = json.loads((index_dir / "documents.json").read_text(encoding="utf-8"))
    return [
        {"score": float(scores[0][i]), **documents[pos]}
        for i, pos in enumerate(positions[0])
        if pos >= 0
    ]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args()
    documents = load_documents(args.items_dir, include_pending=args.include_pending)
    manifest = build_index(documents, args.output_dir, model_name=args.model)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
