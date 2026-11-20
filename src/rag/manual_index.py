"""Build and query a persistent FAISS index for the GOBR manual.

The manual index is intentionally kept separate from the explanation index.
It is an internal retrieval source for the agent: a user description (or an
image converted to a description) can retrieve the relevant syntax and
semantics before the agent emits GOBR.  Page and chunk metadata are preserved
so a later user-facing answer can cite the manual without changing the index.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def clean_text(text: str) -> str:
    """Normalize PDF extraction noise while preserving paragraph boundaries."""
    lines = []
    for line in text.replace("\r", "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(_repair_spaced_line(line))
    return "\n".join(lines)


def normalize_for_embedding(text: str) -> str:
    """Create a search-only representation tolerant to accents and bad glyphs.

    The original ``text`` is never replaced. This normalized form is used only
    to make queries such as ``frequencia`` and ``frequência`` land near the
    same manual passage, and to discard a real U+FFFD replacement glyph when a
    PDF font has no Unicode mapping.
    """
    text = text.replace("\ufffd", " ")
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    without_symbols = re.sub(r"[^\w\s@#<>+\-]", " ", without_accents.casefold())
    return re.sub(r"\s+", " ", without_symbols).strip()


def _repair_spaced_line(line: str) -> str:
    """Repair headings that PDF extraction returns as ``G o v e n d a s``."""
    tokens = line.split()
    if len(tokens) < 5:
        return line
    single_letter_ratio = sum(len(token) == 1 and token.isalpha() for token in tokens) / len(tokens)
    if single_letter_ratio < 0.55:
        return line
    prefix = []
    while tokens and (len(tokens[0]) == 1 and not tokens[0].isalpha()):
        prefix.append(tokens.pop(0))
    repaired = "".join(tokens)
    return f"{' '.join(prefix)} {repaired}".strip() if prefix else repaired


def chunk_text(text: str, *, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    """Split extracted text into bounded chunks with a small textual overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars deve ser positivo.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap deve estar entre zero e max_chars - 1.")

    normalized = clean_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    current = ""
    for paragraph in normalized.split("\n"):
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max_chars - overlap
            for start in range(0, len(paragraph), step):
                part = paragraph[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current)
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
            if len(current) > max_chars:
                current = current[-max_chars:]
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def documents_from_pages(
    pages: Iterable[tuple[int, str]],
    *,
    source_pdf: str,
    max_chars: int = 1800,
    overlap: int = 250,
) -> list[dict[str, Any]]:
    """Turn ``(page_number, text)`` pairs into embedding documents."""
    documents: list[dict[str, Any]] = []
    for page_number, raw_text in pages:
        text = clean_text(raw_text)
        if not text:
            continue
        title = next(
            (
                line
                for line in text.splitlines()
                if line.strip() and set(line.strip()) - {"."}
            ),
            f"Página {page_number}",
        )
        for chunk_number, chunk in enumerate(
            chunk_text(text, max_chars=max_chars, overlap=overlap), start=1
        ):
            document_id = f"manual_p{page_number:03d}_c{chunk_number:02d}"
            documents.append(
                {
                    "id": document_id,
                    "text": chunk,
                    "metadata": {
                        "source": "gobr_manual",
                        "arquivo_origem": source_pdf,
                        "pagina": page_number,
                        "trecho": chunk_number,
                        "titulo_pagina": title[:240],
                    },
                }
            )
    return documents


def load_documents(
    pdf_path: Path,
    *,
    max_chars: int = 1800,
    overlap: int = 250,
) -> list[dict[str, Any]]:
    """Extract the PDF and prepare page/chunk documents."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise RuntimeError("Instale pypdf para indexar o manual: pip install pypdf") from exc

    reader = PdfReader(str(pdf_path))
    pages = ((number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1))
    return documents_from_pages(
        pages,
        source_pdf=pdf_path.name,
        max_chars=max_chars,
        overlap=overlap,
    )


def build_index(
    documents: list[dict[str, Any]],
    output_dir: Path,
    *,
    model_name: str,
    source_pdf: str,
) -> dict[str, Any]:
    """Embed documents and persist FAISS index plus JSON metadata."""
    if not documents:
        raise ValueError("Nenhum texto foi extraído do manual.")

    from sentence_transformers import SentenceTransformer
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _cuda_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        [normalize_for_embedding(doc["text"]) for doc in documents],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "manual.faiss"))
    (output_dir / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pages = sorted({doc["metadata"]["pagina"] for doc in documents})
    manifest = {
        "kind": "gobr_manual_index",
        "count": len(documents),
        "pages": len(pages),
        "dimension": int(vectors.shape[1]),
        "embedding_model": model_name,
        "device": device,
        "source_pdf": source_pdf,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def search_index(index_dir: Path, query: str, *, k: int = 4) -> list[dict[str, Any]]:
    """Search a persisted manual index with semantic and lexical ranking.

    FAISS is excellent for concepts, but GOBR also contains exact DSL clauses
    such as ``nao agendar``.  We therefore retrieve all small-manual
    candidates, then add a lexical overlap/phrase bonus before returning the
    top ``k`` results.  This keeps semantic matches while prioritizing an
    exact syntax occurrence when one exists.
    """
    if k <= 0:
        return []
    from sentence_transformers import SentenceTransformer
    import faiss

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(
        manifest["embedding_model"],
        device="cuda" if _cuda_available() else "cpu",
    )
    vector = model.encode(
        [normalize_for_embedding(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    index = faiss.read_index(str(index_dir / "manual.faiss"))
    scores, positions = index.search(vector, index.ntotal)
    documents = json.loads((index_dir / "documents.json").read_text(encoding="utf-8"))
    query_normalized = normalize_for_embedding(query)
    query_tokens = set(query_normalized.split())
    ranked: list[dict[str, Any]] = []
    for i, pos in enumerate(positions[0]):
        if pos < 0:
            continue
        document = documents[pos]
        document_normalized = normalize_for_embedding(document["text"])
        document_tokens = set(document_normalized.split())
        lexical_score = (
            len(query_tokens & document_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        phrase_bonus = 0.35 if query_normalized in document_normalized else 0.0
        semantic_score = float(scores[0][i])
        hybrid_score = semantic_score + 0.25 * lexical_score + phrase_bonus
        ranked.append(
            {
                "score": hybrid_score,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                **document,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: min(k, len(ranked))]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    args = parser.parse_args()

    documents = load_documents(
        args.pdf,
        max_chars=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    manifest = build_index(
        documents,
        args.output_dir,
        model_name=args.model,
        source_pdf=args.pdf.name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
