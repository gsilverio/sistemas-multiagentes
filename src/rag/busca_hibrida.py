"""Busca híbrida: BM25 (lexical) + embeddings (semântica), fundidos por RRF.

Por que as duas:
- O GOBR tem termos exatos que precisam bater (`nao agendar`, `recencia de
  positivacao`, `roteirizavel`); embeddings pequenos confundem esses termos
  com paráfrases parecidas. O BM25 acerta esses casos.
- O usuário descreve a regra com as próprias palavras ("clientes que não
  compram há 2 meses"); aí o BM25 falha e os embeddings acertam.

A fusão é por Reciprocal Rank Fusion (Cormack et al., 2009): cada lista
contribui com 1/(k + posição). Não depende da escala dos scores, então não
precisa de pesos calibrados.

O BM25 é implementado aqui (sem dependência nova). Os embeddings usam
sentence-transformers e são opcionais: sem o pacote ou sem `vetores.npy`, a
busca continua só com BM25 e o resultado informa `modo="bm25"`.

Formato persistido de um índice (diretório):
    documents.json   lista de {"id", "text", "texto_lexico"?, "metadata"}
    manifest.json    tipo, contagem, modelo de embeddings, dimensão
    vetores.npy      float32 [n_docs, dim], normalizados (opcional)
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
RRF_K = 60

_STOPWORDS = frozenset(
    "a o as os e de da do das dos em no na nos nas um uma uns umas para por pelo pela pelos pelas "
    "com que se ao aos ou como mais seu sua seus suas este esta estes estas esse essa isso ser sao "
    "eh ja tambem quando cada".split()
)


def normalizar_busca(texto: str) -> str:
    """Minúsculas, sem acentos e sem pontuação (mantém `@`, `#`, `<`, `>`)."""
    texto = texto.replace("�", " ")
    sem_acentos = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s@#<>]", " ", sem_acentos.casefold())).strip()


def _radical(token: str) -> str:
    """Stemming mínimo do português: plural e algumas flexões comuns."""
    for sufixo, troca in (("coes", "cao"), ("oes", "ao"), ("aes", "ao"), ("ais", "al"), ("eis", "el")):
        if len(token) > 5 and token.endswith(sufixo):
            return token[: -len(sufixo)] + troca
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokenizar(texto: str) -> list[str]:
    return [_radical(t) for t in normalizar_busca(texto).split() if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 clássico (k1=1.5, b=0.75)."""

    def __init__(self, textos: Iterable[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenizar(t)) for t in textos]
        self.tamanhos = [sum(d.values()) for d in self.docs]
        self.media = (sum(self.tamanhos) / len(self.docs)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {termo: math.log(1 + (n - f + 0.5) / (f + 0.5)) for termo, f in df.items()}

    def scores(self, consulta: str) -> list[float]:
        termos = tokenizar(consulta)
        resultado = []
        for doc, tamanho in zip(self.docs, self.tamanhos):
            s = 0.0
            for termo in termos:
                tf = doc.get(termo)
                if not tf:
                    continue
                norm = self.k1 * (1 - self.b + self.b * tamanho / (self.media or 1))
                s += self.idf[termo] * tf * (self.k1 + 1) / (tf + norm)
            resultado.append(s)
        return resultado


def fusao_rrf(rankings: list[list[int]], *, k: int = RRF_K) -> dict[int, float]:
    """Funde listas de posições de documentos (melhor primeiro)."""
    total: dict[int, float] = {}
    for ranking in rankings:
        for posicao, doc in enumerate(ranking, start=1):
            total[doc] = total.get(doc, 0.0) + 1.0 / (k + posicao)
    return total


class Codificador:
    """Embeddings com sentence-transformers, carregado só quando usado.

    Modelos E5 exigem os prefixos `query: ` / `passage: `.
    """

    _cache: dict[str, Any] = {}

    def __init__(self, modelo: str = MODELO_EMBEDDINGS) -> None:
        self.modelo = modelo
        self._e5 = "e5" in modelo.lower()

    def _carregar(self):
        if self.modelo not in self._cache:
            from sentence_transformers import SentenceTransformer

            self._cache[self.modelo] = SentenceTransformer(self.modelo, device="cuda" if _cuda() else "cpu")
        return self._cache[self.modelo]

    def codificar(self, textos: list[str], *, consulta: bool = False):
        prefixo = ("query: " if consulta else "passage: ") if self._e5 else ""
        return self._carregar().encode(
            [prefixo + t.replace("�", " ") for t in textos],
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=len(textos) > 64, batch_size=32,
        ).astype("float32")


def embeddings_disponiveis() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


Filtro = Callable[[dict[str, Any]], bool]


class IndiceHibrido:
    def __init__(self, documentos: list[dict[str, Any]], *, vetores=None, modelo: str | None = None,
                 tipo: str = "indice") -> None:
        self.documentos = documentos
        self.vetores = vetores
        self.modelo = modelo
        self.tipo = tipo
        self.bm25 = BM25(d.get("texto_lexico") or d["text"] for d in documentos)

    # ---------- construção e persistência ----------
    @classmethod
    def construir(cls, documentos: list[dict[str, Any]], *, tipo: str, modelo: str | None = MODELO_EMBEDDINGS,
                  com_vetores: bool = True) -> "IndiceHibrido":
        vetores = None
        if com_vetores and modelo:
            vetores = Codificador(modelo).codificar([d["text"] for d in documentos])
        return cls(documentos, vetores=vetores, modelo=modelo if vetores is not None else None, tipo=tipo)

    def salvar(self, diretorio: Path, **extra: Any) -> dict[str, Any]:
        diretorio.mkdir(parents=True, exist_ok=True)
        (diretorio / "documents.json").write_text(
            json.dumps(self.documentos, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        vet = diretorio / "vetores.npy"
        if self.vetores is not None:
            import numpy as np

            np.save(vet, self.vetores)
        elif vet.exists():
            vet.unlink()  # vetores antigos não correspondem mais aos documentos
        manifesto = {
            "kind": self.tipo,
            "count": len(self.documentos),
            "embedding_model": self.modelo,
            "dimension": int(self.vetores.shape[1]) if self.vetores is not None else None,
            **extra,
        }
        (diretorio / "manifest.json").write_text(
            json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifesto

    @classmethod
    def carregar(cls, diretorio: Path) -> "IndiceHibrido":
        manifesto = json.loads((diretorio / "manifest.json").read_text(encoding="utf-8"))
        documentos = json.loads((diretorio / "documents.json").read_text(encoding="utf-8"))
        vetores = None
        if (diretorio / "vetores.npy").exists() and manifesto.get("embedding_model"):
            import numpy as np

            vetores = np.load(diretorio / "vetores.npy")
            if len(vetores) != len(documentos):
                raise ValueError(f"{diretorio}: vetores.npy não corresponde a documents.json; reconstrua o índice.")
        return cls(documentos, vetores=vetores, modelo=manifesto.get("embedding_model"),
                   tipo=manifesto.get("kind", "indice"))

    # ---------- busca ----------
    def modo(self, modo: str = "hibrido") -> str:
        if modo == "bm25" or self.vetores is None or not embeddings_disponiveis():
            return "bm25"
        return modo

    def buscar(self, consulta: str, *, k: int = 4, modo: str = "hibrido",
               filtro: Filtro | None = None, candidatos: int = 50) -> list[dict[str, Any]]:
        """`modo`: hibrido | bm25 | denso. Cai para bm25 se não houver embeddings."""
        if k <= 0 or not consulta.strip() or not self.documentos:
            return []
        modo = self.modo(modo)
        validos = [i for i, d in enumerate(self.documentos) if filtro is None or filtro(d)]
        if not validos:
            return []

        rankings: list[list[int]] = []
        lexicos = self.bm25.scores(consulta)
        ranking_bm25 = sorted((i for i in validos if lexicos[i] > 0), key=lambda i: -lexicos[i])[:candidatos]
        semanticos = None
        if modo in ("hibrido", "denso"):
            vetor = Codificador(self.modelo).codificar([consulta], consulta=True)[0]
            semanticos = self.vetores @ vetor
            ranking_denso = sorted(validos, key=lambda i: -float(semanticos[i]))[:candidatos]
        if modo in ("hibrido", "bm25"):
            rankings.append(ranking_bm25)
        if modo in ("hibrido", "denso"):
            rankings.append(ranking_denso)

        fundidos = fusao_rrf(rankings)
        ordem = sorted(fundidos, key=lambda i: -fundidos[i])[:k]
        pos_bm25 = {doc: p for p, doc in enumerate(ranking_bm25, start=1)}
        return [
            {
                "score": round(fundidos[i], 6),
                "modo": modo,
                "bm25": round(lexicos[i], 4),
                "posicao_bm25": pos_bm25.get(i),
                "denso": round(float(semanticos[i]), 4) if semanticos is not None else None,
                **self.documentos[i],
            }
            for i in ordem
        ]


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
