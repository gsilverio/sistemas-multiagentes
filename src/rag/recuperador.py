"""API de recuperação usada pelos agentes: `buscar_manual()` e `buscar_exemplos()`.

Os índices ficam em `outputs/rag/manual` e `outputs/rag/exemplos` (gerados por
`python -m src.pipelines.construir_indices`) e são carregados uma vez por
processo. Sem sentence-transformers ou sem `vetores.npy`, a busca cai para
BM25 e o campo `modo` dos resultados mostra isso.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from src.rag.busca_hibrida import IndiceHibrido

RAIZ = Path(__file__).resolve().parents[2]
DIR_INDICES = RAIZ / "outputs" / "rag"


class TrechoManual(BaseModel):
    id: str
    secao: str
    titulo: str
    caminho: str
    construcao: str
    paginas: list[int]
    texto: str
    score: float
    modo: str

    def citacao(self) -> str:
        paginas = ", ".join(map(str, self.paginas))
        return f"manual GOBR, seção {self.secao} ({self.titulo}), p. {paginas}"


class Exemplo(BaseModel):
    id: str
    nome: str
    gobr: str
    contexto: list[str]
    explicacao: str
    metodo: str
    descricao_cliente: str | None = None
    assinatura: str
    construcoes: list[str]
    score: float
    modo: str


@lru_cache(maxsize=4)
def carregar_indice(diretorio: str) -> IndiceHibrido:
    caminho = Path(diretorio)
    if not (caminho / "documents.json").exists():
        raise FileNotFoundError(
            f"Índice ausente em {caminho}. Gere com: python -m src.pipelines.construir_indices")
    return IndiceHibrido.carregar(caminho)


def buscar_manual(consulta: str, *, k: int = 4, construcao: str | Iterable[str] | None = None,
                  modo: str = "hibrido", diretorio: Path | None = None) -> list[TrechoManual]:
    """Trechos do manual relevantes para `consulta`.

    `construcao` restringe a uma parte da linguagem (ex.: "parametros",
    "filtro_cliente", "filtro_temporal"); ver `src.rag.manual.CONSTRUCAO_POR_CAPITULO`.
    """
    indice = carregar_indice(str(diretorio or DIR_INDICES / "manual"))
    permitidas = {construcao} if isinstance(construcao, str) else set(construcao or ())
    filtro = (lambda d: d["metadata"]["construcao"] in permitidas) if permitidas else None
    return [
        TrechoManual(id=r["id"], secao=r["metadata"]["secao"], titulo=r["metadata"]["titulo"],
                     caminho=r["metadata"]["caminho"], construcao=r["metadata"]["construcao"],
                     paginas=r["metadata"]["paginas"], texto=r["text"], score=r["score"], modo=r["modo"])
        for r in indice.buscar(consulta, k=k, modo=modo, filtro=filtro)
    ]


def buscar_exemplos(consulta: str, *, k: int = 3, modo: str = "hibrido", diversificar: bool = True,
                    exigir: Iterable[str] = (), diretorio: Path | None = None) -> list[Exemplo]:
    """Regras do acervo parecidas com a descrição `consulta`.

    `diversificar` evita devolver duas regras com a mesma forma (mesma
    assinatura estrutural, só números diferentes): para few-shot, variedade
    ensina mais que repetição. `exigir` filtra por construções (ex.:
    ["nao_agendar"], ["frequencia_dinamica"]); ver `src.gobr.assinatura.construcoes`.
    """
    indice = carregar_indice(str(diretorio or DIR_INDICES / "exemplos"))
    exigidas = set(exigir)
    filtro = (lambda d: exigidas <= set(d["metadata"]["construcoes"])) if exigidas else None
    candidatos = indice.buscar(consulta, k=k * 4 if diversificar else k, modo=modo, filtro=filtro)
    escolhidos, formas = [], set()
    for r in candidatos:
        m = r["metadata"]
        if diversificar and m["assinatura"] in formas:
            continue
        formas.add(m["assinatura"])
        escolhidos.append(Exemplo(id=r["id"], nome=m["nome"], gobr=m["gobr"], contexto=m["contexto"],
                                  explicacao=m["explicacao"], metodo=m["metodo"],
                                  descricao_cliente=m.get("descricao_cliente"), assinatura=m["assinatura"],
                                  construcoes=m["construcoes"], score=r["score"], modo=r["modo"]))
        if len(escolhidos) == k:
            break
    return escolhidos
