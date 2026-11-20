"""Agente Recuperador: para cada segmento, exemplos do acervo e trechos do manual.

Sem LLM: usa a busca híbrida da etapa 3 (`src.rag.recuperador`). A consulta
é a descrição do segmento, no mesmo formato que o Gerador recebe. Segmentos
`nao agendar` pedem exemplos que também tenham `nao agendar`, para o modelo
não copiar uma frequência de um exemplo parecido.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.agents.contratos import ContextoSegmento, RuleSpec, Segmento, TrechoContexto
from src.rag import recuperador as rag

LIMITE_TRECHO_MANUAL = 1200


class AgenteRecuperador:
    def __init__(self, *, k_exemplos: int = 3, k_manual: int = 2, modo: str = "hibrido",
                 diretorio: Path | None = None,
                 buscar_exemplos: Callable = rag.buscar_exemplos, buscar_manual: Callable = rag.buscar_manual):
        self.k_exemplos, self.k_manual, self.modo = k_exemplos, k_manual, modo
        self.diretorio = diretorio
        self._exemplos, self._manual = buscar_exemplos, buscar_manual

    def _dir(self, nome: str) -> dict:
        return {"diretorio": self.diretorio / nome} if self.diretorio else {}

    def contexto(self, segmento: Segmento) -> ContextoSegmento:
        consulta = f"{segmento.nome}. {segmento.descricao()}"
        exigir = ["nao_agendar"] if segmento.nao_agendar else []
        exemplos = self._exemplos(consulta, k=self.k_exemplos, modo=self.modo, exigir=exigir, **self._dir("exemplos"))
        manual = self._manual(consulta, k=self.k_manual, modo=self.modo,
                              construcao=["filtro_cliente", "parametros"], **self._dir("manual"))
        return ContextoSegmento(
            exemplos=[TrechoContexto(titulo=e.nome, texto=f"{e.explicacao}\n---\n{e.gobr}") for e in exemplos],
            manual=[TrechoContexto(titulo=f"{t.secao} {t.titulo}", texto=t.texto[:LIMITE_TRECHO_MANUAL])
                    for t in manual],
        )

    def recuperar(self, spec: RuleSpec) -> dict[str, ContextoSegmento]:
        return {s.nome: self.contexto(s) for s in spec.segmentos}
