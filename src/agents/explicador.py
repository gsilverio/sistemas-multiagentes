"""Agente Explicador: bloco GOBR → explicação comercial em português.

Fluxo (etapa 2):

    GOBR ──parser──► AST ──glossário──► rascunho (fatos + texto)
                                            │
                              1 chamada LLM ▼ (reescrita natural)
                                         guard ──ok──► explicação "llm"
                                            │ falhou
                     1 chamada de correção  ▼ (com os problemas apontados)
                                         guard ──ok──► explicação "llm_corrigido"
                                            │ falhou
                                            ▼
                                  rascunho determinístico ("rascunho")

Verificação cega: o agente recebe apenas o GOBR, nunca a descrição que o
originou, para que a explicação sirva de conferência do Gerador (etapa 4).
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.explicacao.guarda import verificar
from src.explicacao.rascunho import Fato, rascunho_de_gobr
from src.prompts import explicador_prompt

Metodo = Literal["llm", "llm_corrigido", "rascunho"]


class Explicacao(BaseModel):
    nome: str
    texto: str
    metodo: Metodo
    aprovada: bool                 # passou no guard (o rascunho sempre passa, salvo cláusulas desconhecidas)
    problemas: list[str] = Field(default_factory=list)   # do guard, na última tentativa do LLM
    rascunho: str
    tentativas_llm: list[str] = Field(default_factory=list)
    fatos: list[Fato] = Field(default_factory=list)


def _limpar(resposta: Any) -> str:
    texto = getattr(resposta, "content", resposta)
    if isinstance(texto, list):  # alguns modelos devolvem partes
        texto = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in texto)
    texto = re.sub(r"<think>.*?</think>", "", str(texto), flags=re.DOTALL)
    return texto.strip().strip('`').strip()


class AgenteExplicador:
    def __init__(self, llm: Any = None, *, usar_llm: bool = True, max_correcoes: int = 1):
        self._llm = llm
        self.usar_llm = usar_llm
        self.max_correcoes = max_correcoes

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from src.llm_config import create_llm  # Importa LangChain só quando precisa.
            self._llm = create_llm()
        return self._llm

    def explicar(self, gobr: str) -> Explicacao:
        regra, rascunho = rascunho_de_gobr(gobr)
        base = dict(nome=regra.nome, rascunho=rascunho.texto, fatos=rascunho.fatos)
        conferencia_rascunho = verificar(rascunho, rascunho.texto, gobr=gobr)

        if not self.usar_llm or rascunho.desconhecidos:
            # Com cláusulas desconhecidas o LLM tenderia a inventar significado.
            return Explicacao(texto=rascunho.texto, metodo="rascunho",
                              aprovada=conferencia_rascunho.aprovado and not rascunho.desconhecidos,
                              problemas=conferencia_rascunho.problemas, **base)

        tentativas: list[str] = []
        texto = _limpar(self.llm.invoke(explicador_prompt.mensagens(gobr, rascunho)))
        tentativas.append(texto)
        resultado = verificar(rascunho, texto, gobr=gobr)
        if resultado.aprovado:
            return Explicacao(texto=texto, metodo="llm", aprovada=True, tentativas_llm=tentativas, **base)

        for _ in range(self.max_correcoes):
            mensagens = explicador_prompt.mensagens_correcao(gobr, rascunho, texto, resultado.como_texto())
            texto = _limpar(self.llm.invoke(mensagens))
            tentativas.append(texto)
            resultado = verificar(rascunho, texto, gobr=gobr)
            if resultado.aprovado:
                return Explicacao(texto=texto, metodo="llm_corrigido", aprovada=True,
                                  tentativas_llm=tentativas, **base)

        return Explicacao(texto=rascunho.texto, metodo="rascunho", aprovada=conferencia_rascunho.aprovado,
                          problemas=resultado.problemas, tentativas_llm=tentativas, **base)
