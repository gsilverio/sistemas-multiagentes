"""Agente Gerador: segmento do `RuleSpec` + contexto do RAG → bloco `parametros`.

Gera um bloco por segmento (é a unidade que o fine-tuning da etapa 6 aprende)
e monta o arquivo de forma determinística: `version 1.0`, o bloco `time` e os
blocos na ordem do `RuleSpec` (já ajustada pelo Analista). Na correção, só
os blocos com erro voltam ao modelo, junto com os erros do Validador.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.comum import texto_da_resposta
from src.agents.contratos import ContextoSegmento, RuleSpec, Segmento
from src.prompts import gerador_prompt

_ESCOPO = re.compile(r"^\s*(?:do|da|dos|das)\s+(?:time|carteira|segmenta[cç][aã]o)\b.*$", re.IGNORECASE | re.MULTILINE)
_OUTRO_BLOCO = re.compile(r"^(?:version|time|carteira|segmentacao|segmentação|parametros|alertas?|tags?)\b",
                          re.IGNORECASE)


def escopo_do_time(spec: RuleSpec) -> str:
    return f'do time("{spec.time}")'


def bloco_time(spec: RuleSpec) -> str:
    return f'time "{spec.time}":\n    vendedores {spec.vendedores} ativados'


def escopo_de(gobr: str) -> str:
    m = _ESCOPO.search(gobr)
    return m.group(0).strip() if m else 'do time("televendas")'


def limpar_bloco(resposta: str, nome: str) -> str:
    """Isola o bloco da resposta, força o nome pedido e normaliza a indentação."""
    texto = re.sub(r"```\w*", "", resposta).strip()
    inicio = re.search(r'^[ \t]*parametros\s+"', texto, flags=re.MULTILINE | re.IGNORECASE)
    linhas = texto[inicio.start():].splitlines() if inicio else texto.splitlines()
    corpo: list[str] = []
    for linha in linhas[1:] if inicio else linhas:
        if not linha.strip():
            continue
        if not linha.startswith((" ", "\t")) and (_OUTRO_BLOCO.match(linha.strip()) or corpo):
            break  # outro bloco ou comentário do modelo depois do bloco
        corpo.append("    " + linha.strip())
    nome_seguro = nome.replace('"', "'")
    return "\n".join([f'parametros "{nome_seguro}":', *corpo])


class AgenteGerador:
    def __init__(self, llm: Any = None):
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from src.llm_config import create_llm_para

            self._llm = create_llm_para("gerador")
        return self._llm

    def _mensagens(self, segmento: Segmento, escopo: str, contexto: ContextoSegmento | None) -> list:
        contexto = contexto or ContextoSegmento()
        exemplos = []
        for ex in contexto.exemplos:
            descricao, _, bloco = ex.texto.partition("\n---\n")
            exemplos.append((ex.titulo, escopo_de(bloco), descricao, bloco))
        manual = [(t.titulo, t.texto) for t in contexto.manual]
        return gerador_prompt.mensagens(segmento.nome, escopo, segmento.descricao(), exemplos=exemplos, manual=manual)

    def gerar_bloco(self, segmento: Segmento, escopo: str, contexto: ContextoSegmento | None = None, *,
                    anterior: str | None = None, erros: list[str] | None = None) -> str:
        mensagens = self._mensagens(segmento, escopo, contexto)
        if anterior and erros:
            mensagens = gerador_prompt.mensagens_correcao(mensagens, anterior, erros)
        return limpar_bloco(texto_da_resposta(self.llm.invoke(mensagens)), segmento.nome)

    def gerar(self, spec: RuleSpec, contextos: dict[str, ContextoSegmento], *,
              anteriores: dict[str, str] | None = None,
              erros_por_bloco: dict[str, list[str]] | None = None) -> dict[str, str]:
        """Blocos por nome de segmento. Com `anteriores`, só refaz os que têm erro."""
        anteriores = anteriores or {}
        erros_por_bloco = erros_por_bloco or {}
        escopo = escopo_do_time(spec)
        blocos: dict[str, str] = {}
        for segmento in spec.segmentos:
            nome = segmento.nome
            if nome in anteriores and nome not in erros_por_bloco:
                blocos[nome] = anteriores[nome]
                continue
            blocos[nome] = self.gerar_bloco(segmento, escopo, contextos.get(nome),
                                            anterior=anteriores.get(nome), erros=erros_por_bloco.get(nome))
        return blocos


def montar_arquivo(spec: RuleSpec, blocos: dict[str, str]) -> str:
    partes = ["version 1.0", bloco_time(spec), *(blocos[s.nome] for s in spec.segmentos if s.nome in blocos)]
    return "\n\n".join(partes) + "\n"
