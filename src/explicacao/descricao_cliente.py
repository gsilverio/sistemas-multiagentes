"""Descrição curta no estilo das tabelas dos clientes, gerada por LLM e conferida.

Complementa a explicação da etapa 2 (completa e formal): esta é a forma como o
cliente escreveria o mesmo grupo. Serve ao RAG (consultas do usuário se
parecem com ela) e ao fine-tuning (o Gerador recebe descrições assim no uso
real).

A conferência é mais leve que o guard da explicação, porque a descrição curta
omite de propósito o escopo e as nuances de redação:
- todo número das condições, frequência e modificadores está no texto, e
  nenhum número novo;
- todo valor entre aspas das condições e modificadores aparece literalmente;
- "não agendar" aparece se, e só se, a regra tem `nao agendar`;
- nada de implementação, justificativa ou texto longo demais.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.agents.comum import texto_da_resposta
from src.explicacao import glossario
from src.explicacao.guarda import _IMPLEMENTACAO, _JUSTIFICATIVA, _numeros
from src.explicacao.rascunho import rascunho_de_gobr
from src.gobr.grammar import normalizar
from src.prompts import descricao_cliente_prompt as prompt

TAMANHO_MAXIMO = 400
_NAO_AGENDAR = re.compile(r"\bnao\s+(?:\w+\s+)?agend")


def conferir(gobr: str, texto: str) -> list[str]:
    regra, rascunho = rascunho_de_gobr(gobr)
    t = normalizar(texto)
    if not t:
        return ["Texto vazio."]
    problemas: list[str] = []
    trechos = [c.texto for c in regra.condicoes] + regra.modificadores + regra.filtros_temporais
    if regra.frequencia:
        trechos.append(regra.frequencia)
    obrigatorios = _numeros(normalizar(" ".join(trechos)))
    corpo = "\n".join(gobr.strip().splitlines()[1:])
    permitidos = obrigatorios | _numeros(normalizar(corpo)) | _numeros(normalizar(rascunho.texto))
    presentes = _numeros(t)
    problemas += [f"Falta o número {n}." for n in sorted(obrigatorios - presentes, key=int)]
    problemas += [f"O número {n} não existe na regra." for n in sorted(presentes - permitidos, key=int)]
    valores = list(dict.fromkeys(v for trecho in trechos for v in glossario.valores_entre_aspas(trecho)))
    for valor in valores:
        if valor.strip() and normalizar(valor) not in t:
            problemas.append(f'Falta o valor "{valor}".')
    permitidos_aspas = {normalizar(v) for v in valores}
    for citado in re.findall(r'"([^"]+)"', texto):
        if normalizar(citado) not in permitidos_aspas:
            problemas.append(f'"{citado}" não é um valor da regra; não cite nomes de time ou carteira.')
    # Frequência e modificadores: as mesmas palavras-chave do guard da etapa 2
    # (ex.: "frequencia de inativos da empresa" precisa citar inativos e empresa).
    # "medi" (média) é redação da explicação longa, dispensada aqui.
    for fato in rascunho.fatos:
        if fato.tipo in ("frequencia", "modificador"):
            faltando = [c for c in fato.chaves if c != "medi" and not re.search(c, t)]
            if faltando:
                problemas.append(f'"{fato.gobr}" não foi descrito: {fato.significado}')
    for escopo in regra.escopos:
        for nome in escopo.valores:
            if len(nome) > 3 and re.search(rf"(?<!\w){re.escape(normalizar(nome))}(?!\w)", t) and not any(
                    normalizar(nome) in normalizar(trecho) for trecho in trechos):
                problemas.append(f'Não cite o nome "{nome}" (time/carteira); descreva só os clientes.')
    if regra.nao_agendar and not _NAO_AGENDAR.search(t):
        problemas.append('A regra é de "não agendar"; a descrição precisa dizer isso.')
    if not regra.nao_agendar and _NAO_AGENDAR.search(t):
        problemas.append('A regra tem frequência; a descrição não pode dizer "não agendar".')
    if _IMPLEMENTACAO.search(t):
        problemas.append("A descrição cita a implementação (GOBR, parser...).")
    if m := _JUSTIFICATIVA.search(t):
        problemas.append(f'A descrição explica o motivo ("{m.group(0)}"); descreva só a regra.')
    limite = max(TAMANHO_MAXIMO, len(rascunho.texto))  # regras longas podem ter descrição mais longa
    if len(texto) > limite:
        problemas.append(f"Longa demais ({len(texto)} caracteres); use no máximo duas frases curtas.")
    return problemas


@dataclass
class DescricaoCliente:
    texto: str | None
    metodo: str                      # llm | llm_corrigido | falhou
    problemas: list[str] = field(default_factory=list)
    tentativas: list[str] = field(default_factory=list)


def _exemplos() -> list[tuple[str, str, str]]:
    return [(g, rascunho_de_gobr(g)[1].texto, d) for g, d in prompt.EXEMPLOS]


class AgenteDescricaoCliente:
    def __init__(self, llm: Any = None, *, max_correcoes: int = 1):
        self._llm = llm
        self.max_correcoes = max_correcoes
        self.exemplos = _exemplos()

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from src.llm_config import create_llm

            self._llm = create_llm()
        return self._llm

    def descrever(self, gobr: str) -> DescricaoCliente:
        _, rascunho = rascunho_de_gobr(gobr)
        mensagens = prompt.mensagens(gobr, rascunho.texto, self.exemplos)
        tentativas: list[str] = []
        problemas: list[str] = []
        for tentativa in range(self.max_correcoes + 1):
            texto = texto_da_resposta(self.llm.invoke(mensagens)).strip().strip('"').strip()
            tentativas.append(texto)
            problemas = conferir(gobr, texto)
            if not problemas:
                return DescricaoCliente(texto, "llm" if tentativa == 0 else "llm_corrigido", [], tentativas)
            mensagens = [*mensagens, ("ai", texto),
                         ("human", prompt.CORRECAO.format(problemas="\n".join(f"- {p}" for p in problemas)))]
        return DescricaoCliente(None, "falhou", problemas, tentativas)
