"""AST de um bloco `parametros` → fatos + texto explicativo determinístico.

O rascunho é a "verdade" da explicação: cada fato liga um trecho do GOBR ao
seu significado (glossário) e às chaves que o guard confere. O LLM só
reescreve esse conteúdo de forma natural; se errar, o próprio rascunho é
usado como explicação.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.explicacao import glossario
from src.gobr.ast import GobrParsedRule
from src.gobr.parser import GobrParser
from src.gobr.scope_text import scope_subject


class Fato(BaseModel):
    tipo: str            # escopo | condicao | frequencia | nao_agendar | modificador | filtro_temporal | desconhecido
    gobr: str            # trecho original
    significado: str     # tradução do glossário
    conhecido: bool = True
    chaves: list[str] = Field(default_factory=list)


class Rascunho(BaseModel):
    nome: str
    fatos: list[Fato]
    texto: str
    nomes_citados: list[str]   # nomes entre aspas que precisam aparecer literalmente
    conectores_ou: int = 0     # quantos `ou` lógicos o texto precisa manter
    nao_agendar: bool = False

    @property
    def desconhecidos(self) -> list[Fato]:
        return [f for f in self.fatos if not f.conhecido]


def _fato(tipo: str, gobr: str, traducao: glossario.Traducao) -> Fato:
    return Fato(tipo=tipo, gobr=gobr.strip(), significado=traducao.frase,
                conhecido=traducao.conhecida, chaves=list(traducao.chaves))


def montar_rascunho(regra: GobrParsedRule) -> Rascunho:
    fatos: list[Fato] = []
    nomes: list[str] = []
    ou = 0

    for escopo in regra.escopos:
        valores = escopo.valores or [escopo.valor]
        nomes.extend(valores)
        if escopo.operador == "ou":
            ou += 1
    sujeito = scope_subject(regra.escopos)
    if regra.escopos:
        fatos.append(Fato(tipo="escopo", gobr=", ".join(f'{e.tipo}({", ".join(e.valores)})' for e in regra.escopos),
                          significado=sujeito))

    partes_condicao = []
    for indice, condicao in enumerate(regra.condicoes):
        traducao = glossario.traduzir_condicao(condicao.texto)
        fatos.append(_fato("condicao", condicao.texto, traducao))
        nomes.extend(glossario.valores_entre_aspas(condicao.texto))
        ou += glossario.conectores_ou(condicao.texto)
        operador = condicao.operador or "e"
        if indice and operador == "ou":
            ou += 1
        partes_condicao.append(traducao.frase if indice == 0 else f"{operador} {traducao.frase}")

    if regra.nao_agendar:
        acao = "não devem ser agendados para atendimento"
        fatos.append(Fato(tipo="nao_agendar", gobr="nao agendar", significado=acao))
    elif regra.frequencia:
        traducao = glossario.traduzir_frequencia(regra.frequencia)
        fatos.append(_fato("frequencia", f"frequencia {regra.frequencia}", traducao))
        acao = f"devem ser atendidos {traducao.frase}"
    else:
        acao = "recebem a parametrização abaixo"

    frase = sujeito
    if partes_condicao:
        frase += " " + " ".join(partes_condicao)
    frases = [f"{frase} {acao}."]

    if regra.filtros_temporais:
        traducoes = [glossario.traduzir_filtro(f) for f in regra.filtros_temporais]
        for filtro, traducao in zip(regra.filtros_temporais, traducoes):
            fatos.append(_fato("filtro_temporal", filtro, traducao))
        frases.append("Esta parametrização só é aplicada quando a regra executa "
                      + " e ".join(t.frase for t in traducoes) + ".")

    for modificador in regra.modificadores:
        traducao = glossario.traduzir_modificador(modificador)
        fatos.append(_fato("modificador", modificador, traducao))
        nomes.extend(glossario.valores_entre_aspas(modificador))
        frases.append(traducao.frase)

    for linha in regra.desconhecidos:
        fatos.append(Fato(tipo="desconhecido", gobr=linha, significado=f'Cláusula não interpretada: "{linha}".',
                          conhecido=False, chaves=[glossario._literal(linha)]))
        frases.append(f'Cláusula não interpretada: "{linha}".')

    texto = " ".join(frases)
    texto = texto[0].upper() + texto[1:]
    return Rascunho(nome=regra.nome, fatos=fatos, texto=texto,
                    nomes_citados=list(dict.fromkeys(n for n in nomes if n.strip())),
                    conectores_ou=ou, nao_agendar=regra.nao_agendar)


def rascunho_de_gobr(gobr: str) -> tuple[GobrParsedRule, Rascunho]:
    regra = GobrParser().parse(gobr)
    return regra, montar_rascunho(regra)
