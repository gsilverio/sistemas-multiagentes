"""Contratos (entradas e saídas tipadas) trocados entre os agentes do grafo.

O `RuleSpec` é a representação intermediária: o que o cliente quer, em
termos comerciais, ainda sem GOBR. A Percepção o produz, o Analista o
critica e ajusta, o Gerador o transforma em blocos `parametros`.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Faixa(BaseModel):
    """Intervalo de dias desde a última compra (ou visita) que define o segmento.

    "compraram nos últimos 90 dias mas estão há 60 sem comprar" → min 60, max 90.
    "sem pedidos há mais de 90 dias" → min 90, max None.
    """

    medida: Literal["dias_sem_compra", "dias_sem_visita"] = "dias_sem_compra"
    min_dias: Optional[int] = None
    max_dias: Optional[int] = None

    def como_texto(self) -> str:
        alvo = "compra" if self.medida == "dias_sem_compra" else "visita"
        if self.min_dias and self.max_dias:
            return f"entre {self.min_dias} e {self.max_dias} dias sem {alvo}"
        if self.max_dias:
            return f"até {self.max_dias} dias sem {alvo}"
        if self.min_dias:
            return f"mais de {self.min_dias} dias sem {alvo}"
        return f"qualquer tempo sem {alvo}"

    def como_descricao(self) -> str:
        """A faixa com as palavras das descrições do acervo, que o Gerador conhece.

        Não é `como_texto`: no acervo, "sem compra entre 20 e 40 dias" é
        `fora da recencia entre 20 e 40`, o contrário do grupo pedido.
        """
        alvo = "compra" if self.medida == "dias_sem_compra" else "visita"
        if self.min_dias and self.max_dias:
            return f"última {alvo} entre {self.min_dias} e {self.max_dias} dias atrás"
        if self.max_dias:
            return f"{alvo} nos últimos {self.max_dias} dias"
        return f"última {alvo} há mais de {self.min_dias} dias" if alvo == "visita" \
            else f"sem compra há mais de {self.min_dias} dias"


# Palavras de um critério que só fala da faixa de dias ("compraram nos últimos 90 dias mas estão há 60 sem comprar").
_SO_FAIXA = re.compile(
    r"(?:\d+|clientes?|com|cuja|que|fizeram|compr\w*|[uú]ltim[ao]s?|pedidos?|entre|n[oa]s?|dias?|d|h[aá]|a|[aà]|mais|menos|"
    r"de|do|da|sem|mas|est[aã]o?|atr[aá]s|acima|at[eé]|em|e|o|os|tiveram|foi|faz|tem|t[eê]m|num|per[ií]odo)")


def so_fala_da_faixa(criterio: str) -> bool:
    palavras = re.findall(r"\w+", criterio.lower())
    return bool(re.search(r"\d", criterio)) and all(_SO_FAIXA.fullmatch(p) for p in palavras)


class Segmento(BaseModel):
    nome: str
    criterio: str = ""                          # como o cliente descreveu o grupo
    faixa: Optional[Faixa] = None
    frequencia_dias: Optional[int] = None       # frequência fixa em dias
    frequencia_texto: Optional[str] = None      # quando não é um número fixo ("a do vendedor")
    nao_agendar: bool = False
    restricoes: list[str] = Field(default_factory=list)  # "somente clientes fora da rota da semana"
    fallback: bool = False                      # "clientes que não se enquadram nas demais"

    def descricao(self) -> str:
        """Texto que o Gerador recebe (mesmo formato dos pares de fine-tuning)."""
        if self.faixa and (self.faixa.min_dias or self.faixa.max_dias) and so_fala_da_faixa(self.criterio):
            partes = [("Clientes " if self.faixa.como_descricao().startswith("sem ") else "Clientes com ")
                      + self.faixa.como_descricao()]
        else:
            partes = [self.criterio.strip() or ("Clientes que não se enquadraram nas regras anteriores"
                                                if self.fallback else "Todos os clientes")]
            if self.faixa and (self.faixa.min_dias or self.faixa.max_dias):
                partes.append(f"({self.faixa.como_descricao()})")
        texto = " ".join(partes).rstrip(".") + "."
        if self.nao_agendar:
            texto += " Não devem ser agendados."
        elif self.frequencia_dias:
            texto += f" Devem ser atendidos a cada {self.frequencia_dias} dias."
        elif self.frequencia_texto:
            texto += f" Frequência: {self.frequencia_texto}."
        for restricao in self.restricoes:
            texto += f" {restricao.strip().rstrip('.')}."
        return texto


class RuleSpec(BaseModel):
    cliente: Optional[str] = None
    time: str = "televendas"
    vendedores: Literal["todostelevendas", "todosdecampo", "todos"] = "todostelevendas"
    segmentos: list[Segmento] = Field(default_factory=list)
    restricoes_gerais: list[str] = Field(default_factory=list)  # o que não virou segmento
    observacoes: list[str] = Field(default_factory=list)


Severidade = Literal["erro", "aviso", "info"]


class Achado(BaseModel):
    """Falha ou ponto de atenção na modelagem comercial."""

    codigo: str
    severidade: Severidade
    mensagem: str
    segmento: Optional[str] = None


class Pergunta(BaseModel):
    id: str
    texto: str
    bloqueante: bool = False          # sem resposta não dá para gerar
    suposicao: Optional[str] = None   # o que será assumido se não houver resposta


class Analise(BaseModel):
    achados: list[Achado] = Field(default_factory=list)
    perguntas: list[Pergunta] = Field(default_factory=list)
    ajustes: list[str] = Field(default_factory=list)   # correções automáticas aplicadas
    spec: RuleSpec                                     # spec ajustada (ex.: reordenada)

    @property
    def bloqueantes(self) -> list[Pergunta]:
        return [p for p in self.perguntas if p.bloqueante]


class TrechoContexto(BaseModel):
    titulo: str
    texto: str


class ContextoSegmento(BaseModel):
    exemplos: list[TrechoContexto] = Field(default_factory=list)   # descrição → GOBR do acervo
    manual: list[TrechoContexto] = Field(default_factory=list)


class Conferencia(BaseModel):
    """Resultado do Validador: sintaxe (GobrValidator) + conferência com o RuleSpec."""

    ok: bool
    erros_por_bloco: dict[str, list[str]] = Field(default_factory=dict)
    erros_gerais: list[str] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
