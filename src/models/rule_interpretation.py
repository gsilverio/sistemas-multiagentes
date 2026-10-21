from typing import Optional
from pydantic import BaseModel, Field


class RuleScope(BaseModel):
    tipo: Optional[str] = Field(
        default=None,
        description="Tipo da origem da regra"
    )

    valor: Optional[str] = Field(
        default=None,
        description="Valor da origem da regra"
    )

    operador: Optional[str] = Field(
        default=None,
        description="Operador lógico em relação ao escopo anterior: e, ou ou null"
    )


class RuleCondition(BaseModel):
    texto: str = Field(
        description="Texto da condição sem quando/e/ou"
    )

    operador: Optional[str] = Field(
        default=None,
        description="Operador lógico: e, ou ou null para a primeira condição"
    )


class RuleInterpretation(BaseModel):
    nome: str

    escopos: list[RuleScope] = Field(
        default_factory=list
    )

    condicoes: list[RuleCondition] = Field(
        default_factory=list
    )

    frequencia: Optional[str] = None

    nao_agendar: bool = False

    modificadores: list[str] = Field(
        default_factory=list,
        description=(
            "Cláusulas relevantes da parametrização que não pertencem "
            "a escopo, condição, frequência ou não agendar"
        )
    )

    observacoes: list[str] = Field(
        default_factory=list
    )