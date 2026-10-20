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

    escopo: Optional[RuleScope] = None

    condicoes: list[RuleCondition] = Field(
        default_factory=list
    )

    frequencia: Optional[str] = None

    nao_agendar: bool = False

    observacoes: list[str] = Field(
        default_factory=list
    )