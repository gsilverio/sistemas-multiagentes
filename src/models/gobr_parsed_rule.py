from typing import Optional
from pydantic import BaseModel, Field


class ParsedScope(BaseModel):
    tipo: str
    valor: str
    operador: Optional[str] = None


class ParsedCondition(BaseModel):
    texto: str
    operador: Optional[str] = None


class GobrParsedRule(BaseModel):
    nome: str

    escopos: list[ParsedScope] = Field(
        default_factory=list
    )

    condicoes: list[ParsedCondition] = Field(
        default_factory=list
    )

    frequencia: Optional[str] = None

    nao_agendar: bool = False

    modificadores: list[str] = Field(
        default_factory=list
    )

    desconhecidos: list[str] = Field(
        default_factory=list
    )