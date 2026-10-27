"""Representação sintática de um bloco `parametros` GOBR."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ParsedScope(BaseModel):
    tipo: str
    # Mantido para compatibilidade com JSON/código da primeira entrega.
    valor: str
    operador: Optional[str] = None
    # A mesma cláusula pode citar vários times/carteiras/segmentações.
    valores: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sincronizar_valores(self):
        if not self.valores:
            self.valores = [self.valor]
        elif self.valores[0] != self.valor:
            raise ValueError("valor deve corresponder ao primeiro item de valores")
        return self


class ParsedCondition(BaseModel):
    texto: str
    operador: Optional[str] = None


class GobrParsedRule(BaseModel):
    nome: str
    escopos: list[ParsedScope] = Field(default_factory=list)
    condicoes: list[ParsedCondition] = Field(default_factory=list)
    frequencia: Optional[str] = None
    nao_agendar: bool = False
    modificadores: list[str] = Field(default_factory=list)
    desconhecidos: list[str] = Field(default_factory=list)
