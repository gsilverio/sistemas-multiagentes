from pydantic import BaseModel, Field


class SemanticInterpretation(BaseModel):
    publico: str | None = Field(
        default=None,
        description="Público comercial ao qual a regra se aplica"
    )

    criterios: list[str] = Field(
        default_factory=list,
        description="Critérios comerciais que os clientes devem atender"
    )

    regra_atendimento: str | None = Field(
        default=None,
        description="Comportamento de atendimento definido pela parametrização"
    )

    restricoes: list[str] = Field(
        default_factory=list,
        description="Restrições ou impedimentos definidos pela regra"
    )

    filtros_temporais: list[str] = Field(
        default_factory=list,
        description="Filtros que restringem quando a parametrizacao e aplicada"
    )

    modificadores: list[str] = Field(
        default_factory=list,
        description="Significado comercial dos modificadores encontrados"
    )

    incertezas: list[str] = Field(
        default_factory=list,
        description="Elementos cujo significado não pôde ser determinado com segurança"
    )