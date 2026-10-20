from pydantic import BaseModel, Field


class BusinessExplanation(BaseModel):
    titulo: str = Field(
        description="Título da regra, preservando o nome recebido"
    )

    explicacao: str = Field(
        description=(
            "Explicação completa da regra, contendo público, critérios "
            "e comportamento sem inventar significados"
        )
    )

    publico: str | None = Field(
        default=None,
        description=(
            "Descrição dos clientes afetados pela regra. "
            "Deve ser preenchido sempre que existir escopo."
        )
    )

    criterios: list[str] = Field(
        default_factory=list,
        description=(
            "Lista dos critérios da regra. Cada condição recebida deve "
            "originar um item desta lista."
        )
    )

    comportamento: str = Field(
        description=(
            "Comportamento definido pela regra, como frequência de atendimento "
            "ou exclusão do agendamento."
        )
    )