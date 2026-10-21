from pydantic import BaseModel, Field


class BusinessExplanation(BaseModel):
    titulo: str
    explicacao: str
    publico: str | None = None
    criterios: list[str] = Field(default_factory=list)
    comportamento: str | None = None