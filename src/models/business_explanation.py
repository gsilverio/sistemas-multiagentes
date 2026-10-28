from typing import Literal

from pydantic import BaseModel, Field


class ExplanationReview(BaseModel):
    # Nada é aprovado automaticamente para entrar no RAG.
    status: Literal["pending_review", "needs_review"] = "pending_review"
    metodo: Literal["llm", "deterministic_fallback"] = "llm"
    problemas: list[str] = Field(default_factory=list)
    rascunho_llm: str | None = None


class BusinessExplanation(BaseModel):
    titulo: str
    explicacao: str
    publico: str | None = None
    criterios: list[str] = Field(default_factory=list)
    comportamento: str | None = None
    filtros_temporais: list[str] = Field(default_factory=list)
    revisao: ExplanationReview | None = None
