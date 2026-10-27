from src.llm_config import create_llm

from src.models.business_explanation import BusinessExplanation, ExplanationReview
from src.models.gobr_parsed_rule import GobrParsedRule
from src.validation.explanation_guard import ExplanationGuard
from src.models.semantic_interpretation import SemanticInterpretation
from src.prompts.business_explainer_prompt import BUSINESS_EXPLAINER_PROMPT


class BusinessExplainerAgent:
    def __init__(self):
        self.model = create_llm()
        self.guard = ExplanationGuard()

        self.chain = (
            BUSINESS_EXPLAINER_PROMPT
            | self.model
        )

    def explain(
        self,
        interpretation: SemanticInterpretation,
        parsed_rule: GobrParsedRule | None = None,
    ) -> BusinessExplanation:

        criterios_formatados = self._format_list(
            interpretation.criterios
        )

        restricoes_formatadas = self._format_list(
            interpretation.restricoes
        )

        modificadores_formatados = self._format_list(
            interpretation.modificadores
        )

        incertezas_formatadas = self._format_list(
            interpretation.incertezas
        )

        response = self.chain.invoke({
            "publico": (
                interpretation.publico
                or "Não informado"
            ),
            "criterios": criterios_formatados,
            "regra_atendimento": (
                interpretation.regra_atendimento
                or "Não informado"
            ),
            "restricoes": restricoes_formatadas,
            "modificadores": modificadores_formatados,
            "incertezas": incertezas_formatadas
        })

        comportamento = (
            interpretation.regra_atendimento
        )

        if interpretation.restricoes:
            restricoes_texto = " ".join(
                interpretation.restricoes
            )

            if comportamento:
                comportamento = (
                    f"{comportamento} "
                    f"{restricoes_texto}"
                )
            else:
                comportamento = restricoes_texto

        draft = response.content.strip()
        if parsed_rule is None:
            # Compatibilidade com chamadas antigas; sem GOBR original,
            # a explicação não está pronta para ser revisada/indexada.
            final_text = draft
            review = ExplanationReview(
                status="needs_review",
                metodo="llm",
                problemas=["A validação não recebeu a regra GOBR original."],
            )
        else:
            check = self.guard.inspect(parsed_rule, draft)
            if check.accepted:
                final_text = draft
                review = ExplanationReview(
                    status="pending_review",
                    metodo="llm",
                )
            else:
                final_text = self.guard.conservative_fallback(parsed_rule)
                review = ExplanationReview(
                    status="needs_review",
                    metodo="deterministic_fallback",
                    problemas=check.issues,
                    rascunho_llm=draft,
                )

        return BusinessExplanation(
            titulo="Regra de negócio",
            explicacao=final_text,
            publico=interpretation.publico,
            criterios=interpretation.criterios,
            comportamento=comportamento,
            revisao=review,
        )

    def _format_list(
        self,
        values: list[str]
    ) -> str:

        if not values:
            return "Nenhum."

        return "\n".join(
            f"- {value}"
            for value in values
        )