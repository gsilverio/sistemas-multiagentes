from src.llm_config import create_llm

from src.models.business_explanation import BusinessExplanation
from src.models.semantic_interpretation import SemanticInterpretation
from src.prompts.business_explainer_prompt import BUSINESS_EXPLAINER_PROMPT


class BusinessExplainerAgent:
    def __init__(self):
        self.model = create_llm()

        self.chain = (
            BUSINESS_EXPLAINER_PROMPT
            | self.model
        )

    def explain(
        self,
        interpretation: SemanticInterpretation
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

        return BusinessExplanation(
            titulo="Regra de negócio",
            explicacao=response.content.strip(),
            publico=interpretation.publico,
            criterios=interpretation.criterios,
            comportamento=comportamento
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