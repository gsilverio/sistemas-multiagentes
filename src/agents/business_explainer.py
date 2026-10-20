from langchain_ollama import ChatOllama

from src.models.business_explanation import BusinessExplanation
from src.models.rule_interpretation import RuleInterpretation
from src.prompts.business_explainer_prompt import BUSINESS_EXPLAINER_PROMPT


class BusinessExplainerAgent:
    def __init__(self):
        self.model = ChatOllama(
            model="gemma3:4b",
            temperature=0
        )

        self.chain = BUSINESS_EXPLAINER_PROMPT | self.model

    def explain(
        self,
        interpretation: RuleInterpretation
    ) -> BusinessExplanation:

        publico = self._build_publico(interpretation)
        criterios = self._build_criterios(interpretation)
        comportamento = self._build_comportamento(interpretation)

        criterios_formatados = self._format_criterios(
            interpretation
        )

        response = self.chain.invoke({
            "nome": interpretation.nome,
            "publico": publico or "Não informado",
            "criterios": criterios_formatados,
            "comportamento": comportamento or "Não informado"
        })

        return BusinessExplanation(
            titulo=interpretation.nome,
            explicacao=response.content.strip(),
            publico=publico,
            criterios=criterios,
            comportamento=comportamento
        )

    def _build_publico(
        self,
        interpretation: RuleInterpretation
    ) -> str | None:

        if interpretation.escopo is None:
            return None

        tipo = interpretation.escopo.tipo
        valor = interpretation.escopo.valor

        if not tipo or not valor:
            return None

        if tipo == "time":
            return f"Clientes do time {valor}"

        if tipo == "carteira":
            return f"Clientes da carteira {valor}"

        if tipo == "segmentacao":
            return f"Clientes da segmentação {valor}"

        return f"Clientes de {tipo} {valor}"

    def _build_criterios(
        self,
        interpretation: RuleInterpretation
    ) -> list[str]:

        return [
            condition.texto
            for condition in interpretation.condicoes
        ]

    def _build_comportamento(
        self,
        interpretation: RuleInterpretation
    ) -> str | None:

        if interpretation.nao_agendar:
            return (
                "Não incluir esses clientes "
                "no agendamento de atendimento."
            )

        if interpretation.frequencia:
            return (
                "Realizar o atendimento com frequência de "
                f"{interpretation.frequencia}."
            )

        return None

    def _format_criterios(
        self,
        interpretation: RuleInterpretation
    ) -> str:

        if not interpretation.condicoes:
            return "Nenhum critério informado."

        partes = []

        for index, condition in enumerate(
            interpretation.condicoes
        ):
            if index == 0:
                partes.append(condition.texto)
                continue

            if condition.operador:
                partes.append(
                    f"{condition.operador} {condition.texto}"
                )
            else:
                partes.append(condition.texto)

        return " ".join(partes)