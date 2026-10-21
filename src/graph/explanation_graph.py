from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.gobr_interpreter import GobrInterpreterAgent
from src.agents.business_explainer import BusinessExplainerAgent
from src.models.rule_interpretation import RuleInterpretation
from src.models.business_explanation import BusinessExplanation


class ExplanationState(TypedDict):
    gobr: str
    interpretation: RuleInterpretation | None
    explanation: BusinessExplanation | None


interpreter_agent = GobrInterpreterAgent()
explainer_agent = BusinessExplainerAgent()


def interpret_gobr(
    state: ExplanationState
) -> ExplanationState:

    interpretation = interpreter_agent.interpret(
        state["gobr"]
    )

    return {
        **state,
        "interpretation": interpretation
    }


def explain_business_rule(
    state: ExplanationState
) -> ExplanationState:

    interpretation = state["interpretation"]

    if interpretation is None:
        raise ValueError(
            "A interpretação GOBR não foi gerada."
        )

    explanation = explainer_agent.explain(
        interpretation
    )

    return {
        **state,
        "explanation": explanation
    }


builder = StateGraph(ExplanationState)

builder.add_node(
    "interpretador_gobr",
    interpret_gobr
)

builder.add_node(
    "explicador_negocio",
    explain_business_rule
)

builder.add_edge(
    START,
    "interpretador_gobr"
)

builder.add_edge(
    "interpretador_gobr",
    "explicador_negocio"
)

builder.add_edge(
    "explicador_negocio",
    END
)

explanation_graph = builder.compile()