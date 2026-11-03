from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.agents.gobr_interpreter import GobrInterpreterAgent
from src.agents.business_explainer import BusinessExplainerAgent
from src.models.semantic_interpretation import SemanticInterpretation
from src.parsers.gobr_parser import GobrParser
from src.models.business_explanation import BusinessExplanation


class ExplanationState(TypedDict):
    gobr: str
    interpretation: SemanticInterpretation | None
    explanation: BusinessExplanation | None


interpreter_agent = GobrInterpreterAgent()
explainer_agent = BusinessExplainerAgent()
gobr_parser = GobrParser()


def interpret_gobr(
    state: ExplanationState
) -> ExplanationState:

    parsed_rule = gobr_parser.parse(state["gobr"])
    interpretation = interpreter_agent.interpret(parsed_rule)

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

    parsed_rule = gobr_parser.parse(state["gobr"])
    explanation = explainer_agent.explain(interpretation, parsed_rule=parsed_rule)

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
