from langchain_ollama import ChatOllama

from src.models.rule_interpretation import RuleInterpretation
from src.prompts.gobr_interpreter_prompt import GOBR_INTERPRETER_PROMPT


class GobrInterpreterAgent:
    def __init__(self):
        self.model = ChatOllama(
            model="gemma3:4b",
            temperature=0
        )

        self.structured_model = self.model.with_structured_output(
            RuleInterpretation,
            method="json_schema"
        )

        self.chain = (
            GOBR_INTERPRETER_PROMPT
            | self.structured_model
        )

    def interpret(self, gobr: str) -> RuleInterpretation:
        result = self.chain.invoke({
            "gobr": gobr
        })

        if result.condicoes:
            result.condicoes[0].operador = None

        return result