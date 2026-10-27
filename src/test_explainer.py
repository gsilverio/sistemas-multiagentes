from src.parsers.gobr_parser import GobrParser
from src.agents.gobr_interpreter import GobrInterpreterAgent
from src.agents.business_explainer import BusinessExplainerAgent


def main():
    parser = GobrParser()
    interpreter = GobrInterpreterAgent()
    explainer = BusinessExplainerAgent()

    gobr = """
parametros "Clientes Prospects":
    da segmentacao("televendas ruptura de fornecedor")
    e da carteira("televendas consumidor")
    quando no ciclo de vida prospect com a empresa
    frequencia de 14 dias
"""

    parsed = parser.parse(gobr)

    interpretation = interpreter.interpret(
        parsed
    )

    explanation = explainer.explain(
        interpretation,
        parsed_rule=parsed,
    )

    print("=== PARSER ===")
    print(parsed.model_dump_json(indent=2))

    print("\n=== INTERPRETAÇÃO ===")
    print(
        interpretation.model_dump_json(
            indent=2
        )
    )

    print("\n=== EXPLICAÇÃO ===")
    print(
        explanation.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()