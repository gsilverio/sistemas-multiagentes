from src.agents.gobr_interpreter import GobrInterpreterAgent
from src.agents.business_explainer import BusinessExplainerAgent


def main():
    interpreter = GobrInterpreterAgent()
    explainer = BusinessExplainerAgent()

    gobr = """
parametros "Ativos inadimplentes":
    da carteira("carteira televendas")
    quando ativos para o vendedor e inadimplentes
    frequencia de 14 dias
    """

    interpretation = interpreter.interpret(gobr)

    print("=== INTERPRETAÇÃO ===")
    print(
        interpretation.model_dump_json(
            indent=2
        )
    )

    explanation = explainer.explain(
        interpretation
    )

    print("\n=== EXPLICAÇÃO DE NEGÓCIO ===")
    print(
        explanation.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()