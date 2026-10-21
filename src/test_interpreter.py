from src.parsers.gobr_parser import GobrParser
from src.agents.gobr_interpreter import GobrInterpreterAgent


def main():
    parser = GobrParser()
    interpreter = GobrInterpreterAgent()

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

    print("=== PARSER ===")
    print(
        parsed.model_dump_json(
            indent=2
        )
    )

    print("\n=== INTERPRETAÇÃO SEMÂNTICA ===")
    print(
        interpretation.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()