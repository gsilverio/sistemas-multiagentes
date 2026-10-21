from src.parsers.gobr_parser import GobrParser


def main():
    parser = GobrParser()

    gobr = """
parametros "Clientes Prospects":
    da segmentacao("televendas ruptura de fornecedor")
    e da carteira("televendas consumidor")
    quando no ciclo de vida prospect com a empresa
    frequencia de 14 dias
"""

    result = parser.parse(gobr)

    print(
        result.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()