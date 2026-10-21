from src.graph.explanation_graph import explanation_graph


def main():
    gobr = """
parametros "Clientes Prospects":
    da segmentacao("televendas ruptura de fornecedor")
    e da carteira("televendas consumidor")
    quando no ciclo de vida prospect com a empresa
    frequencia de 14 dias
"""

    result = explanation_graph.invoke({
        "gobr": gobr,
        "interpretation": None,
        "explanation": None
    })

    print("=== INTERPRETAÇÃO ===")

    interpretation = result["interpretation"]

    if interpretation:
        print(
            interpretation.model_dump_json(
                indent=2
            )
        )

    print("\n=== EXPLICAÇÃO ===")

    explanation = result["explanation"]

    if explanation:
        print(
            explanation.model_dump_json(
                indent=2
            )
        )


if __name__ == "__main__":
    main()