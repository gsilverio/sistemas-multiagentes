from pathlib import Path
import json

from src.parsers.gobr_parser import GobrParser
from src.agents.gobr_interpreter import GobrInterpreterAgent
from src.agents.business_explainer import BusinessExplainerAgent


OUTPUT_DIR = Path("outputs/first_delivery")


CASES = [
    {
        "id": "01_frequencia_simples",
        "gobr": """
parametros "Regra Padrao da Trenas - Customizavel: Frequencia de 7 dias para todos os clientes":
    da carteira("empresa toda")
    frequencia de 7 dias
""".strip(),
    },
    {
        "id": "02_clientes_sem_venda",
        "gobr": """
parametros "Regra Padrao da Empresa Customizavel: 14 dias para clientes sem venda":
    da carteira("empresa toda")
    quando sem venda
    frequencia de 14 dias
""".strip(),
    },
    {
        "id": "03_nao_agendar_inadimplentes",
        "gobr": """
parametros "Regra Padrao da Empresa Customizavel: Não agendar inadimplentes":
    da carteira("empresa toda")
    quando inadimplentes
    nao agendar
""".strip(),
    },
    {
        "id": "04_preferencia_dias_uteis",
        "gobr": """
parametros "Regra Padrao da Empresa Customizavel: Atendimento semanal":
    da carteira("empresa perola")
    frequencia semanal()
    preferencia nos dias uteis
""".strip(),
    },
    {
        "id": "05_nao_agendar_filiais",
        "gobr": """
parametros "Não agendar filiais":
    do time("televendas")
    quando é o cliente filial
    nao agendar
""".strip(),
    },
    {
        "id": "06_multiplos_escopos",
        "gobr": """
parametros "Clientes Prospects":
    da segmentacao("televendas ruptura de fornecedor")
    e da carteira("televendas consumidor")
    quando no ciclo de vida prospect com a empresa
    frequencia de 14 dias
""".strip(),
    },
]


def save_json(path: Path, data):
    with path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def save_text(path: Path, text: str):
    with path.open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    parser = GobrParser()
    interpreter = GobrInterpreterAgent()
    explainer = BusinessExplainerAgent()

    summary = []

    for case in CASES:
        case_id = case["id"]
        gobr = case["gobr"]

        print("=" * 70)
        print(f"PROCESSANDO: {case_id}")
        print("=" * 70)

        parsed = parser.parse(gobr)

        interpretation = interpreter.interpret(
            parsed
        )

        explanation = explainer.explain(
            interpretation,
            parsed_rule=parsed,
        )

        case_dir = OUTPUT_DIR / case_id

        case_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Entrada GOBR
        # -----------------------------------------------------

        save_text(
            case_dir / "input.gobr",
            gobr
        )

        # -----------------------------------------------------
        # Saída do Parser
        # -----------------------------------------------------

        save_json(
            case_dir / "parsed.json",
            parsed.model_dump()
        )

        # -----------------------------------------------------
        # Interpretação semântica
        # -----------------------------------------------------

        save_json(
            case_dir / "interpretation.json",
            interpretation.model_dump()
        )

        # -----------------------------------------------------
        # Explicação completa estruturada
        # -----------------------------------------------------

        save_json(
            case_dir / "explanation.json",
            explanation.model_dump()
        )

        # -----------------------------------------------------
        # Explicação final em texto
        # -----------------------------------------------------

        save_text(
            case_dir / "explanation.txt",
            explanation.explicacao
        )

        summary.append({
            "id": case_id,
            "nome": parsed.nome,
            "explicacao": explanation.explicacao,
            "revisao": explanation.revisao.model_dump() if explanation.revisao else None,
        })

        print(
            explanation.explicacao
        )

        print()

    # ---------------------------------------------------------
    # Resumo geral
    # ---------------------------------------------------------

    save_json(
        OUTPUT_DIR / "summary.json",
        summary
    )

    print("=" * 70)
    print(
        f"Outputs gerados em: "
        f"{OUTPUT_DIR.resolve()}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()