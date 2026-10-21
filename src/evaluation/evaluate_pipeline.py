import json
from pathlib import Path

from src.parsers.gobr_parser import GobrParser
from src.agents.gobr_interpreter import GobrInterpreterAgent
from src.agents.business_explainer import BusinessExplainerAgent


VALIDATION_FILE = Path(
    "tests/fixtures/interpreter_validation_cases.json"
)


def normalize(value: str | None) -> str:
    if not value:
        return ""

    return (
        value
        .lower()
        .replace(".", "")
        .replace(",", "")
        .strip()
    )


def contains(text: str | None, value: str | None) -> bool:
    if not value:
        return True

    return normalize(value) in normalize(text)


def evaluate_case(
    parser: GobrParser,
    interpreter: GobrInterpreterAgent,
    explainer: BusinessExplainerAgent,
    case: dict
):
    gobr = case["gobr"]

    parsed = parser.parse(gobr)
    interpretation = interpreter.interpret(parsed)
    explanation = explainer.explain(interpretation)

    checks = {}

    # ---------------------------------------------------------
    # 1. Escopos devem continuar presentes no público semântico
    # ---------------------------------------------------------

    for index, scope in enumerate(parsed.escopos):
        checks[f"escopo_{index + 1}_interpretacao"] = contains(
            interpretation.publico,
            scope.valor
        )

    # ---------------------------------------------------------
    # 2. Escopos devem continuar presentes na explicação final
    # ---------------------------------------------------------

    for index, scope in enumerate(parsed.escopos):
        checks[f"escopo_{index + 1}_explicacao"] = contains(
            explanation.explicacao,
            scope.valor
        )

    # ---------------------------------------------------------
    # 3. Frequência deve ser preservada
    # ---------------------------------------------------------

    if parsed.frequencia:
        checks["frequencia_interpretacao"] = contains(
            interpretation.regra_atendimento,
            parsed.frequencia
        )

        checks["frequencia_explicacao"] = contains(
            explanation.explicacao,
            parsed.frequencia
        )

    # ---------------------------------------------------------
    # 4. Condições não podem desaparecer
    # ---------------------------------------------------------

    if parsed.condicoes:
        checks["quantidade_condicoes"] = (
            len(interpretation.criterios)
            == len(parsed.condicoes)
        )

    # ---------------------------------------------------------
    # 5. nao_agendar precisa virar restrição
    # ---------------------------------------------------------

    if parsed.nao_agendar:
        checks["nao_agendar_interpretacao"] = (
            len(interpretation.restricoes) > 0
        )

        checks["nao_agendar_explicacao"] = (
            "não" in normalize(explanation.explicacao)
            and "agend" in normalize(explanation.explicacao)
        )

    # ---------------------------------------------------------
    # 6. Modificadores não podem desaparecer
    # ---------------------------------------------------------

    if parsed.modificadores:
        checks["modificadores_preservados"] = (
            len(interpretation.modificadores)
            + len(interpretation.incertezas)
            >= len(parsed.modificadores)
        )

    # ---------------------------------------------------------
    # 7. Elementos desconhecidos precisam ser sinalizados
    # ---------------------------------------------------------

    if parsed.desconhecidos:
        checks["desconhecidos_sinalizados"] = (
            len(interpretation.incertezas) > 0
        )

    total = len(checks)
    passed = sum(checks.values())

    score = (
        (passed / total) * 100
        if total > 0
        else 100
    )

    return {
        "nome": case.get(
            "name",
            case.get("nome", "Regra")
        ),
        "score": score,
        "checks": checks,
        "parsed": parsed,
        "interpretation": interpretation,
        "explanation": explanation,
    }


def main():
    with open(
        VALIDATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        cases = json.load(file)

    parser = GobrParser()
    interpreter = GobrInterpreterAgent()
    explainer = BusinessExplainerAgent()

    scores = []

    print("\n=== AVALIAÇÃO DO PIPELINE ===\n")

    for case in cases:
        result = evaluate_case(
            parser,
            interpreter,
            explainer,
            case
        )

        scores.append(result["score"])

        print(
            f'{result["nome"]}: '
            f'{result["score"]:.1f}%'
        )

        for check_name, success in result["checks"].items():
            status = "OK" if success else "ERRO"

            print(
                f"  [{status}] {check_name}"
            )

        print(
            "  Explicação:",
            result["explanation"].explicacao
        )

        if result["score"] < 100:
            print(
                "  Parser:",
                result["parsed"].model_dump_json(
                    indent=2
                )
            )

            print(
                "  Interpretação:",
                result["interpretation"].model_dump_json(
                    indent=2
                )
            )

        print()

    overall = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    print("=" * 60)
    print(
        f"FIDELIDADE GERAL DO PIPELINE: "
        f"{overall:.2f}%"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()