import json
from pathlib import Path

from src.parsers.gobr_parser import GobrParser


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIXTURE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "interpreter_validation_cases.json"
)


WEIGHTS = {
    "nome": 1,
    "escopos": 2,
    "condicoes": 2,
    "operadores": 2,
    "frequencia": 1,
    "nao_agendar": 1,
    "modificadores": 1
}


TOTAL_WEIGHT = sum(WEIGHTS.values())


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None

    return " ".join(
        value.strip().lower().split()
    )


def compare_nome(actual, expected):
    return (
        normalize_text(actual.nome)
        == normalize_text(expected["nome"])
    )


def compare_frequencia(actual, expected):
    return (
        normalize_text(actual.frequencia)
        == normalize_text(expected["frequencia"])
    )


def compare_nao_agendar(actual, expected):
    return (
        actual.nao_agendar
        == expected["nao_agendar"]
    )


def compare_escopos(actual, expected):
    expected_scopes = expected["escopos"]

    if len(actual.escopos) != len(expected_scopes):
        return False

    for actual_scope, expected_scope in zip(
        actual.escopos,
        expected_scopes
    ):
        if (
            normalize_text(actual_scope.tipo)
            != normalize_text(expected_scope["tipo"])
        ):
            return False

        if (
            normalize_text(actual_scope.valor)
            != normalize_text(expected_scope["valor"])
        ):
            return False

    return True


def compare_condicoes(actual, expected):
    expected_conditions = expected["condicoes"]

    if len(actual.condicoes) != len(expected_conditions):
        return False

    for actual_condition, expected_condition in zip(
        actual.condicoes,
        expected_conditions
    ):
        if (
            normalize_text(actual_condition.texto)
            != normalize_text(expected_condition["texto"])
        ):
            return False

    return True


def compare_operadores(actual, expected):
    expected_scopes = expected["escopos"]
    expected_conditions = expected["condicoes"]

    if len(actual.escopos) != len(expected_scopes):
        return False

    if len(actual.condicoes) != len(expected_conditions):
        return False

    for actual_scope, expected_scope in zip(
        actual.escopos,
        expected_scopes
    ):
        if (
            actual_scope.operador
            != expected_scope["operador"]
        ):
            return False

    for actual_condition, expected_condition in zip(
        actual.condicoes,
        expected_conditions
    ):
        if (
            actual_condition.operador
            != expected_condition["operador"]
        ):
            return False

    return True


def compare_modificadores(actual, expected):
    actual_values = [
        normalize_text(value)
        for value in actual.modificadores
    ]

    expected_values = [
        normalize_text(value)
        for value in expected.get(
            "modificadores",
            []
        )
    ]

    return actual_values == expected_values


def evaluate_case(parser, case):
    expected = case["expected"]

    actual = parser.parse(
        case["gobr"]
    )

    checks = {
        "nome": compare_nome(actual, expected),
        "escopos": compare_escopos(actual, expected),
        "condicoes": compare_condicoes(actual, expected),
        "operadores": compare_operadores(actual, expected),
        "frequencia": compare_frequencia(actual, expected),
        "nao_agendar": compare_nao_agendar(actual, expected),
        "modificadores": compare_modificadores(
            actual,
            expected
        )
    }

    score = sum(
        WEIGHTS[field]
        for field, correct in checks.items()
        if correct
    )

    accuracy = (
        score / TOTAL_WEIGHT
    ) * 100

    return actual, checks, score, accuracy


def print_case_result(
    case_id,
    actual,
    expected,
    checks,
    score,
    accuracy
):
    print(f"\n=== {case_id} ===")

    for field, correct in checks.items():
        status = "OK" if correct else "ERRO"

        print(
            f"{field:<15}: {status}"
        )

    print(
        f"Score: {score}/{TOTAL_WEIGHT}"
    )

    print(
        f"Acurácia: {accuracy:.2f}%"
    )

    failed_fields = [
        field
        for field, correct in checks.items()
        if not correct
    ]

    if failed_fields:
        print("\n--- DETALHES DOS ERROS ---")

        actual_dict = actual.model_dump()

        for field in failed_fields:
            print(f"\n[{field}]")

            if field == "operadores":
                print(
                    "Esperado escopos:",
                    [
                        item.get("operador")
                        for item in expected.get(
                            "escopos",
                            []
                        )
                    ]
                )

                print(
                    "Obtido escopos:",
                    [
                        item.operador
                        for item in actual.escopos
                    ]
                )

                print(
                    "Esperado condições:",
                    [
                        item.get("operador")
                        for item in expected.get(
                            "condicoes",
                            []
                        )
                    ]
                )

                print(
                    "Obtido condições:",
                    [
                        item.operador
                        for item in actual.condicoes
                    ]
                )

                continue

            print(
                "Esperado:",
                expected.get(field)
            )

            print(
                "Obtido:",
                actual_dict.get(field)
            )


def main():
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de casos não encontrado: {FIXTURE_PATH}"
        )

    with FIXTURE_PATH.open(
        encoding="utf-8"
    ) as file:
        cases = json.load(file)

    parser = GobrParser()

    total_score = 0

    max_score = (
        len(cases)
        * TOTAL_WEIGHT
    )

    field_hits = {
        field: 0
        for field in WEIGHTS
    }

    for case in cases:
        actual, checks, score, accuracy = evaluate_case(
            parser,
            case
        )

        total_score += score

        for field, correct in checks.items():
            if correct:
                field_hits[field] += 1

        print_case_result(
            case["id"],
            actual,
            case["expected"],
            checks,
            score,
            accuracy
        )

    global_accuracy = (
        total_score / max_score
    ) * 100

    print("\n==============================")
    print("RESULTADO GERAL DO PARSER")
    print("==============================")

    print(
        f"Score total: "
        f"{total_score}/{max_score}"
    )

    print(
        f"Acurácia geral: "
        f"{global_accuracy:.2f}%"
    )

    print("\nAcurácia por campo:")

    for field, hits in field_hits.items():
        field_accuracy = (
            hits / len(cases)
        ) * 100

        print(
            f"{field:<15}: "
            f"{field_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()