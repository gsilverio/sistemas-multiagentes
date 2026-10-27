"""Apresentação textual conservadora dos escopos da AST GOBR.

Uma lista de nomes dentro de uma cláusula não recebe um operador lógico
sintético. `e`/`ou` só são reproduzidos entre cláusulas diferentes.
"""

from src.models.gobr_parsed_rule import ParsedScope


def _nomes(valores: list[str]) -> str:
    citados = [f'"{valor}"' for valor in valores]
    if len(citados) == 1:
        return citados[0]
    if len(citados) == 2:
        return f'{citados[0]}, {citados[1]}'
    return ", ".join(citados)


def scope_fragment(scope: ParsedScope) -> str:
    valores = scope.valores or [scope.valor]
    if scope.tipo == "time":
        if len(valores) == 1:
            if valores[0].casefold() == "todos":
                return "todos os clientes"
            if valores[0].casefold() in {"toda empresa", "empresa toda"}:
                return "todos os clientes da empresa"
            return f'do time {_nomes(valores)}'
        return f'dos times {_nomes(valores)}'
    if scope.tipo == "carteira":
        return f'{"da carteira" if len(valores) == 1 else "das carteiras"} {_nomes(valores)}'
    if scope.tipo == "segmentacao":
        return f'{"da segmentação" if len(valores) == 1 else "das segmentações"} {_nomes(valores)}'
    return f'de {scope.tipo} {_nomes(valores)}'


def scope_subject(scopes: list[ParsedScope]) -> str:
    if not scopes:
        return "Os clientes"
    pieces = []
    for index, scope in enumerate(scopes):
        fragment = scope_fragment(scope)
        if index == 0:
            if fragment.startswith("todos os clientes"):
                pieces.append(fragment.capitalize())
            else:
                pieces.append(f"Os clientes {fragment}")
        else:
            if scope.operador in ("e", "ou"):
                pieces.append(f"{scope.operador} {fragment}")
            else:
                pieces.append(f"{fragment}")
    return " ".join(pieces)
