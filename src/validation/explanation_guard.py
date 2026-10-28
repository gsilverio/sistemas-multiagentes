from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.models.gobr_parsed_rule import GobrParsedRule, ParsedCondition, ParsedScope
from src.models.scope_text import scope_subject


def _normalize(value: str) -> str:
    """Normalização apenas para conferência de texto, sem alterar nomes na saída."""
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip()


def _condition_anchor(text: str) -> str:
    """Recorta conectivos iniciais; conserva os termos do domínio."""
    normalized = _normalize(text)
    return re.sub(r"^(?:e o |e a |e |o |a |no |na |em |ser |estar )", "", normalized)


@dataclass
class GuardResult:
    accepted: bool
    issues: list[str] = field(default_factory=list)


class ExplanationGuard:
    """Checagens estruturais conservadoras, não são métrica de acurácia.

    Não prova correção semântica. Se detectar desvio, a explicação do LLM
    não será usada como texto principal da saída.
    """

    def inspect(self, parsed: GobrParsedRule, explanation: str) -> GuardResult:
        text = _normalize(explanation)
        issues: list[str] = []
        if not text:
            return GuardResult(False, ["O agente retornou texto vazio."])

        # Conferir TODOS os valores de cada chamada de escopo, na ordem
        # informada. `e` e `ou` referem-se somente a cláusulas distintas.
        positions = []
        offset = 0
        for scope in parsed.escopos:
            scope_positions = []
            for value in scope.valores or [scope.valor]:
                target = _normalize(value)
                # Fronteiras impedem que "campo" passe por "campo_2".
                found = re.search(
                    rf'(?<!\w){re.escape(target)}(?!\w)', text[offset:]
                )
                position = offset + found.start() if found else -1
                if position < 0:
                    issues.append(
                        f'Escopo não preservado literalmente: {scope.tipo}("{value}").'
                    )
                else:
                    offset = position + len(target)
                scope_positions.append((position, position + len(target)))
            positions.append(scope_positions)

        for i, scope in enumerate(parsed.escopos[1:], start=1):
            prev_positions = positions[i - 1]
            next_positions = positions[i]
            prev_end = prev_positions[-1][1]
            next_start = next_positions[0][0]
            if prev_positions[-1][0] < 0 or next_start < 0:
                continue
            middle = text[prev_end:next_start]
            if scope.operador in ("e", "ou"):
                operator = scope.operador
                if not re.search(rf"\b{re.escape(operator)}\b", middle):
                    issues.append(f"Operador entre escopos não preservado: {operator}.")
                elif operator == "e" and re.search(r"\be\s+(?:os\s+)?clientes\b", middle):
                    issues.append("A conjunção foi apresentada como públicos separados.")

        for condition in parsed.condicoes:
            normalized_condition = _normalize(condition.texto)
            # Mantém a correção aplicada anteriormente no Windows: "filiais"
            # é uma formulação aceitável de "é o cliente filial".
            if normalized_condition in ("e o cliente filial", "cliente filial"):
                preserved = bool(re.search(
                    r"\b(?:cliente(?:s)?\s+filia(?:l|is)|filiais)\b", text
                ))
            else:
                anchor = _condition_anchor(condition.texto)
                preserved = not anchor or anchor in text
            if not preserved:
                issues.append(f'Condição não preservada: "{condition.texto}".')

        if parsed.frequencia:
            frequency = _normalize(parsed.frequencia)
            if frequency.startswith("semanal"):
                if not re.search(r"\bsemanal(?:mente)?\b", text):
                    issues.append("Frequência semanal não preservada.")
            elif frequency not in text:
                issues.append(f'Frequência não preservada: "{parsed.frequencia}".')

        if parsed.nao_agendar:
            negative_action = (
                r"\bnao\s+(?:(?:devem|deve|podem|pode|serao|sera)\s+)?(?:ser\s+)?agendad\w*\b"
                r"|\bnao\s+(?:devem|deve|podem|pode)\s+ser\s+incluid\w*\s+no\s+agendamento\b"
                r"|\bnao\s+agendar\b"
                r"|\bproibid\w*\s+agendar\b"
                r"|\bnao\s+e\s+permitido\s+agendar\b"
            )
            if not re.search(negative_action, text):
                issues.append('A proibição de "nao agendar" não foi expressa como ação.')
            # Não confundir o impedimento de criar um agendamento com
            # uma regra sobre agendamentos que já existem.
            if re.search(r"\b(?:ja\s+\w+\s+)?(?:ja\s+)?(?:possuem|possui|tinham|tem|tenham)\s+(?:um\s+)?agendamento\b", text):
                issues.append("A explicação introduz agendamentos já existentes.")
            if re.search(r"\bnao\s+(?:pode|podem|deve|devem)\s+estar\s+agendad\w*", text):
                issues.append("A proibição foi transformada em estado do cliente.")

        # O dia na preferencia de visita NAO basta para comprovar
        # que um filtro de execucao foi explicado.
        sentences = re.split(r"[.!?;\n]+", text)
        for filtro in parsed.filtros_temporais:
            normalized = _normalize(filtro)
            presente_como_filtro = any(
                re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", sentence)
                and re.search(
                    r"\b(?:aplic\w*|execut\w*|somente|apenas|restrit\w*|filtro\w*)\b",
                    sentence,
                )
                and not re.search(
                    r"\b(?:nao|nunca)\s+(?:(?:deve\s+)?(?:ser\s+)?|se\s+)(?:aplic\w*|execut\w*)\b",
                    sentence,
                )
                for sentence in sentences
            )
            if not presente_como_filtro:
                issues.append(f'Filtro temporal de execucao nao preservado: "{filtro}".')

        for modifier in parsed.modificadores:
            normalized = _normalize(modifier)
            if "preferencia" in normalized and "dias uteis" in normalized:
                if "dias uteis" not in text or not re.search(r"\bprefer", text):
                    issues.append(f'Modificador não preservado: "{modifier}".')
            elif normalized not in text:
                issues.append(f'Modificador não preservado literalmente: "{modifier}".')

        if parsed.desconhecidos:
            issues.append("A regra contém cláusulas que o parser ainda não reconhece.")

        return GuardResult(accepted=not issues, issues=issues)

    def conservative_fallback(self, parsed: GobrParsedRule) -> str:
        """Texto rastreável derivado da AST. Não interpreta cláusulas desconhecidas."""
        subject = self._subject(parsed.escopos)
        conditions = self._conditions(parsed.condicoes)
        prefix = f"{subject}{conditions}"

        clauses: list[str] = []
        if parsed.nao_agendar:
            clauses.append("não devem ser agendados para atendimento")
        if parsed.frequencia:
            frequency = parsed.frequencia.strip()
            if _normalize(frequency) == "semanal()":
                clauses.append("devem receber atendimento semanalmente")
            elif _normalize(frequency).startswith("de "):
                clauses.append(f"devem receber atendimento {frequency}")
            else:
                clauses.append(f"devem receber atendimento a cada {frequency}")

        result = prefix
        if clauses:
            result += " " + "; ".join(clauses)
        result += "."

        for filtro in parsed.filtros_temporais:
            result += f" A regra deve ser aplicada somente {filtro}."

        for modifier in parsed.modificadores:
            if "preferencia" in _normalize(modifier) and "dias uteis" in _normalize(modifier):
                result += " Há preferência por dias úteis."
            else:
                result += f' Modificador informado: "{modifier}".'
        if parsed.desconhecidos:
            result += " Cláusulas ainda não interpretadas: "
            result += "; ".join(f'"{item}"' for item in parsed.desconhecidos) + "."
        return result

    @staticmethod
    def _subject(scopes: list[ParsedScope]) -> str:
        return scope_subject(scopes)

    @staticmethod
    def _conditions(conditions: list[ParsedCondition]) -> str:
        if not conditions:
            return ""
        chunks = []
        for i, condition in enumerate(conditions):
            text = _normalize(condition.texto)
            if text == "sem venda":
                clause = "que estiverem sem venda"
            elif text == "inadimplentes":
                clause = "que forem inadimplentes"
            elif text == "e o cliente filial" or text == "é o cliente filial":
                clause = 'que forem classificados como "cliente filial"'
            elif text.startswith("no ciclo de vida "):
                clause = "que estiverem " + condition.texto
            else:
                clause = f'que atenderem à condição "{condition.texto}"'
            if i:
                clause = (condition.operador or "e") + " " + clause
            chunks.append(clause)
        return " " + " ".join(chunks)
