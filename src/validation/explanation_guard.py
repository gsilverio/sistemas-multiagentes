from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.models.gobr_parsed_rule import GobrParsedRule, ParsedCondition, ParsedScope


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

        # Escopos: nomes exatos após normalização básica, inclusive 'empresa toda'.
        positions = []
        for scope in parsed.escopos:
            target = _normalize(scope.valor)
            position = text.find(target)
            if position < 0:
                issues.append(f'Escopo não preservado literalmente: {scope.tipo}("{scope.valor}").')
            positions.append((position, position + len(target)))

        # O operador deve estar entre os dois escopos, e não entre dois
        # conjuntos independentes de 'clientes'.
        for i, scope in enumerate(parsed.escopos[1:], start=1):
            prev_start, prev_end = positions[i - 1]
            next_start, _ = positions[i]
            if prev_start < 0 or next_start < 0:
                continue
            if next_start < prev_end:
                issues.append("A ordem dos escopos foi alterada.")
                continue
            middle = text[prev_end:next_start]
            operator = _normalize(scope.operador or "e")
            if not re.search(rf"\b{re.escape(operator)}\b", middle):
                issues.append(f"Operador entre escopos não preservado: {operator}.")
            elif operator == "e" and re.search(r"\be\s+(?:os\s+)?clientes\b", middle):
                issues.append("A conjunção foi apresentada como públicos separados.")

        for condition in parsed.condicoes:
            anchor = _condition_anchor(condition.texto)
            if anchor and anchor not in text:
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
        if not scopes:
            return "Os clientes"
        chunks: list[str] = []
        for i, scope in enumerate(scopes):
            if scope.tipo == "carteira":
                part = f'da carteira "{scope.valor}"'
            elif scope.tipo == "segmentacao":
                part = f'da segmentação "{scope.valor}"'
            elif scope.tipo == "time":
                part = f'do time "{scope.valor}"'
            else:
                part = f'de {scope.tipo} "{scope.valor}"'
            if i:
                part = f"{scope.operador or 'e'} {part}"
            chunks.append(part)
        return "Os clientes " + " ".join(chunks)

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
