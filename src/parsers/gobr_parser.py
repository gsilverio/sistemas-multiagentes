import re

from src.models.gobr_parsed_rule import (
    GobrParsedRule,
    ParsedCondition,
    ParsedScope,
)


class GobrParser:

    PARAMETROS_REGEX = re.compile(
        r'^parametros\s+"([^"]+)"\s*:$',
        re.IGNORECASE
    )

    TIME_REGEX = re.compile(
        r'^do\s+time\s*\(\s*"([^"]+)"\s*\)$',
        re.IGNORECASE
    )

    CARTEIRA_REGEX = re.compile(
        r'^da\s+carteira\s*\(\s*"([^"]+)"\s*\)$',
        re.IGNORECASE
    )

    SEGMENTACAO_REGEX = re.compile(
        r'^da\s+segmenta[cç][aã]o\s*\(\s*"([^"]+)"\s*\)$',
        re.IGNORECASE
    )

    def parse(self, gobr: str) -> GobrParsedRule:
        lines = [
            line.strip()
            for line in gobr.splitlines()
            if line.strip()
        ]

        nome = self._extract_name(lines)

        result = GobrParsedRule(
            nome=nome
        )

        condition_started = False

        for line in lines:
            if self.PARAMETROS_REGEX.match(line):
                continue

            if self._parse_scope(line, result):
                continue

            if line.lower().startswith("quando "):
                texto = line[7:].strip()

                result.condicoes.append(
                    ParsedCondition(
                        texto=texto,
                        operador=None
                    )
                )

                condition_started = True
                continue

            if (
                condition_started
                and line.lower().startswith("e ")
                and not self._looks_like_scope(line[2:].strip())
            ):
                result.condicoes.append(
                    ParsedCondition(
                        texto=line[2:].strip(),
                        operador="e"
                    )
                )

                continue

            if (
                condition_started
                and line.lower().startswith("ou ")
                and not self._looks_like_scope(line[3:].strip())
            ):
                result.condicoes.append(
                    ParsedCondition(
                        texto=line[3:].strip(),
                        operador="ou"
                    )
                )

                continue

            if line.lower().startswith("frequencia "):
                result.frequencia = self._parse_frequency(
                    line
                )
                continue

            if line.lower() == "nao agendar":
                result.nao_agendar = True
                continue

            if self._is_modifier(line):
                result.modificadores.append(
                    line
                )
                continue

            result.desconhecidos.append(
                line
            )

        return result


    def _extract_name(
        self,
        lines: list[str]
    ) -> str:

        for line in lines:
            match = self.PARAMETROS_REGEX.match(
                line
            )

            if match:
                return match.group(1)

        raise ValueError(
            "Bloco parametros não encontrado."
        )

    def _parse_scope(
        self,
        line: str,
        result: GobrParsedRule
    ) -> bool:

        operador = None
        original = line

        if line.lower().startswith("e "):
            operador = "e"
            line = line[2:].strip()

        elif line.lower().startswith("ou "):
            operador = "ou"
            line = line[3:].strip()

        match = self.TIME_REGEX.match(line)

        if match:
            result.escopos.append(
                ParsedScope(
                    tipo="time",
                    valor=match.group(1),
                    operador=operador
                )
            )
            return True

        match = self.CARTEIRA_REGEX.match(
            line
        )

        if match:
            result.escopos.append(
                ParsedScope(
                    tipo="carteira",
                    valor=match.group(1),
                    operador=operador
                )
            )
            return True

        match = self.SEGMENTACAO_REGEX.match(
            line
        )

        if match:
            result.escopos.append(
                ParsedScope(
                    tipo="segmentacao",
                    valor=match.group(1),
                    operador=operador
                )
            )
            return True

        return False

    def _parse_frequency(
        self,
        line: str
    ) -> str:

        value = line[
            len("frequencia")::
        ].strip()

        if value.lower().startswith("de "):
            value = value[3:].strip()

        return value
    
    def _is_modifier(
        self,
        line: str
    ) -> bool:

        normalized = line.lower()

        return (
            normalized.startswith("preferencia ")
            or normalized == "na ultima semana"
        )

    def _looks_like_scope(
        self,
        line: str
    ) -> bool:

        return any([
            self.TIME_REGEX.match(line),
            self.CARTEIRA_REGEX.match(line),
            self.SEGMENTACAO_REGEX.match(line)
        ])