"""Parser sintatico conservador dos blocos `parametros` da GOBR.

Reconhece apenas construcoes documentadas/observadas; o conteudo nao
suportado permanece em `desconhecidos` para impedir indexacao silenciosa.
"""

import re
import unicodedata

from src.models.gobr_parsed_rule import (
    GobrParsedRule,
    ParsedCondition,
    ParsedScope,
)


def _sem_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto.casefold())
    return "".join(c for c in normalizado if not unicodedata.combining(c))


class GobrParser:
    # O titulo pode conter aspas internas (inclusive parametros com CNAEs).
    # O GobrRuleExtractor ja preserva essas aspas; nao devemos perder a regra.
    PARAMETROS_REGEX = re.compile(
        r'^parametros\s+"(.*)"\s*:\s*(?:#.*)?$',
        re.IGNORECASE,
    )
    # Argumentos sao uma lista de literais entre aspas, separados por virgula.
    # A sintaxe interna e validada separadamente para nao aceitar expressoes
    # arbitrarias nem perder silenciosamente argumentos malformados.
    SCOPE_PATTERNS = (
        ("time", re.compile(r'^do\s+time\s*\((?P<args>.*)\)$', re.IGNORECASE)),
        ("carteira", re.compile(r'^da\s+carteira\s*\((?P<args>.*)\)$', re.IGNORECASE)),
        ("segmentacao", re.compile(
            r'^da\s+segmenta[cç][aã]o\s*\((?P<args>.*)\)$',
            re.IGNORECASE,
        )),
    )
    QUOTED_ARGS = re.compile(r'^\s*"[^"\r\n]+"\s*(?:,\s*"[^"\r\n]+"\s*)*$')
    QUOTED_VALUE = re.compile(r'"([^"\r\n]+)"')

    def parse(self, gobr: str) -> GobrParsedRule:
        lines = self._linhas_logicas(gobr)
        result = GobrParsedRule(nome=self._extract_name(lines))
        condition_started = False
        preference_started = False

        for line in lines:
            normalized = _sem_acentos(line)

            if self.PARAMETROS_REGEX.match(line):
                continue

            if self._parse_scope(line, result):
                preference_started = False
                continue

            if normalized.startswith("quando "):
                result.condicoes.append(
                    ParsedCondition(texto=line[7:].strip(), operador=None)
                )
                condition_started = True
                preference_started = False
                continue

            # Preferencia de intensidade pode ter varias linhas:
            # preferencia\n  maior na segunda e\n  menor na terca
            # Guardamos todas sem deduzir pesos ou preencher dias ausentes.
            if preference_started and self._continua_preferencia(normalized):
                result.modificadores[-1] += "\n" + line
                continue

            if normalized.startswith("frequencia "):
                result.frequencia = self._parse_frequency(line)
                preference_started = False
                continue

            if normalized == "nao agendar":
                result.nao_agendar = True
                preference_started = False
                continue

            if self._is_modifier(line):
                result.modificadores.append(line)
                preference_started = normalized == "preferencia"
                continue

            if (
                condition_started
                and normalized.startswith("e ")
                and not self._looks_like_scope(line[2:].strip())
            ):
                result.condicoes.append(
                    ParsedCondition(texto=line[2:].strip(), operador="e")
                )
                preference_started = False
                continue

            if (
                condition_started
                and normalized.startswith("ou ")
                and not self._looks_like_scope(line[3:].strip())
            ):
                result.condicoes.append(
                    ParsedCondition(texto=line[3:].strip(), operador="ou")
                )
                preference_started = False
                continue

            result.desconhecidos.append(line)
            preference_started = False

        return result

    @staticmethod
    def _linhas_logicas(gobr: str) -> list[str]:
        """Junta continuacoes de argumentos entre parenteses; ignora comentarios.

        Nao inventa conteudo e mantem as clausulas multivalor nao reconhecidas
        em `desconhecidos`, para uma etapa posterior de suporte completo.
        """
        result: list[str] = []
        acumulado = ""
        balance = 0

        for original in gobr.splitlines():
            line = original.strip()
            if not line or line.startswith("#"):
                continue
            if acumulado:
                acumulado += " " + line
            else:
                acumulado = line
            balance += line.count("(") - line.count(")")
            if balance <= 0:
                result.append(acumulado)
                acumulado = ""
                balance = 0

        if acumulado:
            result.append(acumulado)  # Sintaxe incompleta continua detectavel.
        return result

    def _extract_name(self, lines: list[str]) -> str:
        for line in lines:
            match = self.PARAMETROS_REGEX.match(line)
            if match:
                return match.group(1)
        raise ValueError("Bloco parametros não encontrado.")

    def _parse_scope(self, line: str, result: GobrParsedRule) -> bool:
        operador = None
        normalized = _sem_acentos(line)
        if normalized.startswith("e "):
            operador = "e"
            line = line[2:].strip()
        elif normalized.startswith("ou "):
            operador = "ou"
            line = line[3:].strip()

        for tipo, pattern in self.SCOPE_PATTERNS:
            match = pattern.fullmatch(line)
            if not match:
                continue
            argumentos = match.group("args")
            if not self.QUOTED_ARGS.fullmatch(argumentos):
                return False  # Permanece em desconhecidos; nao reduz a 1 valor.
            valores = self.QUOTED_VALUE.findall(argumentos)
            result.escopos.append(
                ParsedScope(
                    tipo=tipo,
                    valor=valores[0],
                    valores=valores,
                    operador=operador,
                )
            )
            return True
        return False

    @staticmethod
    def _parse_frequency(line: str) -> str:
        # Comprimento e identico para "frequencia" e "frequência".
        value = line[len("frequencia"):].strip()
        if _sem_acentos(value).startswith("de "):
            value = value[3:].strip()
        return value

    @staticmethod
    def _is_modifier(line: str) -> bool:
        normalized = _sem_acentos(line)
        return (
            normalized.startswith("preferencia ")
            or normalized == "preferencia"
            or normalized == "na ultima semana"
            or normalized in ("fixo", "roteirizavel")
            or normalized.startswith("sem choque ")
            or bool(re.fullmatch(r"a partir do dia\s+\d{1,2}", normalized))
        )

    @staticmethod
    def _continua_preferencia(normalized: str) -> bool:
        return bool(re.match(r"^(?:e\s+)?(?:maior|menor|normal|na|no|nos|nas)\s+", normalized))

    def _looks_like_scope(self, line: str) -> bool:
        # Mesmo uma lista malformada precisa continuar como desconhecida,
        # nao virar condicao acidentalmente depois de `quando`.
        return any(pattern.fullmatch(line) for _, pattern in self.SCOPE_PATTERNS)
