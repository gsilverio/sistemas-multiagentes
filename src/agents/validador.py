"""Agente Validador: o GOBR gerado está correto e corresponde ao `RuleSpec`?

Sem LLM. Duas camadas:

1. `GobrValidator` (etapa 1): sintaxe, referências, ordem dos blocos e
   vocabulário. Avisos de vocabulário (A001/A002) também voltam ao Gerador:
   num bloco gerado, uma condição desconhecida quase sempre é invenção.
2. Conferência com o `RuleSpec`, bloco a bloco, pela AST: existe um bloco
   por segmento, `nao agendar` só onde foi pedido, frequência com o número de
   dias pedido e os limites da faixa presentes nas condições.

Os erros saem agrupados pelo nome do bloco, para o Gerador refazer só o
que está errado.
"""

from __future__ import annotations

import re

from src.agents.contratos import Conferencia, RuleSpec, Segmento
from src.gobr.ast import GobrParsedRule
from src.gobr.grammar import normalizar
from src.validation.gobr_validator import GobrValidator

FREQUENCIAS_NOMEADAS = {"diaria": 1, "semanal": 7, "quinzenal": 15, "mensal": 30, "bimestral": 60,
                        "trimestral": 90, "semestral": 180, "anual": 365}
CODIGOS_A_CORRIGIR = {"A001", "A002"}


def dias_da_frequencia(frequencia: str | None) -> int | None:
    if not frequencia:
        return None
    f = normalizar(frequencia)
    if m := re.fullmatch(r"(?:de\s+)?(\d+)\s+dias?", f):
        return int(m.group(1))
    for nome, dias in FREQUENCIAS_NOMEADAS.items():
        if re.match(rf"{nome}\s*\(", f):
            return dias
    return None  # dinâmica ou cascata: não dá para conferir o número


def _numeros(regra: GobrParsedRule) -> set[int]:
    texto = " ".join(c.texto for c in regra.condicoes)
    return {int(n) for n in re.findall(r"\d+", texto)}


def conferir_bloco(segmento: Segmento, regra: GobrParsedRule) -> list[str]:
    erros = []
    if segmento.nao_agendar and not regra.nao_agendar:
        erros.append("O segmento pede `nao agendar`, mas o bloco não tem `nao agendar`.")
    if not segmento.nao_agendar and regra.nao_agendar:
        erros.append("O bloco tem `nao agendar`, mas o segmento deve ser atendido.")
    if segmento.frequencia_dias and not segmento.nao_agendar:
        dias = dias_da_frequencia(regra.frequencia)
        if dias is None and not regra.frequencia:
            erros.append(f"Falta a frequência: `frequencia de {segmento.frequencia_dias} dias`.")
        elif dias is not None and dias != segmento.frequencia_dias:
            erros.append(f"A frequência do bloco é de {dias} dias; o segmento pede "
                         f"{segmento.frequencia_dias} dias (`frequencia de {segmento.frequencia_dias} dias`).")
    f = segmento.faixa
    if f and not segmento.nao_agendar:
        numeros = _numeros(regra)
        if f.max_dias and f.max_dias not in numeros:
            erros.append(f"A faixa do segmento vai até {f.max_dias} dias, mas {f.max_dias} não aparece nas condições.")
        elif not f.max_dias and f.min_dias and regra.condicoes and f.min_dias not in numeros:
            erros.append(f"O segmento é de clientes com mais de {f.min_dias} dias, mas {f.min_dias} "
                         "não aparece nas condições.")
    return erros


class AgenteValidador:
    def __init__(self, validador: GobrValidator | None = None):
        self.validador = validador or GobrValidator()

    def validar(self, gobr: str, spec: RuleSpec) -> Conferencia:
        relatorio = self.validador.validar(gobr)
        por_bloco: dict[str, list[str]] = {}
        gerais: list[str] = []
        avisos: list[str] = []
        nomes = {normalizar(s.nome.replace('"', "'")): s for s in spec.segmentos}
        for p in relatorio.problemas:
            texto = p.como_texto()
            alvo = normalizar(p.bloco) if p.bloco else None
            if p.severidade == "erro" or p.codigo in CODIGOS_A_CORRIGIR:
                if alvo in nomes:
                    por_bloco.setdefault(nomes[alvo].nome, []).append(texto)
                else:
                    gerais.append(texto)
            else:
                avisos.append(texto)

        presentes = {normalizar(n): r for n, r in relatorio.parametros.items()}
        for chave, segmento in nomes.items():
            regra = presentes.get(chave)
            if regra is None:
                if segmento.nome not in por_bloco:
                    por_bloco.setdefault(segmento.nome, []).append("Bloco ausente ou ilegível no arquivo.")
                continue
            for erro in conferir_bloco(segmento, regra):
                por_bloco.setdefault(segmento.nome, []).append(f"[CONFERÊNCIA] {erro}")

        return Conferencia(ok=not por_bloco and not gerais, erros_por_bloco=por_bloco,
                           erros_gerais=gerais, avisos=avisos)
