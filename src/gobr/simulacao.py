"""Acerto por comportamento: o GOBR gerado trata os clientes como o esperado?

Em vez de comparar o texto dos blocos, aplica os dois arquivos a clientes
simulados e confere, para cada um, se o tratamento é o mesmo: não agendar, ou
agendar com a mesma frequência e as mesmas preferências. Duas redações
equivalentes (`recencia em 45` depois da regra de 30 × `recencia entre 30 e
45`) dão o mesmo resultado; uma faixa, frequência ou ordem errada aparece nos
clientes afetados.

Como o cliente é tratado (manual, 7.6): fica com a **primeira** regra cujas
condições ele atende; sem nenhuma, não é agendado (o mesmo que `nao agendar`).

Clientes simulados:
- recência da última compra: um valor no meio de cada faixa formada pelos
  limites de dias dos dois arquivos, mais "nunca comprou" (prospect);
- cada outra condição (inadimplentes, baixados na Receita, ...) é um atributo
  sim/não independente.

Peso: cada faixa de recência (com todos os atributos "não") vale 1; cada
atributo vale 1 no total, dividido entre as faixas. Assim cada grupo que o
usuário descreveu pesa parecido, e as regras de "não agendar" não dominam a
conta só por se combinarem com todas as faixas.

Simplificações: recência global, do vendedor e do time são a mesma variável;
"ativos/inativos para o vendedor" e afins são atributos opacos; escopos
(time, carteira) não contam, só o tratamento.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.validador import dias_da_frequencia
from src.gobr.ast import GobrParsedRule
from src.gobr.grammar import normalizar
from src.validation.gobr_validator import GobrValidator

_RECENCIA = re.compile(r"^(?P<fora>fora da )?recencia de positivacao(?: global| do vendedor| do time)?\s+"
                       r"(?:em (?P<n>\d+) dias?|entre (?P<a>\d+) dias? e (?P<b>\d+) dias?)$")
_INATIVIDADE = re.compile(r"^inatividade(?: global)? de (?P<n>\d+) dias?$")
_PROSPECT = re.compile(r"^(?:no ciclo de vida prospect com (?:a empresa|o vendedor)|prospects?(?: da empresa| do vendedor)?"
                       r"|sem vendas?(?: ou prospects)?(?: do vendedor)?)$")
_E_INLINE = re.compile(r"\s+e\s+(?!\d)")  # "inativos e inadimplentes"; não quebra "entre 30 dias e 60 dias"
_DIAS = ("segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo")
NAO_AGENDAR = "não agendar"


@dataclass(frozen=True)
class Cliente:
    recencia: float | None          # dias desde a última compra; None = nunca comprou
    atributos: frozenset[str]       # condições opacas verdadeiras para este cliente

    def descricao(self) -> str:
        partes = ["nunca comprou" if self.recencia is None else f"última compra há {self.recencia:g} dias"]
        partes += sorted(self.atributos)
        return ", ".join(partes)


def _termos(texto: str) -> list[str]:
    t = normalizar(texto)
    if _RECENCIA.match(t) or _INATIVIDADE.match(t):
        return [t]
    return [p.strip() for p in _E_INLINE.split(t) if p.strip()]


def _limites(termo: str) -> list[int]:
    if m := _RECENCIA.match(termo):
        return [int(m["n"])] if m["n"] else [int(m["a"]), int(m["b"])]
    if m := _INATIVIDADE.match(termo):
        return [int(m["n"])]
    return []


def _atende_termo(termo: str, c: Cliente) -> bool:
    if m := _RECENCIA.match(termo):
        if c.recencia is None:
            return False  # "fora da recência" exige uma última compra (manual, 6.2)
        dentro = c.recencia <= int(m["n"]) if m["n"] else int(m["a"]) <= c.recencia <= int(m["b"])
        return dentro != bool(m["fora"])
    if m := _INATIVIDADE.match(termo):
        return c.recencia is not None and c.recencia >= int(m["n"])
    if _PROSPECT.match(termo):
        return c.recencia is None
    return termo in c.atributos


def _atende(regra: GobrParsedRule, c: Cliente) -> bool:
    if not regra.condicoes:
        return True
    grupos: list[bool] = []
    for cond in regra.condicoes:
        valor = all(_atende_termo(t, c) for t in _termos(cond.texto))
        if cond.operador == "ou" or not grupos:
            grupos.append(valor)
        else:
            grupos[-1] = grupos[-1] and valor
    return any(grupos)


def _modificador(texto: str) -> str:
    t = normalizar(" ".join(texto.split()))
    if t.startswith("preferencia") and any(d in t for d in _DIAS + ("uteis",)):
        intensidade = re.findall(r"\b(maior|menor)\b", t)
        dias = sorted({d for d in _DIAS if re.search(rf"\b{d}s?\b", t)} | ({"uteis"} if "uteis" in t else set()))
        return "preferencia " + "/".join(intensidade) + " " + ",".join(dias)
    return t


def tratamento(regra: GobrParsedRule | None) -> str:
    if regra is None or regra.nao_agendar:
        return NAO_AGENDAR
    dias = dias_da_frequencia(regra.frequencia)
    texto = f"a cada {dias} dias" if dias is not None else f"frequência {normalizar(regra.frequencia or '?')}"
    extras = sorted(_modificador(m) for m in regra.modificadores + regra.filtros_temporais)
    return texto + (" + " + "; ".join(extras) if extras else "")


def tratar(regras: list[GobrParsedRule], c: Cliente) -> str:
    return tratamento(next((r for r in regras if _atende(r, c)), None))


def _regras(gobr: str | None) -> list[GobrParsedRule]:
    return list(GobrValidator().validar(gobr).parametros.values()) if gobr else []


def clientes(*conjuntos: list[GobrParsedRule]) -> list[tuple[Cliente, float]]:
    """Clientes simulados e o peso de cada um (ver o docstring do módulo)."""
    limites, atributos = {0}, set()
    for regras in conjuntos:
        for regra in regras:
            for cond in regra.condicoes:
                for termo in _termos(cond.texto):
                    if ls := _limites(termo):
                        limites.update(ls)
                    elif not _PROSPECT.match(termo):
                        atributos.add(termo)
    pontos = sorted(limites)
    # Um dia inteiro no meio de cada faixa; faixas sem dia inteiro dentro (30 e 31) não existem
    # na prática e são puladas, para "em 30" + "entre 31 e 60" valer o mesmo que "em 30" + "em 60".
    recencias: list[float | None] = [(a + b) // 2 for a, b in zip(pontos, pontos[1:]) if b - a >= 2]
    recencias += [pontos[-1] + 30, None]
    pesos = [(Cliente(r, frozenset()), 1.0) for r in recencias]
    for atributo in sorted(atributos):
        pesos += [(Cliente(r, frozenset({atributo})), 1 / len(recencias)) for r in recencias]
    return pesos


def acerto_comportamento(gobr_gerado: str | None, gobr_esperado: str, *, max_exemplos: int = 8) -> dict[str, Any]:
    """`acerto` = fração ponderada dos clientes simulados tratados como no esperado."""
    if not (gobr_gerado or "").strip():
        return {"acerto": 0.0, "clientes_simulados": 0, "erros_comportamento": []}
    esperadas, geradas = _regras(gobr_esperado), _regras(gobr_gerado)
    populacao = clientes(esperadas, geradas)
    total = certo = 0.0
    erros = []
    for cliente, peso in populacao:
        esperado, gerado = tratar(esperadas, cliente), tratar(geradas, cliente)
        total += peso
        if esperado == gerado:
            certo += peso
        else:
            erros.append({"cliente": cliente.descricao(), "esperado": esperado, "gerado": gerado,
                          "peso": round(peso, 3)})
    erros.sort(key=lambda e: -e["peso"])
    return {"acerto": round(certo / total, 3) if total else 1.0, "clientes_simulados": len(populacao),
            "erros_comportamento": erros[:max_exemplos]}
