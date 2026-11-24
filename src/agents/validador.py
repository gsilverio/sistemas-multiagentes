"""Agente Validador: o GOBR gerado está correto e corresponde ao `RuleSpec`?

Sem LLM. Duas camadas:

1. `GobrValidator` (etapa 1): sintaxe, referências, ordem dos blocos e
   vocabulário. Avisos de vocabulário (A001/A002) também voltam ao Gerador:
   num bloco gerado, uma condição desconhecida quase sempre é invenção.
2. Conferência com o `RuleSpec`, bloco a bloco, pela AST: existe um bloco
   por segmento, `nao agendar` só onde foi pedido, frequência com o número de
   dias pedido, os limites da faixa presentes nas condições e as preferências
   de dia pedidas. A faixa também é conferida **por comportamento**: clientes
   simulados dentro e fora da faixa do segmento precisam ser pegos (ou não)
   pelo bloco. É o que apanha o erro mais comum dos modelos, escrever
   `fora da recencia em 90` para quem comprou há até 90 dias: o número está
   certo e a condição, invertida.

Os erros saem agrupados pelo nome do bloco, para o Gerador refazer só o
que está errado.
"""

from __future__ import annotations

import re

from src.agents.contratos import Conferencia, RuleSpec, Segmento, so_fala_da_faixa
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


_DIAS_SEMANA = ("segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo")
_NUNCA_COMPROU = re.compile(r"\bprospects?\b|\bnunca (?:compr|tir|fizeram)")
CONDICAO_PROSPECT = "quando no ciclo de vida prospect com a empresa"
# Grupos de bloqueio frequentes: o que o texto diz, o que a condição precisa citar e a forma padrão.
_CONCEITOS = (
    (re.compile(r"\binadimpl"), "inadimpl", "quando inadimplentes"),
    (re.compile(r"\breceita federal\b|\bbaixad[oa]s?\b"), "receita federal",
     "quando status na receita federal definido como baixada"),
    (re.compile(r"\brotas?\b"), "rota", "quando data da rota na semana"),
)
_TODOS_OS_CLIENTES = re.compile(r"\b(?:demais|todos|todas|restante|outros|qualquer|nao se enquadr)")


def _dias_citados(texto: str) -> set[str]:
    t = normalizar(texto)
    dias = {d for d in _DIAS_SEMANA if re.search(rf"\b{d}s?\b", t)}
    return dias | ({"uteis"} if re.search(r"\bdias uteis\b", t) else set())


def _preferencia_gobr(dias: set[str]) -> str:
    partes = [("no " if d in ("sabado", "domingo") else "na ") + d for d in _DIAS_SEMANA if d in dias]
    if "uteis" in dias:
        partes.insert(0, "nos dias uteis")
    return "preferencia " + " e ".join(partes)


def conferir_modificadores(segmento: Segmento, regra: GobrParsedRule) -> list[str]:
    """Preferência de dia pedida e ausente (ou diferente), e modificadores que ninguém pediu."""
    pedido = normalizar(" ".join([segmento.nome, segmento.criterio, segmento.frequencia_texto or "",
                                  *segmento.restricoes]))
    pedidos = _dias_citados(pedido)
    preferencias = [m for m in regra.modificadores if normalizar(m).startswith("preferencia")]
    erros = []
    if "prefer" in pedido and pedidos and not segmento.nao_agendar \
            and _dias_citados(" ".join(preferencias)) != pedidos:
        erros.append("O segmento pede preferência de dia: escreva, depois da frequência, a linha "
                     f"`{_preferencia_gobr(pedidos)}`.")
    elif segmento.restricoes:  # sem restrição nenhuma, `conferir_bloco` já pede para remover todos
        sobrando = [m for m in preferencias if "prefer" not in pedido and not pedidos]
        sobrando += [m for m in regra.modificadores if normalizar(m) == "fixo" and "fix" not in pedido]
        if sobrando:
            erros.append("O bloco tem modificadores que o segmento não pede ("
                         + "; ".join(f"`{m}`" for m in sobrando) + "); remova-os.")
    return erros


def _condicao_da_faixa(minimo: int, maximo: int | None) -> str:
    if maximo is None:
        return f"quando fora da recencia de positivacao em {minimo} dias"
    if minimo:
        return f"quando recencia de positivacao entre {minimo} dias e {maximo} dias"
    return f"quando recencia de positivacao em {maximo} dias"


def conferir_faixa(segmento: Segmento, regra: GobrParsedRule, spec: RuleSpec) -> list[str]:
    """O bloco pega os clientes da faixa do segmento, e só eles?

    Clientes mais recentes que a faixa podem ser pegos quando uma regra
    anterior já os atende (o cliente fica com a primeira, manual 7.6).
    """
    from src.gobr.simulacao import Cliente, _atende, _limites, _PROSPECT, _termos

    f = segmento.faixa
    if f is None or f.medida != "dias_sem_compra" or not (f.min_dias or f.max_dias):
        return []
    termos = [t for c in regra.condicoes for t in _termos(c.texto)]
    if termos and not any(_limites(t) for t in termos):
        return []  # condição sem número de dias: não dá para simular
    opacos = frozenset(t for t in termos if not _limites(t) and not _PROSPECT.match(t))
    if opacos and not segmento.restricoes and so_fala_da_faixa(segmento.criterio):
        return [f"O grupo é definido só pela faixa de dias, mas o bloco tem outras condições "
                f"({'; '.join(f'`{t}`' for t in sorted(opacos))}); remova-as."]

    def pega(recencia: float | None) -> bool:
        return _atende(regra, Cliente(recencia, opacos))

    minimo, maximo = f.min_dias or 0, f.max_dias
    if maximo is not None and maximo <= minimo:
        return []
    correta = _condicao_da_faixa(minimo, maximo)
    dentro = (minimo + maximo + 1) // 2 if maximo is not None else minimo + 30
    if not pega(dentro):
        return [f"O bloco não pega um cliente que comprou há {dentro} dias, que é deste grupo "
                f"({f.como_texto()}). Condição correta: `{correta}`."]
    if maximo is not None and pega(maximo + 30):
        return [f"O bloco pega um cliente que comprou há {maximo + 30} dias, que não é deste grupo "
                f"({f.como_texto()}). Condição correta: `{correta}`."]

    indice = next((i for i, s in enumerate(spec.segmentos) if s is segmento or s.nome == segmento.nome), 0)
    anteriores = [s.faixa for s in spec.segmentos[:indice]
                  if s.faixa and s.faixa.medida == f.medida and (s.faixa.min_dias or s.faixa.max_dias)]
    limites = sorted({0, minimo, *(n for a in anteriores for n in (a.min_dias or 0, a.max_dias or 0))})
    for inicio, fim in zip(limites, limites[1:]):
        meio = (inicio + fim) / 2
        if fim > minimo or fim - inicio < 2:
            continue
        coberto = any((a.min_dias or 0) <= meio <= (a.max_dias if a.max_dias is not None else meio)
                      for a in anteriores)
        if pega(meio) and not coberto:
            return [f"O bloco pega um cliente que comprou há {meio:g} dias, que não é deste grupo "
                    f"({f.como_texto()}) nem de uma regra anterior. Condição correta: `{correta}`."]
    if pega(None) and indice + 1 < len(spec.segmentos):
        return ["O bloco pega também quem nunca comprou, antes das regras seguintes. "
                f"Condição correta: `{correta}`."]
    return []


def conferir_prospects(segmento: Segmento, regra: GobrParsedRule) -> list[str]:
    """Um grupo de quem nunca comprou precisa pegar quem nunca comprou."""
    from src.gobr.simulacao import Cliente, _atende, _limites, _PROSPECT, _termos

    texto = normalizar(segmento.nome + " " + segmento.criterio)
    if segmento.faixa or segmento.fallback or not _NUNCA_COMPROU.search(texto):
        return []
    termos = [t for c in regra.condicoes for t in _termos(c.texto)]
    extras = {t for t in termos if not _limites(t) and not _PROSPECT.match(t)}
    if not any(_PROSPECT.match(t) for t in termos) and extras:
        return []  # outra forma de dizer prospect; não dá para simular
    atributos = frozenset(extras) if segmento.restricoes else frozenset()
    if _atende(regra, Cliente(None, atributos)):
        return []
    return [f"O bloco não pega quem nunca comprou. Condição correta: `{CONDICAO_PROSPECT}`."]


def conferir_conceito(segmento: Segmento, regra: GobrParsedRule) -> tuple[str, str] | None:
    """(erro, condição padrão) quando o grupo é de inadimplentes, baixados ou rota e o bloco fala de outra coisa.

    Só para bloqueios que citam um único conceito.
    """
    if segmento.faixa or segmento.fallback or not segmento.nao_agendar:
        return None
    texto = normalizar(segmento.nome + " " + segmento.criterio)
    citados = [(palavra, padrao) for busca, palavra, padrao in _CONCEITOS if busca.search(texto)]
    if len(citados) != 1:
        return None
    palavra, padrao = citados[0]
    if any(palavra in normalizar(c.texto) for c in regra.condicoes):
        return None
    return f"A condição do bloco não corresponde ao grupo. Condição esperada: `{padrao}`.", padrao


def conferir_bloco(segmento: Segmento, regra: GobrParsedRule, spec: RuleSpec | None = None) -> list[str]:
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
    if regra.modificadores and not segmento.restricoes:
        erros.append("O bloco tem modificadores que o segmento não pede ("
                     + "; ".join(f"`{m}`" for m in regra.modificadores) + "); remova-os.")
    erros += conferir_modificadores(segmento, regra)
    erros += conferir_prospects(segmento, regra)
    if conceito := conferir_conceito(segmento, regra):
        erros.append(conceito[0])
    if not regra.condicoes and segmento.criterio.strip() and not segmento.fallback and not segmento.faixa \
            and not _TODOS_OS_CLIENTES.search(normalizar(segmento.nome + " " + segmento.criterio)):
        erros.append("O bloco não tem condição e pegaria todos os clientes; falta o `quando` que define "
                     f'"{segmento.criterio.strip()}".')
    f = segmento.faixa
    if f and not segmento.nao_agendar:
        numeros = _numeros(regra)
        if f.max_dias and f.max_dias not in numeros:
            erros.append(f"A faixa do segmento vai até {f.max_dias} dias, mas {f.max_dias} não aparece nas condições.")
        elif not f.max_dias and f.min_dias and regra.condicoes and f.min_dias not in numeros:
            erros.append(f"O segmento é de clientes com mais de {f.min_dias} dias, mas {f.min_dias} "
                         "não aparece nas condições.")
    if f and spec is not None:
        erros += conferir_faixa(segmento, regra, spec)
    return erros


def reparar_bloco(segmento: Segmento, regra: GobrParsedRule, spec: RuleSpec, escopo: str) -> str | None:
    """Reescreve, sem LLM, as linhas que a conferência com o `RuleSpec` reprovou.

    Último recurso, quando o Gerador já errou o mesmo bloco mais de uma vez:
    a faixa, a frequência e a preferência estão no `RuleSpec`, e o resto do
    bloco do Gerador é mantido. None se não houver o que reparar com segurança
    (o bloco reparado ainda seria reprovado).
    """
    from src.gobr.parser import GobrParser

    condicoes = [f"{c.operador} {c.texto}" if c.operador in ("e", "ou") else f"quando {c.texto}"
                 for c in regra.condicoes]
    if segmento.faixa and (conferir_faixa(segmento, regra, spec) or not condicoes):
        condicoes = [_condicao_da_faixa(segmento.faixa.min_dias or 0, segmento.faixa.max_dias)]
    if conferir_prospects(segmento, regra):
        condicoes = [CONDICAO_PROSPECT]
    if conceito := conferir_conceito(segmento, regra):
        condicoes = [conceito[1]]
    if segmento.nao_agendar:
        acao = "nao agendar"
    elif segmento.frequencia_dias:
        acao = f"frequencia de {segmento.frequencia_dias} dias"
    elif regra.frequencia:
        acao = f"frequencia {regra.frequencia}"
    else:
        return None
    modificadores = [] if segmento.nao_agendar else list(regra.modificadores)
    if conferir_modificadores(segmento, regra) or (modificadores and not segmento.restricoes):
        pedido = normalizar(" ".join([segmento.nome, segmento.criterio, *segmento.restricoes]))
        modificadores = [m for m in modificadores if not normalizar(m).startswith("preferencia")
                         and normalizar(m) != "fixo" and segmento.restricoes]
        if "prefer" in pedido and _dias_citados(pedido):
            modificadores.insert(0, _preferencia_gobr(_dias_citados(pedido)))
    nome = segmento.nome.replace('"', "'")
    linhas = [escopo, *condicoes, acao, *(l.strip() for m in modificadores for l in m.splitlines()),
              *(regra.filtros_temporais if not segmento.nao_agendar else [])]
    bloco = "\n".join([f'parametros "{nome}":', *("    " + l for l in linhas)])
    try:
        nova = GobrParser().parse(bloco)
    except ValueError:
        return None
    return None if nova.desconhecidos or conferir_bloco(segmento, nova, spec) else bloco


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
            for erro in conferir_bloco(segmento, regra, spec):
                por_bloco.setdefault(segmento.nome, []).append(f"[CONFERÊNCIA] {erro}")

        return Conferencia(ok=not por_bloco and not gerais, erros_por_bloco=por_bloco,
                           erros_gerais=gerais, avisos=avisos)
