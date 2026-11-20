"""Tradução determinística das construções GOBR para português.

Cada função recebe um trecho de um bloco `parametros` (já separado pelo
parser) e devolve uma `Traducao`:

- `frase`: o significado em português, conforme o manual;
- `conhecida`: False quando o trecho não tem tradução (a frase então cita o
  texto original entre aspas, sem inventar significado);
- `chaves`: expressões regulares (sobre texto normalizado, sem acentos) que
  um texto precisa conter para ser considerado fiel a este trecho. São
  usadas pelo guard (`src/explicacao/guarda.py`).

Fontes no manual: filtros de clientes (cap. 6), frequência (7.4),
preferência (7.2 e 7.3), restrito (7.5), filtros temporais (cap. 5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.gobr.grammar import normalizar


@dataclass(frozen=True)
class Traducao:
    frase: str
    conhecida: bool = True
    chaves: tuple[str, ...] = field(default_factory=tuple)


def _literal(texto: str) -> str:
    """Chave que exige o trecho original (para o que não tem tradução)."""
    return re.escape(normalizar(texto))


def _desconhecida(texto: str, frase: str) -> Traducao:
    return Traducao(frase, conhecida=False, chaves=(_literal(texto),))


def valores_entre_aspas(texto: str) -> list[str]:
    return [a or b for a, b in re.findall(r'"([^"]*)"|\'([^\']*)\'', texto)]


def _lista(valores: list[str]) -> str:
    return ", ".join(f'"{v}"' for v in valores)


# ---------------------------------------------------------------------------
# Tempo (manual 6.11 a 6.13)
# ---------------------------------------------------------------------------

_SINGULAR = {"ano": "ano", "semestre": "semestre", "trimestre": "trimestre",
             "bimestre": "bimestre", "mes": "mês", "semana": "semana", "dia": "dia",
             "hora": "hora", "minuto": "minuto"}
_PLURAL = {"ano": "anos", "semestre": "semestres", "trimestre": "trimestres",
           "bimestre": "bimestres", "mes": "meses", "semana": "semanas", "dia": "dias",
           "hora": "horas", "minuto": "minutos"}
_U = r"(anos?|semestres?|trimestres?|bimestres?|mes|meses|semanas?|dias?|horas?|minutos?)"


def _qtd(n: str, unidade: str) -> str:
    base = "mes" if unidade.startswith("mes") else unidade.rstrip("s")
    return f"{n} {_SINGULAR[base] if n == '1' else _PLURAL[base]}"


_TEMPO_FIXO = {
    "hoje": "hoje", "de hoje": "hoje", "do dia": "hoje", "no dia": "hoje",
    "ontem": "ontem", "de ontem": "ontem",
    "na semana": "na semana atual", "na quinzena": "na quinzena atual",
    "no mes": "no mês atual", "no bimestre": "no bimestre atual",
    "no trimestre": "no trimestre atual", "no semestre": "no semestre atual",
    "no ano": "no ano atual",
    "na semana anterior": "na semana anterior", "na quinzena anterior": "na quinzena anterior",
    "no mes anterior": "no mês anterior", "no bimestre anterior": "no bimestre anterior",
    "no trimestre anterior": "no trimestre anterior",
    "no semestre anterior": "no semestre anterior",
    "no ano anterior": "no ano anterior", "no ano passado": "no ano anterior",
    "na manha": "na parte da manhã", "na tarde": "na parte da tarde",
    "na noite": "na parte da noite",
}


def _tempo(t: str) -> Optional[str]:
    t = t.strip()
    if t in _TEMPO_FIXO:
        return _TEMPO_FIXO[t]
    if m := re.fullmatch(rf"(?:em |de )?(\d+) {_U}", t):
        return f"nos últimos {_qtd(m[1], m[2])}"
    if m := re.fullmatch(rf"entre (\d+) {_U} e (\d+) {_U}", t):
        return f"entre {_qtd(m[1], m[2])} e {_qtd(m[3], m[4])} atrás"
    if m := re.fullmatch(r"entre '([^']+)' e '([^']+)'", t):
        return f"entre {m[1]} e {m[2]}"
    if m := re.fullmatch(rf"ha (\d+) {_U} atras", t):
        return f"há mais de {_qtd(m[1], m[2])}"
    if m := re.fullmatch(rf"nos proximos (\d+) {_U}", t):
        return f"nos próximos {_qtd(m[1], m[2])}"
    if m := re.fullmatch(rf"em (\d+) {_U} a frente", t):
        return f"daqui a mais de {_qtd(m[1], m[2])}"
    return None


# ---------------------------------------------------------------------------
# Condições `quando` (manual cap. 6)
# ---------------------------------------------------------------------------

_Regra = Callable[[re.Match, str], Optional[Traducao]]
_REGRAS: list[tuple[re.Pattern, _Regra]] = []


def _regra(padrao: str):
    rx = re.compile(padrao)

    def registrar(fn: _Regra) -> _Regra:
        _REGRAS.append((rx, fn))
        return fn
    return registrar


_NEG = r"\bnao\b|\bsem\b|\bfora\b|\bnunca\b|\bnenhum"


def _quem(chave: Optional[str], prep: str = "com") -> str:
    """Quem fez a venda/visita, conforme o manual (6.1 e 6.2)."""
    pessoa = {
        None: "algum vendedor do grupo",
        "global": "qualquer vendedor da empresa",
        "da empresa": "qualquer vendedor da empresa",
        "a empresa": "qualquer vendedor da empresa",
        "do vendedor": "o próprio vendedor da carteira",
        "o vendedor": "o próprio vendedor da carteira",
    }[chave]
    if prep == "de" and pessoa.startswith("o "):
        return "do " + pessoa[2:]
    return f"{prep} {pessoa}"


def _chave_quem(chave: Optional[str]) -> tuple[str, ...]:
    if chave in ("global", "da empresa", "a empresa"):
        return (r"empresa|qualquer vendedor",)
    if chave in ("do vendedor", "o vendedor"):
        return (r"vendedor",)
    return ()


_EVENTO = {
    "positivacao": ("compra (positivação)", "com", r"positiv|compr"),
    "visita": ("visita", "de", r"visit"),
    "venda": ("venda", "com", r"vend|compr"),
}


@_regra(r"^(?P<fora>fora da )?recencia de (?P<ev>positivacao|visita|venda)"
        r"(?: (?P<quem>global|do vendedor))? (?P<t>.+)$")
def _recencia(m, _o):
    tempo = _tempo(m["t"])
    if tempo is None:
        return None
    evento, prep, chave = _EVENTO[m["ev"]]
    verbo = "não ocorreu" if m["fora"] else "ocorreu"
    chaves = (chave,) + ((_NEG,) if m["fora"] else ()) + _chave_quem(m["quem"])
    return Traducao(f"cuja última {evento} {_quem(m['quem'], prep)} {verbo} {tempo}", chaves=chaves)


@_regra(r"^(?P<neg>in)?ativos(?: para (?P<quem>o vendedor|a empresa))?$")
def _ativos(m, _o):
    estado = "inativos" if m["neg"] else "ativos"
    if m["quem"] == "o vendedor":
        detalhe = ("sem compra com o próprio vendedor há tempo suficiente para serem inativos"
                   if m["neg"] else "segundo o histórico de compras com o próprio vendedor")
    elif m["quem"] == "a empresa":
        detalhe = ("sem compra com nenhum vendedor da empresa" if m["neg"]
                   else "segundo o histórico de compras com qualquer vendedor da empresa")
    else:
        detalhe = "pelos critérios padrão do sistema"
    alvo = f" para {m['quem']}" if m["quem"] else ""
    chave = r"inativ" if m["neg"] else r"(?<!in)ativ"
    return Traducao(f"que estão {estado}{alvo} ({detalhe})",
                    chaves=(chave,) + _chave_quem(m["quem"]))


@_regra(rf"^inatividade(?P<g> global)? (?P<op>de|maior que) (?P<n>\d+) {_U}$")
def _inatividade(m, _o):
    quanto = "há pelo menos" if m["op"] == "de" else "há mais de"
    quem = " com nenhum vendedor da empresa" if m["g"] else ""
    return Traducao(f"sem compra{quem} {quanto} {_qtd(m['n'], m[4])}",
                    chaves=(r"compr|inativ|positiv",) + (_chave_quem("global") if m["g"] else ()))


@_regra(r"^atividade(?: (?P<quem>global|do vendedor))? de (?P<t>.+)$")
def _atividade(m, _o):
    tempo = _tempo(m["t"])
    if tempo is None:
        return None
    return Traducao(f"que compraram {_quem(m['quem'], 'de')} {tempo}",
                    chaves=(r"compr|positiv|ativ",) + _chave_quem(m["quem"]))


@_regra(r"^(?P<k>sem vendas?|prospects?)(?: (?P<quem>do vendedor|da empresa))?$")
def _sem_venda(m, _o):
    rotulo = "sem venda" if m["k"].startswith("sem") else "prospects"
    if m["quem"] == "do vendedor":
        detalhe = "nunca compraram com o próprio vendedor"
    elif m["quem"] == "da empresa":
        detalhe = "nunca compraram com nenhum vendedor da empresa"
    else:
        detalhe = "nunca compraram"
    alvo = f" {m['quem']}" if m["quem"] else ""
    return Traducao(f"{rotulo}{alvo} (que {detalhe})",
                    chaves=(r"sem venda|nunca compr|prospect",) + _chave_quem(m["quem"]))


@_regra(r"^unica compra com (?P<quem>a empresa|o vendedor)$")
def _unica_compra(m, _o):
    return Traducao(f"que compraram uma única vez {_quem(m['quem'])}",
                    chaves=(r"unica|uma vez|so uma",) + _chave_quem(m["quem"]))


@_regra(r"^(?P<neg>nao )?tem (?P<ev>vendas?|visitas?)(?: (?P<quem>do vendedor|da empresa))? (?P<t>.+)$")
def _tem_evento(m, _o):
    tempo = _tempo(m["t"])
    if tempo is None:
        return None
    evento = "venda" if m["ev"].startswith("venda") else "visita"
    neg = "não " if m["neg"] else ""
    chave = r"vend|compr" if evento == "venda" else r"visit"
    return Traducao(f"que {neg}têm {evento} {_quem(m['quem'], 'de')} {tempo}",
                    chaves=(chave,) + ((_NEG,) if m["neg"] else ()) + _chave_quem(m["quem"]))


@_regra(r"^positivado (?P<t>.+)$")
def _positivado(m, _o):
    tempo = _tempo(m["t"])
    if tempo is None:
        return None
    return Traducao(f"que compraram (foram positivados) {tempo}", chaves=(r"positiv|compr",))


@_regra(r"^adimplentes$")
def _adimplentes(_m, _o):
    return Traducao("que estão adimplentes (sem pendências financeiras em aberto)",
                    chaves=(r"(?<!in)adimpl|sem pendencia",))


@_regra(r"^inadimplentes$")
def _inadimplentes(_m, _o):
    return Traducao("que estão inadimplentes (com pendências financeiras em aberto)",
                    chaves=(r"inadimpl|com pendencia",))


@_regra(r"^(?P<neg>nao )?tem credito$")
def _tem_credito(m, _o):
    neg = "não " if m["neg"] else ""
    return Traducao(f"que {neg}têm crédito disponível",
                    chaves=(r"credito",) + ((_NEG,) if m["neg"] else ()))


@_regra(r"^com (?P<n>\d+)% de credito$")
def _pct_credito(m, _o):
    return Traducao(f"com pelo menos {m['n']}% do limite de crédito disponível", chaves=(r"credito",))


@_regra(r"^com menos de r\$ ?(?P<v>\d[\d.,]*) de credito disponivel$")
def _menos_credito(m, _o):
    return Traducao(f"com menos de R$ {m['v']} de crédito disponível",
                    chaves=(r"credito", r"menos|abaixo|inferior"))


@_regra(r"^com (?P<lim>limite de )?r\$ ?(?P<v>\d[\d.,]*) de credito$")
def _valor_credito(m, _o):
    oque = "limite de crédito" if m["lim"] else "crédito"
    return Traducao(f"com {oque} de pelo menos R$ {m['v']}", chaves=(r"credito",))


def _comparacao(texto: str) -> Optional[str]:
    partes = re.split(r" e (?=maior|menor|igual)", texto)
    saida = []
    for parte in partes:
        m = re.fullmatch(r"(maior que|menor que|igual a) (r\$ ?)?(\d[\d.,]*)", parte)
        if not m:
            return None
        operador = {"maior que": "acima de", "menor que": "abaixo de", "igual a": "igual a"}[m[1]]
        saida.append(f"{operador} {'R$ ' if m[2] else ''}{m[3]}")
    return " e ".join(saida)


_INDEFINIDO = " (ou sem valor calculado)"
_CHAVE_INDEFINIDO = r"indefinid|sem valor|nao calculad|desconhecid|nao informad"
_PERIODO = {"mes": "mês", "bimestre": "bimestre", "trimestre": "trimestre",
            "semestre": "semestre", "ano": "ano"}


@_regra(r"^com ticket medio (?P<c>.+?)(?P<ind> ou indefinido)?$")
def _ticket(m, _o):
    comparacao = _comparacao(m["c"])
    if comparacao is None:
        return None
    return Traducao(f"com ticket médio {comparacao}{_INDEFINIDO if m['ind'] else ''}",
                    chaves=(r"ticket",) + ((_CHAVE_INDEFINIDO,) if m["ind"] else ()))


@_regra(r"^(?P<k>total faturado|media faturada) no (?P<p>mes|bimestre|trimestre|semestre|ano) "
        r"com a empresa (?P<c>.+?)(?P<ind> ou indefinido)?$")
def _faturamento(m, _o):
    comparacao = _comparacao(m["c"])
    if comparacao is None:
        return None
    tipo = "faturamento total" if m["k"].startswith("total") else "faturamento médio mensal"
    return Traducao(
        f"com {tipo} no {_PERIODO[m['p']]} com a empresa {comparacao}{_INDEFINIDO if m['ind'] else ''}",
        chaves=(r"fatur",) + ((r"medi",) if m["k"].startswith("media") else ())
        + ((_CHAVE_INDEFINIDO,) if m["ind"] else ()))


@_regra(r"^queda de faturamento (?P<op>maior|menor) que (?P<n>\d+)%$")
def _queda(m, _o):
    quanto = "acima de" if m["op"] == "maior" else "de até"
    return Traducao(f"com queda de faturamento {quanto} {m['n']}%", chaves=(r"queda|cai|reduc",))


@_regra(r"^total de compras com (?P<quem>o vendedor|a empresa) (?P<c>.+?)(?P<ind> ou indefinido)?$")
def _total_compras(m, _o):
    comparacao = _comparacao(m["c"])
    if comparacao is None:
        return None
    return Traducao(
        f"com total de compras {_quem(m['quem'])} {comparacao}{_INDEFINIDO if m['ind'] else ''}",
        chaves=(r"compra",) + ((_CHAVE_INDEFINIDO,) if m["ind"] else ()))


_CAMPOS = {
    "ramo de atividade": ("ramo de atividade", r"ramo"),
    "nome da cidade": ("cidade", r"cidade|municipio"),
    "nome do bairro": ("bairro", r"bairro"),
    "uf do estado": ("UF", r"\buf\b|estado"),
    "codigo da rota": ("código de rota", r"rota"),
    "nome da rota": ("nome de rota", r"rota"),
    "o rotulo": ("rótulo de classificação", r"rotulo"),
    "rotulo": ("rótulo de classificação", r"rotulo"),
    "cnae": ("CNAE (mapeado pelo Govendas)", r"cnae"),
    "cnae da receita": ("CNAE da Receita Federal", r"cnae"),
    "cnae do erp": ("CNAE do ERP", r"cnae"),
    "nome do cnae da receita": ("nome do CNAE da Receita Federal", r"cnae"),
    "id do cliente": ("id do cliente", r"\bid\b|identificador|codigo"),
    "id do cliente principal": ("id do cliente principal", r"principal"),
    "tipo de cobranca": ("tipo de cobrança", r"cobranca"),
    "tipo de pagamento": ("tipo de pagamento", r"pagamento"),
    "conteudo da nota": ("conteúdo da nota", r"nota"),
    "nome da nota": ("nome da nota", r"nota"),
}
_CAMPOS_RX = "|".join(re.escape(c) for c in sorted(_CAMPOS, key=len, reverse=True))


@_regra(rf"^(?P<campo>{_CAMPOS_RX})(?: e)? (?P<op>igual|diferente|contem|inicia com) ?\((?P<a>.*)\)$")
def _campo(m, original):
    valores = valores_entre_aspas(original)
    if not valores:
        return None
    rotulo, chave = _CAMPOS[m["campo"]]
    varios = len(valores) > 1
    operador = {
        "igual": "igual a um destes valores:" if varios else "igual a",
        "diferente": "diferente de todos estes valores:" if varios else "diferente de",
        "contem": "contendo um destes textos:" if varios else "contendo",
        "inicia com": "começando com um destes textos:" if varios else "começando com",
    }[m["op"]]
    chaves = (chave,)
    if m["op"] == "diferente":
        chaves += (r"diferente|exceto|\bnao\b|fora|menos|exclu",)
    return Traducao(f"com {rotulo} {operador} {_lista(valores)}", chaves=chaves)


@_regra(r"^status na receita federal definido como (?P<s>ativa|suspensa|inapta|baixada)$")
def _receita(m, _o):
    return Traducao(f"com situação {m['s']} na Receita Federal", chaves=(r"receita", m["s"][:-1]))


@_regra(r"^pessoa (?P<t>fisica|juridica)$")
def _pessoa(m, _o):
    if m["t"] == "fisica":
        return Traducao("que são pessoa física (CPF)", chaves=(r"fisica|cpf",))
    return Traducao("que são pessoa jurídica (CNPJ)", chaves=(r"juridica|cnpj",))


@_regra(r"^(?:e o |o )?cliente (?P<t>filial|principal)$")
def _filial(m, _o):
    if m["t"] == "filial":
        return Traducao("que são filiais", chaves=(r"filia",))
    return Traducao("que são o cadastro principal (não filial)", chaves=(r"principal",))


_FLAGS = {
    "bloqueados": ("bloqueados para atendimento", r"bloquead"),
    "excluidos": ("marcados como excluídos no sistema", r"exclui"),
    "agendavel": ("marcados como agendáveis", r"agendav"),
    "reagendado": ("reagendados para algum vendedor do grupo", r"reagend"),
}


@_regra(r"^(?P<neg>nao )?(?P<f>bloqueados|excluidos|agendavel|reagendado)$")
def _flag(m, _o):
    frase, chave = _FLAGS[m["f"]]
    if m["neg"]:
        return Traducao(f"que não estão {frase}", chaves=(chave, _NEG))
    return Traducao(f"que estão {frase}", chaves=(chave,))


@_regra(r"^elegivel(?P<r> incluindo roteirizavel)?$")
def _elegivel(m, _o):
    extra = " e para roteirizar" if m["r"] else ""
    return Traducao(f"elegíveis (não excluídos, ativos e marcados para agendar{extra})",
                    chaves=(r"elegiv",))


@_regra(r"^(?P<neg>nao )?tem coordenada confiavel$")
def _coordenada(m, _o):
    if m["neg"]:
        return Traducao("sem geolocalização confiável", chaves=(r"geoloc|coordenada|localiza", _NEG))
    return Traducao("com geolocalização confiável", chaves=(r"geoloc|coordenada|localiza",))


@_regra(r"^(?P<neg>nao )?tem parametro(?P<man> manual)?$")
def _tem_parametro(m, _o):
    tipo = "parametrização manual" if m["man"] else "parametrização"
    if m["neg"]:
        return Traducao(f"que ainda não têm {tipo} configurada", chaves=(r"parametriz", _NEG))
    return Traducao(f"que já têm {tipo} configurada", chaves=(r"parametriz",))


@_regra(r"^curva (?P<c>[abc]) com (?P<quem>a empresa|o vendedor)$")
def _curva(m, _o):
    return Traducao(f"da curva {m['c'].upper()} de faturamento {_quem(m['quem'])}",
                    chaves=(rf"curva {m['c']}\b",) + _chave_quem(m["quem"]))


_CICLO = {
    "prospect": " (ainda não compraram)",
    "ativado": " (fizeram a primeira compra quando eram prospects)",
    "reativado": " (voltaram a comprar depois de ficarem inativos)",
    "ativo": "", "inativo": "",
}


@_regra(r"^no ciclo de vida (?P<e>prospect|ativo|inativo|ativado|reativado) com (?P<quem>a empresa|o vendedor)$")
def _ciclo(m, _o):
    alvo = "a empresa" if m["quem"] == "a empresa" else "o vendedor"
    return Traducao(f"que estão no ciclo de vida {m['e']} com {alvo}{_CICLO[m['e']]}",
                    chaves=(r"ciclo de vida", rf"\b{m['e']}") + _chave_quem(m["quem"]))


@_regra(r"^frequencia historica (?P<op>maior que|menor que|de|entre) (?P<r>.+?)(?P<ind> ou indefinida)?$")
def _freq_historica(m, _o):
    resto = m["r"]
    if m["op"] == "entre":
        f = re.fullmatch(rf"(\d+) {_U} e (\d+) {_U}", resto)
        if not f:
            return None
        valor = f"está entre {_qtd(f[1], f[2])} e {_qtd(f[3], f[4])}"
    else:
        f = re.fullmatch(rf"(\d+) {_U}", resto)
        if not f:
            return None
        valor = {"maior que": "é maior que", "menor que": "é menor que", "de": "é de"}[m["op"]]
        valor += f" {_qtd(f[1], f[2])}"
    extra = " (ou ainda não pôde ser calculada)" if m["ind"] else ""
    return Traducao(f"cuja frequência histórica de compra {valor}{extra}",
                    chaves=(r"historic",) + ((_CHAVE_INDEFINIDO,) if m["ind"] else ()))


@_regra(r"^(?P<neg>nao )?contido na segmentacao ?\((?P<a>.*)\)$")
def _contido(m, original):
    valores = valores_entre_aspas(original)
    if not valores:
        return None
    alvo = "à segmentação" if len(valores) == 1 else "às segmentações"
    if m["neg"]:
        return Traducao(f"que não pertencem {alvo} {_lista(valores)}", chaves=(r"segment", _NEG))
    return Traducao(f"que pertencem {alvo} {_lista(valores)}", chaves=(r"segment",))


@_regra(rf"^cadastrado (?:faz|ha) (?P<n>\d+) {_U}$")
def _cadastrado_ha(m, _o):
    return Traducao(f"cadastrados no ERP há pelo menos {_qtd(m['n'], m[2])}", chaves=(r"cadastr",))


@_regra(r"^cadastrado a partir do dia '(?P<d>[^']+)'$")
def _cadastrado_desde(m, _o):
    return Traducao(f"cadastrados a partir de {m['d']}", chaves=(r"cadastr",))


@_regra(r"^data da rota(?P<fut> futura)?(?P<v> do vendedor)? (?P<t>.+)$")
def _data_rota(m, _o):
    if m["fut"] and (f := re.fullmatch(rf"em (\d+) {_U}", m["t"])):
        tempo = f"daqui a {_qtd(f[1], f[2])}"
    else:
        tempo = _tempo(m["t"])
    if tempo is None:
        return None
    qual = " futura" if m["fut"] else ""
    dono = " (rota do vendedor da carteira)" if m["v"] else ""
    return Traducao(f"cuja data de rota{qual}{dono} cai {tempo}", chaves=(r"rota",))


@_regra(r"^data da (?P<q>proxima|ultima) visita do representante (?P<t>.+)$")
def _visita_representante(m, _o):
    if m["q"] == "proxima" and (f := re.fullmatch(rf"em (\d+) {_U}", m["t"])):
        tempo = f"daqui a {_qtd(f[1], f[2])}"
    else:
        tempo = _tempo(m["t"])
    if tempo is None:
        return None
    qual, verbo = ("próxima", "é") if m["q"] == "proxima" else ("última", "foi")
    return Traducao(f"cuja {qual} visita do representante externo (registrada no ERP) {verbo} {tempo}",
                    chaves=(r"representante|externo", r"visita"))


_RUPTURA = {"bimestre": "sem compra entre 30 e 60 dias", "trimestre": "sem compra entre 61 e 90 dias",
            "semestre": "sem compra entre 91 e 180 dias", "ano": "sem compra entre 181 e 365 dias"}


@_regra(r"^em ruptura global do (?P<p>bimestre|trimestre|semestre|ano)$")
def _ruptura(m, _o):
    return Traducao(f"em ruptura global do {m['p']} ({_RUPTURA[m['p']]})", chaves=(r"ruptura",))


def _uma_condicao(texto: str) -> Optional[Traducao]:
    normalizado = normalizar(texto)
    for rx, fn in _REGRAS:
        if m := rx.match(normalizado):
            traducao = fn(m, texto)
            if traducao is not None:
                return traducao
    return None


def _partes(texto: str) -> tuple[list[str], list[str]]:
    """Divide em ` e `/` ou ` que estejam fora de parênteses e aspas."""
    partes, conectores = [], []
    nivel, aspas, inicio, i = 0, None, 0, 0
    while i < len(texto):
        c = texto[i]
        if aspas:
            aspas = None if c == aspas else aspas
        elif c in "\"'":
            aspas = c
        elif c == "(":
            nivel += 1
        elif c == ")":
            nivel = max(0, nivel - 1)
        elif nivel == 0 and c.isspace():
            m = re.match(r"\s+(e|ou)\s+", texto[i:], re.IGNORECASE)
            if m:
                partes.append(texto[inicio:i])
                conectores.append(m[1].lower())
                i += m.end()
                inicio = i
                continue
        i += 1
    partes.append(texto[inicio:])
    return partes, conectores


def conectores_ou(texto: str) -> int:
    """Quantos `ou` lógicos há numa condição (fora de aspas e parênteses)."""
    return _partes(texto)[1].count("ou")


def traduzir_condicao(texto: str) -> Traducao:
    """Uma condição `quando` (pode ter `e`/`ou` internos) em português."""
    texto = texto.strip()
    inteira = _uma_condicao(texto)
    if inteira is not None:
        return inteira
    partes, conectores = _partes(texto)
    memo: dict[int, Optional[Traducao]] = {}

    def cobrir(i: int) -> Optional[Traducao]:
        # Tradução de partes[i:], preferindo grupos maiores
        # (`entre 30 dias e 60 dias` não deve ser quebrado).
        if i in memo:
            return memo[i]
        memo[i] = None
        for j in range(len(partes), i, -1):
            grupo = partes[i]
            for k in range(i + 1, j):
                grupo += f" {conectores[k - 1]} {partes[k]}"
            primeira = _uma_condicao(grupo)
            if primeira is None:
                continue
            if j == len(partes):
                memo[i] = primeira
                break
            resto = cobrir(j)
            if resto is not None:
                memo[i] = Traducao(f"{primeira.frase} {conectores[j - 1]} {resto.frase}",
                                   chaves=primeira.chaves + resto.chaves)
                break
        return memo[i]

    composta = cobrir(0) if len(partes) > 1 else None
    if composta is not None:
        return composta
    return _desconhecida(texto, f'que atendem à condição "{texto}"')


# ---------------------------------------------------------------------------
# Frequência (manual 7.4)
# ---------------------------------------------------------------------------

_PERIODICA = {
    "diaria": ("diária", 1, r"diari|todo dia|todos os dias|1 dia"),
    "semanal": ("semanal", 7, r"seman|7 dias"),
    "quinzenal": ("quinzenal", 15, r"quinzen|15 dias"),
    "mensal": ("mensal", 30, r"mens|30 dias"),
    "bimestral": ("bimestral", 60, r"bimestr|60 dias"),
    "trimestral": ("trimestral", 90, r"trimestr|90 dias"),
    "semestral": ("semestral", 180, r"semestr|180 dias"),
    "anual": ("anual", 365, r"anual|365 dias"),
}


def _periodica(texto: str) -> Optional[Traducao]:
    m = re.fullmatch(r"(diaria|semanal|quinzenal|mensal|bimestral|trimestral|semestral|anual)"
                     r"\(\s*(?:'([^']*)')?\s*\)", texto)
    if not m:
        return None
    nome, dias, chave = _PERIODICA[m[1]]
    pivo = f", contando a partir de {m[2]}" if m[2] else ""
    return Traducao(f"com frequência {nome} (a cada {dias} dias{pivo})", chaves=(chave,))


def _fonte_frequencia(texto: str) -> Optional[Traducao]:
    if m := re.fullmatch(r"(?:de )?(\d+) dias?", texto):
        return Traducao(f"a cada {_qtd(m[1], 'dias')}")
    if t := _periodica(texto):
        return t
    if m := re.fullmatch(r"do (vendedor|historico)(?: (mais|menos) (\d+) dias?)?", texto):
        if m[1] == "vendedor":
            frase, chave = "com a frequência que o próprio vendedor configurou para o cliente", r"vendedor"
        else:
            frase, chave = "com a frequência calculada pelo histórico de visitas do cliente", r"historic"
        if m[2]:
            ajuste = "acrescida de" if m[2] == "mais" else "reduzida em"
            frase += f", {ajuste} {_qtd(m[3], 'dias')}"
            chave_ajuste = r"acresc|mais|soma|adicion" if m[2] == "mais" else r"reduz|menos|subtra|diminu"
            return Traducao(frase, chaves=(chave, chave_ajuste))
        return Traducao(frase, chaves=(chave,))
    if texto == "de qualquer vendedor":
        return Traducao("com a frequência configurada por qualquer vendedor para o cliente",
                        chaves=(r"qualquer vendedor|outro vendedor|algum vendedor",))
    if re.fullmatch(r"(?:do |de )?govendas", texto):
        return Traducao("com a frequência padrão do Govendas", chaves=(r"govendas|padrao",))
    if m := re.fullmatch(r"(?:de )?(ativos|inativos) (da empresa|do vendedor)", texto):
        return Traducao(f"com a frequência média dos clientes {m[1]} {m[2]}",
                        chaves=(r"medi", rf"\b{m[1]}\b", r"empresa" if m[2] == "da empresa" else r"vendedor"))
    if texto == "remover":
        return Traducao("sem frequência definida (a frequência configurada é removida)",
                        chaves=(r"remov|retir|sem frequencia|apag",))
    return None


def _limites(texto: str) -> Optional[Traducao]:
    partes = re.split(r" e (?=no (?:maximo|minimo))", texto)
    frases, chaves = [], []
    for parte in partes:
        m = re.fullmatch(r"no (maximo|minimo) (?:de )?(.+)", parte)
        if not m:
            return None
        if d := re.fullmatch(r"(\d+) dias?", m[2]):
            valor = _qtd(d[1], "dias")
        elif p := _periodica(m[2]):
            valor = f"{_PERIODICA[m[2].split('(')[0]][1]} dias"
        else:
            return None
        frases.append(f"no {'máximo' if m[1] == 'maximo' else 'mínimo'} {valor}")
        chaves.append(r"maxim|ate|no maximo|limit" if m[1] == "maximo" else r"minim|pelo menos")
    return Traducao(" e ".join(frases), chaves=tuple(chaves))


def traduzir_frequencia(valor: str) -> Traducao:
    """Complemento de "devem ser atendidos ..." para o valor de `frequencia`."""
    normalizado = normalizar(valor.replace("\n", " "))
    alternativas = [a.strip() for a in re.split(r"\s+ou\s+", normalizado)]
    fontes: list[Traducao] = []
    limites: list[Traducao] = []
    for alternativa in alternativas:
        if alternativa.startswith(("no maximo", "no minimo")):
            limite = _limites(alternativa)
            if limite is None:
                return _desconhecida(valor, f'com a frequência "{valor.strip()}"')
            limites.append(limite)
            continue
        fonte = _fonte_frequencia(alternativa)
        if fonte is None:
            return _desconhecida(valor, f'com a frequência "{valor.strip()}"')
        fontes.append(fonte)
    if not fontes:
        return _desconhecida(valor, f'com a frequência "{valor.strip()}"')
    frase = fontes[0].frase
    for indice, fonte in enumerate(fontes[1:]):
        frase += f"; se {'ela' if indice == 0 else 'também'} não existir, {fonte.frase}"
    if limites:
        intervalo = " e ".join(l.frase for l in limites)
        frase += f" (intervalo de {intervalo} entre atendimentos)"
    chaves = tuple(c for t in fontes + limites for c in t.chaves)
    return Traducao(frase, chaves=chaves)


# ---------------------------------------------------------------------------
# Preferência e outros modificadores (manual 7.2, 7.3 e 7.5)
# ---------------------------------------------------------------------------

_DIAS = {
    "segunda": ("segunda-feira", r"segunda"), "terca": ("terça-feira", r"terca"),
    "quarta": ("quarta-feira", r"quarta"), "quinta": ("quinta-feira", r"quinta"),
    "sexta": ("sexta-feira", r"sexta"), "sabado": ("sábado", r"sabado"),
    "domingo": ("domingo", r"domingo"),
    "dias uteis": ("dias úteis (segunda a sexta)", r"uteis|segunda a sexta"),
    "fim de semana": ("fim de semana (sábado e domingo)", r"fim de semana|sabado e domingo"),
}
_INTENSIDADE = {"maior": r"maior|prioritari", "menor": r"menor|baixa"}


def _preferencia_dias(texto: str) -> Optional[Traducao]:
    itens = [i.strip() for i in re.split(r"\s+e\s+", texto) if i.strip()]
    dias, chaves, intensidades = [], [], []
    for item in itens:
        m = re.fullmatch(r"(?:(maior|menor|normal) )?(?:na|no|nos|em) (segunda|terca|quarta|quinta|sexta|sabado"
                         r"|domingo|dias uteis|fim de semana)(?:-feira)?", item)
        if not m:
            return None
        nome, chave = _DIAS[m[2]]
        artigo = "nos" if m[2] == "dias uteis" else ("no" if m[2] in ("sabado", "domingo", "fim de semana") else "na")
        dias.append((m[1], f"{artigo} {nome}"))
        chaves.append(chave)
        if m[1] and m[1] != "normal":
            intensidades.append(m[1])
    if intensidades:
        partes = [f"{i or 'normal'} {d}" for i, d in dias]
        chaves += [_INTENSIDADE[i] for i in sorted(set(intensidades))]
    else:
        partes = [d for _, d in dias]
    if len(partes) == 1:
        lista = partes[0]
    else:
        lista = ", ".join(partes[:-1]) + " e " + partes[-1]
    return Traducao(lista, chaves=tuple(chaves))


def _preferencia(corpo: str) -> Optional[Traducao]:
    alternativas = [a.strip() for a in re.split(r"\s+ou\s+", corpo)]
    frases, chaves = [], [r"prefer|prioridade|prioriz"]
    so_dias = True
    for alternativa in alternativas:
        if alternativa == "do vendedor":
            frases.append("a preferência de dias que o próprio vendedor configurou")
            chaves.append(r"vendedor")
            so_dias = False
        elif alternativa == "do historico":
            frases.append("a preferência de dias calculada pelo histórico de visitas")
            chaves.append(r"historic")
            so_dias = False
        elif alternativa == "remover":
            frases.append("nenhuma preferência (a configuração de dias é removida)")
            chaves.append(r"remov|retir|sem preferencia")
            so_dias = False
        else:
            dias = _preferencia_dias(alternativa)
            if dias is None:
                return None
            frases.append(dias.frase)
            chaves.extend(dias.chaves)
    if so_dias and len(frases) == 1:
        frase = (f"Preferência de visita {frases[0]}; nos demais dias o cliente não é visitado.")
    else:
        frase = "Preferência de dias de visita: " + frases[0]
        for indice, alternativa in enumerate(frases[1:]):
            frase += f"; se {'ela' if indice == 0 else 'também'} não existir, {alternativa}"
        frase += "."
    return Traducao(frase, chaves=tuple(chaves))


_FILTROS_CHOQUE = {"na rota": "considerando apenas clientes roteirizáveis",
                   "fora da rota": "considerando apenas clientes não roteirizáveis",
                   "na pasta": "considerando apenas clientes da mesma pasta"}
_TIPO_ESCOPO = {"time": "do time", "carteira": "da carteira", "segmentacao": "da segmentação"}


def traduzir_modificador(texto: str) -> Traducao:
    """Uma frase completa para `preferencia`, `fixo`, `roteirizavel`, `sem choque` ou `restrito`."""
    normalizado = normalizar(texto.replace("\n", " ")).rstrip('"')
    if m := re.fullmatch(r"preferencia (.+)", normalizado):
        traducao = _preferencia(m[1])
        if traducao is not None:
            return traducao
    elif normalizado == "fixo":
        return Traducao("A parametrização é fixa: o gerador de agendas segue a frequência e a "
                        "preferência de forma mais rígida.", chaves=(r"\bfix",))
    elif normalizado == "roteirizavel":
        return Traducao("O cliente entra na roteirização (otimização da rota de visita presencial).",
                        chaves=(r"roteiriz|rota",))
    elif m := re.fullmatch(r"sem choque(?P<f>(?: (?:e )?(?:na rota|fora da rota|na pasta))*)"
                           r" (?:no|na|nas|nos|com|do|da|com o|com a) ?(?P<tipo>time|carteira|segmentacao)"
                           r"\s*\((?P<a>.*)\)", normalizado):
        valores = valores_entre_aspas(texto)
        if valores:
            filtros = [_FILTROS_CHOQUE[f] for f in re.findall(r"na rota|fora da rota|na pasta", m["f"] or "")]
            extra = f" ({', '.join(filtros)})" if filtros else ""
            return Traducao(
                f"A preferência de dias não pode coincidir com a {_TIPO_ESCOPO[m['tipo']]} "
                f"{_lista(valores)}{extra}, para que o cliente não receba dois atendimentos no mesmo dia.",
                chaves=(r"choque|coincid|mesmo dia|conflit|sobrepo",))
    elif m := re.fullmatch(r"restrito (a partir da|ate a) (primeira|segunda|terceira|quarta) semana", normalizado):
        quando = "a partir da" if m[1] == "a partir da" else "até a"
        return Traducao(f"O cliente só pode entrar na agenda {quando} {m[2]} semana do mês.",
                        chaves=(rf"{m[2]} semana", r"a partir|desde" if m[1] == "a partir da" else r"\bate\b"))
    return _desconhecida(texto, f'Modificador adicional: "{texto.strip()}".')


# ---------------------------------------------------------------------------
# Filtros temporais de execução (manual cap. 5)
# ---------------------------------------------------------------------------

_SEMANA = {
    "segunda": "às segundas-feiras", "terca": "às terças-feiras", "quarta": "às quartas-feiras",
    "quinta": "às quintas-feiras", "sexta": "às sextas-feiras", "sabado": "aos sábados",
    "domingo": "aos domingos",
}
_SEMANA_DO_MES = {
    "primeira semana": "na primeira semana do mês (dias 1 a 7)",
    "segunda semana": "na segunda semana do mês (dias 8 a 14)",
    "terceira semana": "na terceira semana do mês (dias 15 a 21)",
    "quarta semana": "na quarta semana do mês (dias 22 a 28)",
    "ultima semana": "na última semana do mês (últimos 7 dias)",
    "primeira quinzena": "na primeira quinzena do mês (dias 1 a 15)",
    "segunda quinzena": "na segunda quinzena do mês (do dia 16 ao fim do mês)",
}
_PERIODOS_ANO = {
    "primeiro bimestre": "janeiro e fevereiro", "segundo bimestre": "março e abril",
    "terceiro bimestre": "maio e junho", "quarto bimestre": "julho e agosto",
    "quinto bimestre": "setembro e outubro", "sexto bimestre": "novembro e dezembro",
    "primeiro trimestre": "janeiro a março", "segundo trimestre": "abril a junho",
    "terceiro trimestre": "julho a setembro", "quarto trimestre": "outubro a dezembro",
    "primeiro semestre": "janeiro a junho", "segundo semestre": "julho a dezembro",
}
_MESES = ("janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto",
          "setembro", "outubro", "novembro", "dezembro")
CHAVE_RESTRICAO = r"\bso\b|somente|apenas|exclusivamente|aplicad|execut|restrit"


def _um_filtro(f: str) -> Optional[Traducao]:
    if m := re.fullmatch(r"no dia (\d+)", f):
        return Traducao(f"no dia {m[1]} do mês")
    if f == "no ultimo dia do mes":
        return Traducao("no último dia do mês", chaves=(r"ultimo dia",))
    if m := re.fullmatch(r"(antes|depois|a partir) do dia (\d+)", f):
        return Traducao(f"{m[1]} do dia {m[2]} do mês",
                        chaves=({"antes": r"antes|ate", "depois": r"depois|apos", "a partir": r"a partir|apos|desde"}[m[1]],))
    if m := re.fullmatch(r"entre os dias (\d+) e (\d+)", f):
        return Traducao(f"entre os dias {m[1]} e {m[2]} do mês")
    if m := re.fullmatch(r"(?:na|no) (segunda|terca|quarta|quinta|sexta|sabado|domingo)(?:-feira)?", f):
        return Traducao(_SEMANA[m[1]], chaves=(m[1],))
    if f == "todos os dias":
        return Traducao("todos os dias", chaves=(r"todos os dias|todo dia",))
    if f == "nos dias uteis":
        return Traducao("em dias úteis (segunda a sexta)", chaves=(r"uteis|segunda a sexta",))
    if f == "no fim de semana":
        return Traducao("no fim de semana (sábado e domingo)", chaves=(r"fim de semana|sabado",))
    if m := re.fullmatch(r"na (primeira|segunda|terceira|quarta|ultima) (semana|quinzena)", f):
        chave = f"{m[1]} {m[2]}"
        return Traducao(_SEMANA_DO_MES[chave], chaves=(chave,))
    if m := re.fullmatch(r"no (primeiro|segundo|terceiro|quarto|quinto|sexto) (bimestre|trimestre|semestre)", f):
        chave = f"{m[1]} {m[2]}"
        if chave in _PERIODOS_ANO:
            return Traducao(f"no {chave} ({_PERIODOS_ANO[chave]})", chaves=(chave,))
    if m := re.fullmatch(r"em (\w+)", f):
        if m[1] in _MESES:
            return Traducao(f"em {'março' if m[1] == 'marco' else m[1]}", chaves=(m[1],))
    return None


def traduzir_filtro(texto: str) -> Traducao:
    """Complemento de "só é aplicada quando a regra executa ..." para um filtro temporal."""
    normalizado = normalizar(texto.replace("\n", " ")).rstrip(" e")
    partes = [p for p in re.split(r"\s+e\s+", normalizado) if p]
    traducoes = [_um_filtro(p) for p in partes]
    if not traducoes or any(t is None for t in traducoes):
        return _desconhecida(texto, f'"{texto.strip()}"')
    frase = " e ".join(t.frase for t in traducoes)
    chaves = (CHAVE_RESTRICAO,) + tuple(c for t in traducoes for c in t.chaves)
    return Traducao(frase, chaves=chaves)
