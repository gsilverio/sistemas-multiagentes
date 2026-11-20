"""Padrões sintáticos da linguagem GOBR (gorules 1.0) extraídos do manual.

Fonte: `data/manual/govendas-rules.pdf`. Cada padrão indica o capítulo do
manual de onde veio. Todos os padrões operam sobre texto já normalizado por
`normalizar` (minúsculas, sem acentos e espaços colapsados), para que
`frequência` e `frequencia` sejam tratados da mesma forma.

Este módulo não interpreta regras. Ele apenas responde se um trecho tem a
forma esperada; o parser (`src.gobr.parser`) e o validador
(`src.validation.gobr_validator`) é que decidem o que fazer com a resposta.
"""

from __future__ import annotations

import re
import unicodedata


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e com espaços colapsados."""
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip()


# ---------------------------------------------------------------------------
# Estrutura do arquivo (cap. 2 e cap. 10)
# ---------------------------------------------------------------------------

VERSION = re.compile(r"^version\s+1\.0\s*(?:#.*)?$", re.IGNORECASE)

# Cabeçalho de bloco de nível zero: `<categoria> "nome":`.
CABECALHO = re.compile(
    r'^(?P<categoria>time|segmentacao|carteira\s+livre|carteira|'
    r'limpar\s+parametros|parametros|alertas?|tags?)\s+'
    r'"(?P<nome>.*)"\s*:\s*(?:#.*)?$',
    re.IGNORECASE,
)

# Ordem obrigatória das categorias no arquivo (cap. 10, regra `gorules`).
ORDEM_CATEGORIAS = {
    "time": 1,
    "segmentacao": 2,
    "carteira": 3,
    "parametros": 4,
}

# Categorias que existem na linguagem mas que o agente não gera.
CATEGORIAS_NAO_SUPORTADAS = {"alerta", "alertas", "tag", "tags"}


def categoria_canonica(categoria: str) -> str:
    """`carteira livre` e `carteira` compartilham nomes e posição na ordem."""
    categoria = normalizar(categoria)
    if categoria == "carteira livre":
        return "carteira"
    if categoria == "limpar parametros":
        return "parametros"
    return categoria


# ---------------------------------------------------------------------------
# Corpo de `time` (cap. 3)
# ---------------------------------------------------------------------------

_LISTA = r'\(\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\)'
_REFINAMENTO_TIME = (
    rf"(?:dos\s+supervisores\s*{_LISTA}|do\s+supervisor\s*{_LISTA}"
    rf"|ativados|inativados|menos\s*{_LISTA})"
)
TIME_VENDEDORES = re.compile(
    rf"^vendedores\s+(?:todos|todostelevendas|todosdecampo|porid\s*{_LISTA})"
    rf"(?:\s+{_REFINAMENTO_TIME})*$"
)
# O manual permite os refinamentos em linhas próprias (exemplo 3.4).
TIME_REFINAMENTO = re.compile(rf"^{_REFINAMENTO_TIME}(?:\s+{_REFINAMENTO_TIME})*$")


# ---------------------------------------------------------------------------
# Corpo de `carteira` e `segmentacao` (cap. 4)
# ---------------------------------------------------------------------------

_FONTE_CLIENTES = (
    rf"(?:carteirafixa|carteiraacessivel|carteiraruptura|portfolio|segmentados|todos)\s*{_LISTA}"
    r'|inativos\s*\(\s*"[^"]*"\s*,\s*[^)]+\)'
    r'|ruptura(?:\s+da\s+segmentacao)?\s*\(\s*"[^"]*"\s+para\s+"[^"]*"\s*\)'
)
# Opções de ruptura (4.6.1), possivelmente combinadas com `e`.
_OPCAO_RUPTURA = r'(?:e\s+)?de\s+(?:positivacao|visita|inativos)\s*\([^)]*\)(?:\s+ou\s+prospects)?'
CARTEIRA_OPCAO_RUPTURA = re.compile(rf"^{_OPCAO_RUPTURA}$")
# As opções de ruptura podem vir na mesma linha de `clientes ruptura(...)`.
CARTEIRA_CLIENTES = re.compile(rf"^clientes\s+(?:{_FONTE_CLIENTES})(?:\s+{_OPCAO_RUPTURA})*$")

# Funções de seleção que citam outros blocos pelo nome.
REFERENCIAS_EM_DEFINICAO = (
    ("time", re.compile(
        r'(?:carteirafixa|carteiraacessivel|carteiraruptura|portfolio|todos)\s*\((?P<args>[^)]*)\)')),
    ("time", re.compile(r'inativos\s*\(\s*(?P<args>"[^"]*")')),
    ("segmentacao", re.compile(r'segmentados\s*\((?P<args>[^)]*)\)')),
    ("time", re.compile(r'ruptura\s*\(\s*(?P<args>"[^"]*")\s+para')),
    ("segmentacao", re.compile(r'ruptura\s+da\s+segmentacao\s*\(\s*(?P<args>"[^"]*")\s+para')),
    ("time", re.compile(r'para\s+(?P<args>"[^"]*")\s*\)')),
    ("segmentacao", re.compile(r'contido\s+na\s+segmentacao\s*\((?P<args>[^)]*)\)')),
)


# ---------------------------------------------------------------------------
# Filtros temporais (cap. 5)
# ---------------------------------------------------------------------------

_DIA_SEMANA = r"(?:na\s+segunda|na\s+terca|na\s+quarta|na\s+quinta|na\s+sexta|no\s+sabado|no\s+domingo)"
_FILTRO_TEMPORAL_ATOMO = (
    # 5.2 dia do mês
    r"no\s+dia\s+\d{1,2}|no\s+ultimo\s+dia\s+do\s+mes|antes\s+do\s+dia\s+\d{1,2}"
    r"|depois\s+do\s+dia\s+\d{1,2}|a\s+partir\s+do\s+dia\s+\d{1,2}"
    r"|entre\s+os\s+dias\s+\d{1,2}\s+e\s+\d{1,2}"
    # 5.3 dia da semana
    rf"|{_DIA_SEMANA}|todos\s+os\s+dias|nos\s+dias\s+uteis|no\s+fim\s+de\s+semana"
    # 5.4 semana do mês
    r"|na\s+(?:primeira|segunda|terceira|quarta|ultima)\s+semana"
    r"|na\s+(?:primeira|segunda)\s+quinzena"
    # 5.5 mês do ano
    r"|em\s+(?:janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
    r"|no\s+(?:primeiro|segundo|terceiro|quarto|quinto|sexto)\s+bimestre"
    r"|no\s+(?:primeiro|segundo|terceiro|quarto)\s+trimestre"
    r"|no\s+(?:primeiro|segundo)\s+semestre"
)
FILTRO_TEMPORAL = re.compile(
    rf"^(?:{_FILTRO_TEMPORAL_ATOMO})(?:\s+e\s+(?:{_FILTRO_TEMPORAL_ATOMO}))*(?:\s+e)?$"
)


# ---------------------------------------------------------------------------
# Frequência (cap. 7.4)
# ---------------------------------------------------------------------------

_FREQ_FIXA = (
    r"diaria\(\)|semanal\(\)|quinzenal\((?:'[^']*')?\)|mensal\(\)"
    r"|bimestral\((?:'[^']*')?\)|trimestral\(\)|semestral\(\)|anual\(\)"
    r"|(?:de\s+)?\d+\s+dias?"
)
_FREQ_DINAMICA = (
    r"do\s+govendas|do\s+vendedor(?:\s+(?:mais|menos)\s+\d+\s+dias?)?"
    r"|do\s+historico(?:\s+(?:mais|menos)\s+\d+\s+dias?)?"
    r"|de\s+qualquer\s+vendedor|(?:de\s+)?(?:ativos|inativos)\s+d[oa]\s+(?:empresa|vendedor)"
    r"|remover"
)
_FREQ_LIMITE = rf"no\s+(?:maximo|minimo)\s+(?:{_FREQ_FIXA})"
FREQUENCIA_ATOMO = re.compile(rf"^(?:{_FREQ_FIXA}|{_FREQ_DINAMICA}|{_FREQ_LIMITE})$")


def frequencia_valida(valor: str) -> bool:
    """Valida o texto após `frequencia`, com cascatas `ou` e limites `e`.

    Ex.: `do historico ou no minimo de 7 dias e no maximo de 30 dias`.
    """
    valor = normalizar(valor)
    if not valor:
        return False
    for alternativa in re.split(r"\s+ou\s+", valor):
        for parte in re.split(r"\s+e\s+(?=no\s+(?:maximo|minimo)\b)", alternativa):
            if not FREQUENCIA_ATOMO.match(parte.strip()):
                return False
    return True


# Linha que continua a frequência da linha anterior (`ou no maximo de 30 dias`).
CONTINUACAO_FREQUENCIA = re.compile(r"^(?:ou|e)\s+(?P<resto>.+)$")


# ---------------------------------------------------------------------------
# Modificadores de parâmetros (cap. 7.3 e 7.5)
# ---------------------------------------------------------------------------

_DIA_PREFERENCIA = (
    r"(?:na|no|nos|nas)\s+(?:segunda|terca|quarta|quinta|sexta)(?:-feira)?"
    r"|no\s+(?:sabado|domingo)|nos\s+dias\s+uteis|no\s+fim\s+de\s+semana"
)
_ITEM_PREFERENCIA = rf"(?:(?:maior|menor|normal)\s+)?(?:{_DIA_PREFERENCIA})"
PREFERENCIA = re.compile(
    rf"^preferencia(?:"
    rf"\s+{_ITEM_PREFERENCIA}(?:\s+e\s+{_ITEM_PREFERENCIA})*"
    rf"|\s+do\s+(?:vendedor|historico)(?:\s+ou\s+(?:remover|{_ITEM_PREFERENCIA}(?:\s+e\s+{_ITEM_PREFERENCIA})*))?"
    rf")?(?:\s+e)?$"
)
SEM_CHOQUE = re.compile(
    r'^sem\s+choque(?:\s+(?:e\s+)?(?:na|fora\s+da)\s+(?:rota|pasta))*'
    r'(?:\s+(?:com|no|na|do|da))?\s+(?:time|carteira|segmentacao)\s*\(\s*"[^"]*"\s*\)$'
)
SEM_CHOQUE_REFERENCIA = re.compile(r'(?P<tipo>time|carteira|segmentacao)\s*\(\s*"(?P<nome>[^"]*)"\s*\)')
RESTRITO = re.compile(
    r"^restrito\s+(?:a\s+partir\s+da|ate\s+a)\s+(?:primeira|segunda|terceira|quarta)\s+semana$"
)
MODIFICADOR_SIMPLES = re.compile(r"^(?:fixo|roteirizavel)$")


def modificador_valido(texto: str) -> bool:
    """Valida um modificador já agrupado pelo parser (pode ter várias linhas)."""
    linhas = [normalizar(linha) for linha in texto.splitlines() if linha.strip()]
    if not linhas:
        return False
    primeira = linhas[0]
    if primeira.startswith("preferencia"):
        return bool(PREFERENCIA.match(" ".join(linhas)))
    if len(linhas) > 1:
        return False
    return bool(
        MODIFICADOR_SIMPLES.match(primeira)
        or SEM_CHOQUE.match(primeira)
        or RESTRITO.match(primeira)
    )
