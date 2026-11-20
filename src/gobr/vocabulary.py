"""Vocabulário de condições (`quando ...`) aceitas pelo GOBR.

Duas fontes, combinadas:

1. **Acervo**: todas as condições usadas nas 246 regras de referência,
   normalizadas em *modelos* (números viram `N`, listas entre aspas viram
   `("X")`). Gerado por `python -m src.pipelines.build_vocabulary` e salvo em
   `data/vocabulario/condicoes.json`.
2. **Manual**: padrões do capítulo 6 (filtros de clientes), escritos à mão
   como expressões regulares em `PADROES_MANUAL`, para aceitar filtros
   documentados que o acervo não usa.

Uma condição desconhecida não é necessariamente inválida (o manual pode ter
variações que não mapeamos), por isso o validador a trata como **aviso** e
sugere as condições conhecidas mais parecidas.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.gobr.grammar import normalizar

CAMINHO_PADRAO = Path(__file__).resolve().parents[2] / "data" / "vocabulario" / "condicoes.json"


def modelo_condicao(texto: str) -> str:
    """Normaliza uma condição para comparação com o vocabulário.

    `recencia de positivação em 60 dias` -> `recencia de positivacao em N dias`
    `ramo de atividade igual("1", "12")`  -> `ramo de atividade igual("X")`
    """
    texto = normalizar(texto)
    texto = re.sub(r"\bn\b", "N", texto)  # Idempotente: um modelo continua modelo.
    texto = re.sub(r'\(\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*,?\s*\)', '("X")', texto)
    texto = re.sub(r'"[^"]*"', '"X"', texto)
    texto = re.sub(r"'[^']*'", "'X'", texto)
    texto = re.sub(r"\d+(?:[.,]\d+)?", "N", texto)
    texto = re.sub(r"\bN dia\b", "N dias", texto)
    return re.sub(r"\s+", " ", texto).strip()


# ---------------------------------------------------------------------------
# Padrões do manual (cap. 6), sobre o texto já passado por `modelo_condicao`.
# Cada item: (expressão regular, exemplo legível usado nas sugestões).
# ---------------------------------------------------------------------------

_UNIDADE = r"(?:anos?|semestres?|trimestres?|bimestres?|mes|meses|semanas?|dias|horas?|minutos?)"
_PERIODO = r"(?:semana|quinzena|mes|bimestre|trimestre|semestre|ano)"
_COND_TEMPO = (
    rf"(?:em N {_UNIDADE}(?: a frente)?|ha N {_UNIDADE} atras|nos proximos N {_UNIDADE}"
    rf"|entre N {_UNIDADE} e N {_UNIDADE}|entre 'X' e 'X'"
    rf"|na manha|na tarde|na noite|no dia|de ontem|ontem"
    rf"|na {_PERIODO}|no {_PERIODO}|na {_PERIODO} anterior|no {_PERIODO} anterior|no ano passado)"
)
_TEMPO = rf"N {_UNIDADE}"
_TEMPO_FIXO = rf"(?:do dia|de ontem|na {_PERIODO}|no {_PERIODO}|na {_PERIODO} anterior|no {_PERIODO} anterior)"
_OPERADOR_TEXTO = r"(?:e )?(?:igual|diferente|contem|inicia com)"
_CAMPOS_TEXTO = (
    r"(?:nome da cidade|nome do bairro|uf do estado|nome da rota|conteudo da nota|nome da nota"
    r"|o rotulo|cnae|cnae da receita|cnae do erp|nome do cnae da receita|ramo de atividade"
    r"|codigo da rota|id do cliente|id do cliente principal|tipo de pagamento|tipo de cobranca)"
)

PADROES_MANUAL: tuple[tuple[str, str], ...] = (
    # 6.1 atividade e inatividade
    (r"(?:ativos|inativos)(?: para (?:o vendedor|a empresa))?", "ativos para o vendedor"),
    (r"inatividade(?: global)? de N dias", "inatividade de 90 dias"),
    (r"sem vendas?(?: do vendedor| da empresa)?(?: ou prospects)?", "sem venda"),
    (r"unica compra com (?:a empresa|o vendedor)", "unica compra com o vendedor"),
    (r"prospects? (?:do vendedor|da empresa)", "prospects da empresa"),
    (rf"(?:nao )?tem venda(?: do vendedor| da empresa)? {_COND_TEMPO}", "tem venda da empresa em 30 dias"),
    (rf"tem N% de visita (?:remota|incerta) (?:em )?{_TEMPO}", "tem 50% de visita remota em 30 dias"),
    (rf"atividade(?: global| do vendedor)? de {_TEMPO}", "atividade do vendedor de 30 dias"),
    # 6.2 recência
    (rf"(?:fora da )?recencia de (?:visita|positivacao|venda)(?: global| do vendedor)? {_COND_TEMPO}",
     "recencia de positivacao em 60 dias"),
    # 6.3 financeiros
    (r"(?:in)?adimplentes", "adimplentes"),
    (r"(?:nao )?tem credito", "tem credito"),
    (r"com limite de r\$ N de credito|com r\$ N de credito|com menos de r\$ N de credito disponivel",
     "com menos de R$ 500 de credito disponivel"),
    (r"com N% de credito", "com 50% de credito"),
    (r"com ticket medio (?:maior que|menor que|igual a) r\$ N(?: ou indefinido)?", "com ticket medio maior que R$ 300"),
    (rf"(?:total|media) faturad[oa] no {_PERIODO} com a empresa (?:maior|menor) que r\$ N"
     rf"(?: e (?:maior|menor) que r\$ N)?(?: ou indefinido)?",
     "media faturada no ano com a empresa maior que R$ 1000"),
    # 6.4 agenda
    (rf"(?:na|fora da) agenda(?: (?:padrao|de ruptura|complementar))?(?: do vendedor| global)?"
     rf"(?: de hoje| de ontem| na semana| em N dias)", "na agenda de hoje"),
    (r"(?:nao )?agendavel|(?:nao )?reagendado(?: para (?:o vendedor|a empresa))?", "agendavel"),
    # 6.5 / 6.6 localização, classificação e ids
    (rf"{_CAMPOS_TEXTO} {_OPERADOR_TEXTO}\(\"X\"\)", 'nome da cidade igual("SAO PAULO")'),
    (r"status na receita federal definido como (?:ativa|suspensa|inapta|baixada)",
     "status na receita federal definido como baixada"),
    (r"pessoa (?:fisica|juridica)|cliente (?:principal|filial)|e o cliente (?:principal|filial)", "pessoa juridica"),
    (r"(?:nao )?(?:excluidos|bloqueados)|(?:nao )?tem coordenada confiavel", "nao bloqueados"),
    # 6.7 curva ABC e ciclo de vida
    (r"curva [abc] com (?:a empresa|o vendedor)", "curva A com a empresa"),
    (r"no ciclo de vida (?:ativo|inativo|prospect|ativado|reativado) com (?:a empresa|o vendedor)",
     "no ciclo de vida prospect com a empresa"),
    # 6.8 frequência e parâmetros
    (r"nao tem parametro(?: manual)?", "nao tem parametro"),
    (rf"fora da frequencia de venda(?: indicada)?(?: {_COND_TEMPO})?(?: ou maior que N dias)?",
     "fora da frequencia de venda ou maior que 30 dias"),
    (r"frequencia historica (?:maior que N dias|menor que N dias(?: ou indefinida)?|entre N dias e N dias|de N dias)",
     "frequencia historica menor que 30 dias"),
    (r"atendimentos? N% (?:menor|maior) que o parametrizado", "atendimentos 20% menor que o parametrizado"),
    (r"a ultima positivacao N% (?:menor|maior) que o ticket medio", "a ultima positivacao 20% menor que o ticket medio"),
    (r"parametro de frequencia N% (?:menor|maior) que o previsto", "parametro de frequencia 20% menor que o previsto"),
    # 6.9 / 6.10 segmentação e ruptura
    (r"(?:nao )?contido na segmentacao\(\"X\"\)", 'contido na segmentacao("vip")'),
    (r"em ruptura global do (?:bimestre|trimestre|semestre|ano)", "em ruptura global do trimestre"),
    (r"(?:sem|com) contato no ultimo agendamento(?: para (?:o vendedor|a empresa))?"
     r"|sem contato nos ultimos N agendamentos(?: para (?:o vendedor|a empresa))?",
     "sem contato no ultimo agendamento"),
    # 6.11 outros
    (rf"positivado {_TEMPO_FIXO}|positivado de hoje|atendimento remoto em {_TEMPO}", "positivado de hoje"),
    (r"sem perfil com (?:o vendedor|a empresa)", "sem perfil com o vendedor"),
    (r"(?:total de compras|compras no (?:mes|bimestre|trimestre|semestre|ano)) com (?:o vendedor|a empresa)"
     r" (?:maior que N|menor que N)(?: ou indefinido)?", "total de compras com o vendedor maior que 3"),
    (r"queda de faturamento (?:menor|maior) que N%", "queda de faturamento maior que 20%"),
    (rf"cadastrado (?:faz|ha) {_TEMPO}|cadastrado a partir do dia 'X'", "cadastrado faz 90 dias"),
    (r"elegivel(?: incluindo roteirizavel)?", "elegivel"),
    (rf"data da (?:proxima|ultima) visita do representante (?:{_COND_TEMPO}|em N {_UNIDADE})",
     "data da proxima visita do representante em 2 semanas"),
    (rf"data da rota(?: futura)?(?: do vendedor)? (?:{_COND_TEMPO}|em N {_UNIDADE}(?: a frente)?|hoje)",
     "data da rota na semana"),
)
_PADROES_COMPILADOS = tuple((re.compile(rf"^(?:{padrao})$"), exemplo) for padrao, exemplo in PADROES_MANUAL)


@dataclass
class EntradaVocabulario:
    modelo: str
    ocorrencias: int
    exemplo: str
    fonte: str = "acervo"


@dataclass
class VocabularioCondicoes:
    entradas: dict[str, EntradaVocabulario] = field(default_factory=dict)

    # -- construção ---------------------------------------------------------

    @classmethod
    def carregar(cls, caminho: Path | None = None) -> "VocabularioCondicoes":
        caminho = Path(caminho or CAMINHO_PADRAO)
        if not caminho.is_file():
            return cls()  # Só os padrões do manual.
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return cls({
            item["modelo"]: EntradaVocabulario(**item) for item in dados["condicoes"]
        })

    def registrar(self, texto: str) -> None:
        modelo = modelo_condicao(texto)
        if modelo in self.entradas:
            self.entradas[modelo].ocorrencias += 1
        else:
            self.entradas[modelo] = EntradaVocabulario(modelo, 1, texto.strip())

    def salvar(self, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        itens = sorted(self.entradas.values(), key=lambda e: (-e.ocorrencias, e.modelo))
        caminho.write_text(json.dumps({
            "descricao": "Condicoes `quando` do acervo data/referencias, normalizadas.",
            "total": len(itens),
            "condicoes": [vars(item) for item in itens],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # -- consulta -----------------------------------------------------------

    def conhece(self, texto: str) -> bool:
        """True se a condição (ou cada parte dela ligada por e/ou) é conhecida."""
        return _cobre(tuple(_partes(modelo_condicao(texto))), self._conhece_modelo)

    def _conhece_modelo(self, modelo: str) -> bool:
        if modelo in self.entradas:
            return True
        return any(padrao.match(modelo) for padrao, _ in _PADROES_COMPILADOS)

    def sugestoes(self, texto: str, limite: int = 3) -> list[str]:
        """Exemplos reais das condições conhecidas mais parecidas."""
        modelo = modelo_condicao(texto)
        candidatos: dict[str, str] = {m: e.exemplo for m, e in self.entradas.items()}
        for _, exemplo in _PADROES_COMPILADOS:
            candidatos.setdefault(modelo_condicao(exemplo), exemplo)
        proximos = difflib.get_close_matches(modelo, list(candidatos), n=limite, cutoff=0.5)
        return [candidatos[m] for m in proximos]


def _partes(modelo: str) -> list[str]:
    """Separa em segmentos mantendo os conectivos: ['a', 'e', 'b', 'ou', 'c']."""
    return re.split(r"\s+(e|ou)\s+", modelo)


def _cobre(partes: tuple[str, ...], conhece) -> bool:
    """Programação dinâmica: os segmentos podem ser agrupados em condições
    conhecidas? Necessário porque `entre N dias e N dias` também contém `e`."""

    @lru_cache(maxsize=None)
    def pode(inicio: int) -> bool:
        if inicio >= len(partes):
            return True
        for fim in range(inicio, len(partes), 2):
            if conhece(" ".join(partes[inicio:fim + 1])) and pode(fim + 2):
                return True
        return False

    return pode(0)
