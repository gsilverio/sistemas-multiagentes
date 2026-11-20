"""Agente Analista de negócio: aponta falhas na modelagem comercial do `RuleSpec`.

Determinístico (sem LLM): as falhas que importam são verificáveis com
aritmética sobre as faixas de dias e com a semântica do GOBR. Recebe só o
`RuleSpec`, nunca o GOBR (verificação cega).

A semântica que mais gera erro (manual, seção 7.6): **clientes parametrizados
por uma regra anterior não são reprocessados pelas seguintes; a ordem
importa**. Por isso o Analista:

- reordena os segmentos (bloqueios `nao agendar` primeiro, faixas de dias
  da menor para a maior, regra de "demais clientes" por último) e registra
  o ajuste;
- aponta buracos entre faixas (clientes que nenhuma regra pega),
  sobreposições (a regra de cima vence) e clientes além da última faixa sem
  regra de fallback;
- pergunta o que falta para gerar (segmento sem ação; frequência inválida);
- avisa o que não se escreve em `parametros` (capacidade diária, por
  exemplo) e bloqueios comuns que o cliente não mencionou.
"""

from __future__ import annotations

import re
import unicodedata

from src.agents.contratos import Achado, Analise, Pergunta, RuleSpec, Segmento

INFINITO = 10**9
FREQUENCIA_MAXIMA = 365

_FORA_DO_GOBR = re.compile(r"capacidade|clientes\s+(?:por|/)\s*dia|meta|comiss|or[cç]amento|ligar\s+\d+")
_BLOQUEIOS_COMUNS = {
    "inadimpl": "Não há regra para clientes inadimplentes. É comum não agendá-los (`quando inadimplentes` + `nao agendar`).",
    "receita": "Não há regra para clientes baixados na Receita Federal. É comum não agendá-los "
               "(`quando status na receita federal definido como baixada` + `nao agendar`).",
}


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto.casefold()) if not unicodedata.combining(c))


def _faixa(segmento: Segmento) -> tuple[int, int] | None:
    f = segmento.faixa
    if f is None or (f.min_dias is None and f.max_dias is None):
        return None
    return (f.min_dias or 0, f.max_dias if f.max_dias is not None else INFINITO)


def _ordenar(spec: RuleSpec) -> tuple[list[Segmento], bool]:
    """Bloqueios primeiro, faixas em ordem crescente nos lugares delas, fallback por último."""
    bloqueios = [s for s in spec.segmentos if s.nao_agendar and not s.fallback]
    fallback = [s for s in spec.segmentos if s.fallback]
    meio = [s for s in spec.segmentos if s not in bloqueios and s not in fallback]
    com_faixa = sorted((s for s in meio if _faixa(s)), key=lambda s: (_faixa(s)[1], _faixa(s)[0]))
    fila = iter(com_faixa)
    meio = [next(fila) if _faixa(s) else s for s in meio]
    nova = bloqueios + meio + fallback
    return nova, [s.nome for s in nova] != [s.nome for s in spec.segmentos]


class AgenteAnalista:
    def analisar(self, spec: RuleSpec) -> Analise:
        achados: list[Achado] = []
        perguntas: list[Pergunta] = []
        ajustes: list[str] = []
        spec = spec.model_copy(deep=True)

        if not spec.segmentos:
            perguntas.append(Pergunta(id="sem_segmentos", bloqueante=True,
                                      texto="Não identifiquei nenhum grupo de clientes. Quais grupos existem, "
                                            "como cada um é definido e com que frequência deve ser atendido?"))
            return Analise(achados=achados, perguntas=perguntas, ajustes=ajustes, spec=spec)

        # Nomes repetidos: o GOBR exige nomes distintos.
        vistos: dict[str, int] = {}
        for s in spec.segmentos:
            chave = _sem_acento(s.nome.strip())
            if chave in vistos:
                vistos[chave] += 1
                novo = f"{s.nome} ({vistos[chave]})"
                ajustes.append(f'Renomeei o segundo segmento "{s.nome}" para "{novo}": o GOBR exige nomes distintos.')
                s.nome = novo
            else:
                vistos[chave] = 1

        for s in spec.segmentos:
            if not s.nao_agendar and not s.frequencia_dias and not s.frequencia_texto:
                perguntas.append(Pergunta(
                    id=f"acao:{s.nome}", bloqueante=True,
                    texto=f'O grupo "{s.nome}" não tem frequência de atendimento. Com que frequência '
                          "esses clientes devem ser atendidos (ou eles não devem ser agendados)?"))
            if s.frequencia_dias is not None and not 1 <= s.frequencia_dias <= FREQUENCIA_MAXIMA:
                perguntas.append(Pergunta(
                    id=f"frequencia:{s.nome}", bloqueante=True,
                    texto=f'A frequência de {s.frequencia_dias} dias do grupo "{s.nome}" está fora do intervalo '
                          f"aceito (1 a {FREQUENCIA_MAXIMA} dias). Qual é a frequência correta?"))
            if s.nao_agendar and s.frequencia_dias:
                achados.append(Achado(codigo="acao_dupla", severidade="aviso", segmento=s.nome,
                                      mensagem=f'O grupo "{s.nome}" tem frequência e também "não agendar"; '
                                               "vale o não agendar e a frequência foi descartada."))
                s.frequencia_dias = None
            f = s.faixa
            if f and f.min_dias is not None and f.max_dias is not None and f.min_dias >= f.max_dias:
                perguntas.append(Pergunta(
                    id=f"faixa:{s.nome}", bloqueante=True,
                    texto=f'A faixa do grupo "{s.nome}" vai de {f.min_dias} a {f.max_dias} dias, '
                          "o início é maior que o fim. Qual é o intervalo correto?"))

        ordenados, mudou = _ordenar(spec)
        if mudou:
            ajustes.append(
                "Reordenei as regras: no GOBR o cliente fica com a primeira regra que o atende (manual, 7.6). "
                "Bloqueios vêm primeiro, faixas de dias da menor para a maior e a regra dos demais clientes por "
                "último. Nova ordem: " + " → ".join(f'"{s.nome}"' for s in ordenados) + ".")
        spec.segmentos = ordenados

        self._faixas(spec, achados, perguntas)

        for restricao in spec.restricoes_gerais:
            if _FORA_DO_GOBR.search(_sem_acento(restricao)):
                achados.append(Achado(codigo="fora_do_gobr", severidade="aviso",
                                      mensagem=f'"{restricao}" não é configurado nas regras de parâmetros do GOBR; '
                                               "precisa ser ajustado em outra configuração do goVendas."))

        textos = _sem_acento(" ".join([s.nome + " " + s.criterio for s in spec.segmentos] + spec.restricoes_gerais))
        for chave, mensagem in _BLOQUEIOS_COMUNS.items():
            if chave not in textos:
                achados.append(Achado(codigo=f"sugestao_{chave}", severidade="info", mensagem=mensagem))

        return Analise(achados=achados, perguntas=perguntas, ajustes=ajustes, spec=spec)

    @staticmethod
    def _faixas(spec: RuleSpec, achados: list[Achado], perguntas: list[Pergunta]) -> None:
        por_medida: dict[str, list[Segmento]] = {}
        for s in spec.segmentos:
            if _faixa(s) and not s.nao_agendar:
                por_medida.setdefault(s.faixa.medida, []).append(s)
        tem_fallback = any(s.fallback or (not s.criterio.strip() and not s.faixa and not s.nao_agendar)
                           for s in spec.segmentos)
        for medida, segmentos in por_medida.items():
            alvo = "compra" if medida == "dias_sem_compra" else "visita"
            coberto = 0
            anterior = None
            for s in sorted(segmentos, key=lambda s: _faixa(s)):
                inicio, fim = _faixa(s)
                if inicio > coberto:
                    achados.append(Achado(
                        codigo="buraco_faixa", severidade="erro", segmento=s.nome,
                        mensagem=f"Clientes com {coberto} a {inicio} dias sem {alvo} não se encaixam em nenhum "
                                 f'grupo (o anterior termina em {coberto} e "{s.nome}" começa em {inicio}).'))
                elif inicio < coberto and inicio > 0 and anterior:
                    achados.append(Achado(
                        codigo="sobreposicao_faixa", severidade="aviso", segmento=s.nome,
                        mensagem=f'"{anterior}" e "{s.nome}" se sobrepõem entre {inicio} e {coberto} dias sem '
                                 f'{alvo}; esses clientes ficam com a regra "{anterior}", que vem antes.'))
                coberto = max(coberto, fim)
                anterior = s.nome
            if coberto < INFINITO and not tem_fallback:
                perguntas.append(Pergunta(
                    id=f"alem_da_faixa:{medida}",
                    texto=f"Clientes com mais de {coberto} dias sem {alvo} não se encaixam em nenhum grupo. "
                          "O que fazer com eles (por exemplo, não agendar ou atender a cada N dias)?",
                    suposicao="ficam sem regra de parametrização (não são agendados por estas regras)"))
                achados.append(Achado(
                    codigo="sem_fallback", severidade="aviso",
                    mensagem=f"Não há regra para clientes com mais de {coberto} dias sem {alvo} "
                             "nem uma regra final para os demais clientes."))
