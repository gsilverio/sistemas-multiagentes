"""Extrai blocos GOBR sem misturar regras com declaracoes do mesmo arquivo.

Nao executa IA e nao interpreta a DSL: apenas separa os blocos e vincula
as definicoes locais referenciadas, preservando rastreabilidade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GobrDefinition:
    tipo: str
    nome: str
    texto: str
    linha_inicio: int
    linha_fim: int


@dataclass(frozen=True)
class GobrRuleBlock:
    id: str
    arquivo_origem: str
    indice: int
    nome: str
    linha_inicio: int
    linha_fim: int
    gobr: str
    contexto: list[GobrDefinition] = field(default_factory=list)
    referencias_nao_resolvidas: list[str] = field(default_factory=list)


class GobrRuleExtractor:
    # Cabecalhos de nivel zero. Greedy para suportar aspas internas no titulo.
    CABECALHO = re.compile(
        r'^(?P<tipo>parametros|time|carteira(?:\s+livre)?|segmentacao)\s+'
        r'"(?P<nome>.*)"\s*:\s*(?:#.*)?$',
        flags=re.IGNORECASE,
    )
    ESCOPO = re.compile(
        r'\b(?:do\s+time|da\s+carteira|da\s+segmenta[cç][aã]o)\s*'
        r'\((?P<argumentos>[^)]*)\)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    ASPAS = re.compile(r'"([^"\r\n]+)"')

    def extract_file(self, arquivo: Path) -> list[GobrRuleBlock]:
        arquivo = Path(arquivo)
        linhas = arquivo.read_text(encoding="utf-8-sig").splitlines()
        secoes: list[tuple[str, str, int, int, str]] = []
        cabecalhos: list[tuple[int, re.Match[str]]] = []

        for posicao, linha in enumerate(linhas):
            # Regras e definicoes sao declaracoes de nivel zero.
            if not linha or linha[0].isspace():
                continue
            encontrado = self.CABECALHO.match(linha)
            if encontrado:
                cabecalhos.append((posicao, encontrado))

        for indice, (inicio, match) in enumerate(cabecalhos):
            fim = (cabecalhos[indice + 1][0]
                   if indice + 1 < len(cabecalhos) else len(linhas))
            trecho = linhas[inicio:fim]
            # Comentarios separadores entre declaracoes nao pertencem ao bloco.
            while trecho and (not trecho[-1].strip() or
                              trecho[-1].lstrip().startswith("#")):
                trecho.pop()
            if not trecho:
                continue
            secoes.append((
                match.group("tipo").lower(), match.group("nome"),
                inicio + 1, inicio + len(trecho), "\n".join(trecho),
            ))

        definicoes = [
            GobrDefinition(tipo, nome, texto, inicio, fim)
            for tipo, nome, inicio, fim, texto in secoes
            if tipo != "parametros"
        ]
        por_nome: dict[str, list[GobrDefinition]] = {}
        for definicao in definicoes:
            por_nome.setdefault(definicao.nome.casefold(), []).append(definicao)

        blocos: list[GobrRuleBlock] = []
        for tipo, nome, inicio, fim, texto in secoes:
            if tipo != "parametros":
                continue
            numero = len(blocos) + 1
            diretas = self._referencias_diretas(texto)
            ausentes = sorted({valor for valor in diretas
                               if valor.casefold() not in por_nome})
            pendentes = list(diretas)
            encontradas: dict[tuple[str, int], GobrDefinition] = {}

            # Resolve tambem referencias indiretas, por exemplo:
            # parametros -> carteira("empresa toda") -> time("empresa toda").
            while pendentes:
                nome_ref = pendentes.pop(0)
                for definicao in por_nome.get(nome_ref.casefold(), []):
                    chave = (definicao.tipo, definicao.linha_inicio)
                    if chave in encontradas:
                        continue
                    encontradas[chave] = definicao
                    pendentes.extend(self._citacoes(definicao.texto.split("\n", 1)[-1]))

            blocos.append(GobrRuleBlock(
                id=f"{arquivo.stem}_{numero:03d}",
                arquivo_origem=arquivo.name,
                indice=numero,
                nome=nome,
                linha_inicio=inicio,
                linha_fim=fim,
                gobr=texto,
                contexto=sorted(encontradas.values(), key=lambda d: d.linha_inicio),
                referencias_nao_resolvidas=ausentes,
            ))
        return blocos

    def extract_directory(self, diretorio: Path) -> list[GobrRuleBlock]:
        diretorio = Path(diretorio)
        return [bloco for arquivo in sorted(diretorio.glob("*.gobr"))
                for bloco in self.extract_file(arquivo)]

    def _referencias_diretas(self, regra: str) -> list[str]:
        corpo = regra.split("\n", 1)[-1]  # Ignora o titulo do bloco.
        return list(dict.fromkeys(
            nome for escopo in self.ESCOPO.finditer(corpo)
            for nome in self._citacoes(escopo.group("argumentos"))
        ))

    def _citacoes(self, texto: str) -> list[str]:
        return [match.group(1) for match in self.ASPAS.finditer(texto)]
