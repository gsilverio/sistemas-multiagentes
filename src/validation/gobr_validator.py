"""Validador determinístico de arquivos GOBR completos.

Responde a uma pergunta: **este texto é GOBR correto?** Não usa LLM. É o
juiz do agente Gerador (o grafo devolve os erros para ele corrigir) e a base
da avaliação do fine-tuning.

Verifica, nesta ordem:

1. Estrutura do arquivo: `version 1.0`, cabeçalhos de bloco, indentação,
   ordem das categorias, nomes repetidos, pelo menos um `time`.
2. Sintaxe de cada linha: parênteses e aspas balanceados, caracteres
   invisíveis, corpo de `time`, `carteira`, `segmentacao` e `parametros`
   conforme a gramática do manual (`src.gobr.grammar`).
3. Referências: todo `do time(...)`, `da carteira(...)`, `segmentados(...)`,
   `sem choque no time(...)` etc. aponta para um bloco do tipo certo,
   definido **antes** do uso.
4. Regras do manual para `parametros`: escopo obrigatório, alguma ação
   (frequência, `nao agendar` ou modificador), `nao agendar` exclusivo.
5. Vocabulário das condições `quando` (aviso, com sugestões).

Fica de fora, de propósito, a análise de negócio (faixas de recência com
buracos, falta de regra de fallback etc.): isso é papel do agente Analista.

Uso:
    python -m src.validation.gobr_validator caminho.gobr [outro.gobr ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.gobr import grammar
from src.gobr.ast import GobrParsedRule
from src.gobr.grammar import normalizar
from src.gobr.parser import GobrParser
from src.gobr.vocabulary import VocabularioCondicoes

Severidade = Literal["erro", "aviso"]
_CONTROLE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass(frozen=True)
class Problema:
    codigo: str
    severidade: Severidade
    mensagem: str
    linha: int | None = None
    bloco: str | None = None
    sugestoes: tuple[str, ...] = ()

    def como_texto(self) -> str:
        onde = []
        if self.linha is not None:
            onde.append(f"linha {self.linha}")
        if self.bloco is not None:
            onde.append(f'bloco "{self.bloco}"')
        prefixo = f"[{self.severidade.upper()} {self.codigo}]"
        texto = f"{prefixo} {', '.join(onde)}: {self.mensagem}" if onde else f"{prefixo} {self.mensagem}"
        if self.sugestoes:
            texto += " Sugestões: " + "; ".join(f"`{s}`" for s in self.sugestoes)
        return texto


@dataclass
class Bloco:
    categoria: str          # como escrito: time, carteira livre, parametros...
    nome: str
    linha: int              # linha do cabeçalho (1-based)
    corpo: list[tuple[int, str]] = field(default_factory=list)

    @property
    def canonica(self) -> str:
        return grammar.categoria_canonica(self.categoria)


@dataclass
class LinhaLogica:
    numero: int
    texto: str
    balanceada: bool = True


@dataclass
class RelatorioValidacao:
    problemas: list[Problema] = field(default_factory=list)
    blocos: list[Bloco] = field(default_factory=list)
    parametros: dict[str, GobrParsedRule] = field(default_factory=dict)

    @property
    def erros(self) -> list[Problema]:
        return [p for p in self.problemas if p.severidade == "erro"]

    @property
    def avisos(self) -> list[Problema]:
        return [p for p in self.problemas if p.severidade == "aviso"]

    @property
    def ok(self) -> bool:
        return not self.erros

    def como_texto(self) -> str:
        estado = "GOBR válido" if self.ok else "GOBR inválido"
        cabecalho = f"{estado}: {len(self.erros)} erro(s), {len(self.avisos)} aviso(s)."
        ordenados = sorted(self.problemas, key=lambda p: (p.severidade != "erro", p.linha or 0))
        return "\n".join([cabecalho, *(p.como_texto() for p in ordenados)])

    def para_dict(self) -> dict:
        return {
            "ok": self.ok,
            "erros": len(self.erros),
            "avisos": len(self.avisos),
            "problemas": [asdict(p) for p in self.problemas],
            "blocos": [{"categoria": b.categoria, "nome": b.nome, "linha": b.linha} for b in self.blocos],
        }


class GobrValidator:
    def __init__(self, vocabulario: VocabularioCondicoes | None = None) -> None:
        self.vocabulario = vocabulario or VocabularioCondicoes.carregar()
        self.parser = GobrParser()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def validar(self, texto: str) -> RelatorioValidacao:
        relatorio = RelatorioValidacao()
        self._caracteres(texto, relatorio)
        relatorio.blocos = self._estrutura(texto, relatorio)
        self._ordem_e_nomes(relatorio)

        definidos: dict[str, dict[str, int]] = {"time": {}, "segmentacao": {}, "carteira": {}}
        for bloco in relatorio.blocos:
            if bloco.canonica == "time":
                self._time(bloco, relatorio)
            elif bloco.canonica in ("carteira", "segmentacao"):
                self._carteira_ou_segmentacao(bloco, definidos, relatorio)
            elif bloco.canonica == "parametros":
                self._parametros(bloco, definidos, relatorio)
            if bloco.canonica in definidos:
                definidos[bloco.canonica].setdefault(normalizar(bloco.nome), bloco.linha)
        return relatorio

    def validar_arquivo(self, caminho: Path) -> RelatorioValidacao:
        return self.validar(Path(caminho).read_text(encoding="utf-8-sig"))

    # ------------------------------------------------------------------
    # 1. Estrutura do arquivo
    # ------------------------------------------------------------------

    def _caracteres(self, texto: str, rel: RelatorioValidacao) -> None:
        for numero, linha in enumerate(texto.splitlines(), start=1):
            if _CONTROLE.search(linha):
                rel.problemas.append(Problema(
                    "E101", "erro", "Caractere de controle invisível na linha; remova-o.", numero))

    def _estrutura(self, texto: str, rel: RelatorioValidacao) -> list[Bloco]:
        blocos: list[Bloco] = []
        versao_vista = False
        primeira = True
        for numero, original in enumerate(texto.splitlines(), start=1):
            linha = _sem_comentario(original).rstrip()
            if not linha.strip():
                continue
            indentada = linha[0].isspace()
            conteudo = linha.strip()
            if not indentada and grammar.VERSION.match(conteudo):
                if versao_vista:
                    rel.problemas.append(Problema("E001", "erro", "`version 1.0` repetido.", numero))
                elif not primeira:
                    rel.problemas.append(Problema(
                        "E001", "erro", "`version 1.0` deve ser a primeira linha do arquivo.", numero))
                versao_vista = True
                primeira = False
                continue
            if primeira and not versao_vista:
                rel.problemas.append(Problema(
                    "E001", "erro", "O arquivo deve começar com `version 1.0`.", numero))
            primeira = False
            cabecalho = grammar.CABECALHO.match(conteudo)
            if cabecalho:
                if indentada:
                    rel.problemas.append(Problema(
                        "A004", "aviso", "Cabeçalho de bloco indentado; escreva-o na coluna 1.", numero))
                blocos.append(Bloco(
                    normalizar(cabecalho.group("categoria")), cabecalho.group("nome"), numero))
                continue
            if not indentada:
                rel.problemas.append(Problema(
                    "E002", "erro",
                    f"Linha fora de qualquer bloco: `{conteudo}`. Linhas de corpo devem ser "
                    "indentadas; cabeçalhos seguem o formato `<categoria> \"nome\":`.", numero))
                continue
            if not blocos:
                rel.problemas.append(Problema(
                    "E002", "erro", f"Linha indentada antes do primeiro bloco: `{conteudo}`.", numero))
                continue
            blocos[-1].corpo.append((numero, conteudo))
        if not versao_vista and not any(p.codigo == "E001" for p in rel.problemas):
            rel.problemas.append(Problema("E001", "erro", "Arquivo vazio ou sem `version 1.0`."))
        return blocos

    def _ordem_e_nomes(self, rel: RelatorioValidacao) -> None:
        if not any(b.canonica == "time" for b in rel.blocos):
            rel.problemas.append(Problema(
                "E007", "erro", "O arquivo precisa definir pelo menos um `time` (manual, cap. 10)."))
        maior_rank = 0
        anterior = ""
        vistos: dict[tuple[str, str], int] = {}
        tem_parametros = any(b.canonica == "parametros" for b in rel.blocos)
        for bloco in rel.blocos:
            if not bloco.corpo:
                rel.problemas.append(Problema("E004", "erro", "Bloco sem corpo.", bloco.linha, bloco.nome))
            if bloco.canonica in grammar.CATEGORIAS_NAO_SUPORTADAS:
                if tem_parametros:
                    rel.problemas.append(Problema(
                        "E003", "erro", "Alertas/tags não podem ser combinados com parametros no mesmo "
                        "arquivo (manual, cap. 8).", bloco.linha, bloco.nome))
                else:
                    rel.problemas.append(Problema(
                        "A003", "aviso", f"Categoria `{bloco.categoria}` não é verificada por este validador.",
                        bloco.linha, bloco.nome))
                continue
            rank = grammar.ORDEM_CATEGORIAS[bloco.canonica]
            if rank < maior_rank:
                rel.problemas.append(Problema(
                    "E005", "erro", f"`{bloco.categoria}` aparece depois de `{anterior}`. A ordem é "
                    "time → segmentacao → carteira → parametros.", bloco.linha, bloco.nome))
            else:
                maior_rank, anterior = rank, bloco.categoria
            chave = (bloco.canonica, normalizar(bloco.nome))
            if chave in vistos:
                rel.problemas.append(Problema(
                    "E006", "erro", f"Nome repetido: já existe `{bloco.canonica} \"{bloco.nome}\"` na linha "
                    f"{vistos[chave]}. Os nomes devem ser únicos por categoria.", bloco.linha, bloco.nome))
            else:
                vistos[chave] = bloco.linha

    # ------------------------------------------------------------------
    # 2. Corpo dos blocos
    # ------------------------------------------------------------------

    def _logicas(self, bloco: Bloco, rel: RelatorioValidacao) -> list[LinhaLogica]:
        """Junta linhas com parênteses abertos e acusa desbalanceamento."""
        logicas: list[LinhaLogica] = []
        atual: LinhaLogica | None = None
        saldo = 0
        for numero, conteudo in bloco.corpo:
            if atual is None:
                atual = LinhaLogica(numero, conteudo)
            else:
                atual.texto += " " + conteudo
            saldo += _saldo_parenteses(conteudo)
            if saldo <= 0:
                logicas.append(atual)
                atual, saldo = None, 0
        if atual is not None:
            atual.balanceada = False
            logicas.append(atual)
        for logica in logicas:
            if not logica.balanceada:
                rel.problemas.append(Problema(
                    "E008", "erro", "Parêntese aberto e não fechado; o restante do bloco foi engolido.",
                    logica.numero, bloco.nome))
            elif logica.texto.count('"') % 2:
                logica.balanceada = False
                rel.problemas.append(Problema(
                    "E008", "erro", f"Aspas desbalanceadas em `{logica.texto}`.", logica.numero, bloco.nome))
        return logicas

    def _time(self, bloco: Bloco, rel: RelatorioValidacao) -> None:
        tem_vendedores = False
        for linha in self._logicas(bloco, rel):
            if not linha.balanceada:
                continue
            texto = normalizar(linha.texto)
            if grammar.TIME_VENDEDORES.match(texto):
                tem_vendedores = True
            elif tem_vendedores and grammar.TIME_REFINAMENTO.match(texto):
                pass
            else:
                rel.problemas.append(Problema(
                    "E010", "erro", f"Linha inválida em time: `{linha.texto}`.", linha.numero, bloco.nome,
                    ("vendedores todos", "vendedores todostelevendas ativados", 'vendedores porid("1020", "1021")')))
        if bloco.corpo and not tem_vendedores:
            rel.problemas.append(Problema(
                "E010", "erro", "Time sem linha `vendedores ...`.", bloco.linha, bloco.nome))

    def _carteira_ou_segmentacao(self, bloco: Bloco, definidos, rel: RelatorioValidacao) -> None:
        tem_clientes = False
        apos_ruptura = False
        em_condicao = False
        segmentacoes_fonte = 0
        for linha in self._logicas(bloco, rel):
            if not linha.balanceada:
                continue
            texto = normalizar(linha.texto)
            if grammar.CARTEIRA_CLIENTES.match(texto):
                tem_clientes = True
                apos_ruptura = texto.startswith("clientes ruptura")
                em_condicao = False
                segmentacoes_fonte += bool(re.match(r"clientes (?:segmentados|ruptura da segmentacao)", texto))
            elif apos_ruptura and grammar.CARTEIRA_OPCAO_RUPTURA.match(texto):
                pass
            elif texto.startswith("quando "):
                em_condicao = True
                apos_ruptura = False
                self._condicao(linha.texto.strip()[len("quando "):], linha.numero, bloco, rel)
            elif em_condicao and texto.startswith(("e ", "ou ")):
                self._condicao(linha.texto.strip().split(" ", 1)[1], linha.numero, bloco, rel)
            else:
                rel.problemas.append(Problema(
                    "E011", "erro", f"Linha inválida em {bloco.categoria}: `{linha.texto}`.", linha.numero,
                    bloco.nome, ('clientes carteirafixa("nome do time")', "quando adimplentes")))
                continue
            for tipo, padrao in grammar.REFERENCIAS_EM_DEFINICAO:
                for achado in padrao.finditer(texto):
                    for nome in re.findall(r'"([^"]*)"', achado.group("args")):
                        self._referencia(tipo, nome, linha.numero, bloco, definidos, rel)
        if bloco.corpo and not tem_clientes:
            rel.problemas.append(Problema(
                "E011", "erro", f"{bloco.categoria} sem linha `clientes ...`.", bloco.linha, bloco.nome))
        if bloco.canonica == "carteira" and segmentacoes_fonte > 1:
            rel.problemas.append(Problema(
                "E012", "erro", "Uma carteira aceita uma única segmentação como fonte (manual, 4.5).",
                bloco.linha, bloco.nome))

    def _parametros(self, bloco: Bloco, definidos, rel: RelatorioValidacao) -> None:
        logicas = self._logicas(bloco, rel)
        if not all(l.balanceada for l in logicas):
            return  # Evita erros em cascata sobre um bloco sintaticamente quebrado.
        if bloco.categoria == "limpar parametros":
            if not any(re.match(r"(?:do time|da carteira|da segmentacao)\s*\(", normalizar(l.texto)) for l in logicas):
                rel.problemas.append(Problema("E031", "erro", "Bloco sem escopo.", bloco.linha, bloco.nome))
            return

        texto = f'parametros "{bloco.nome}":\n' + "\n".join(f"    {c}" for _, c in bloco.corpo)
        regra = self.parser.parse(texto)
        rel.parametros[bloco.nome] = regra
        linha_de = lambda trecho: _linha_de(bloco, trecho)

        for desconhecido in regra.desconhecidos:
            rel.problemas.append(Problema(
                "E030", "erro", f"Cláusula não reconhecida: `{desconhecido}`. Condições começam com "
                "`quando` (ou `e`/`ou` na linha seguinte).", linha_de(desconhecido), bloco.nome))

        if not regra.escopos:
            rel.problemas.append(Problema(
                "E031", "erro", "Parametros sem escopo: informe `da carteira(\"...\")`, `do time(\"...\")` "
                "ou `da segmentacao(\"...\")`.", bloco.linha, bloco.nome))
        for escopo in regra.escopos:
            for nome in escopo.valores:
                self._referencia(escopo.tipo, nome, linha_de(nome), bloco, definidos, rel)

        preferencias = [m for m in regra.modificadores if normalizar(m).startswith("preferencia")]
        if not regra.nao_agendar and not regra.frequencia and not regra.modificadores:
            rel.problemas.append(Problema(
                "E032", "erro", "Parametros sem ação: defina `frequencia ...`, `nao agendar` ou um modificador.",
                bloco.linha, bloco.nome, ("frequencia de 14 dias", "nao agendar")))
        if regra.nao_agendar and (regra.frequencia or preferencias):
            rel.problemas.append(Problema(
                "E033", "erro", "`nao agendar` é exclusivo: não pode haver frequência nem preferência no "
                "mesmo bloco (manual, 7.3.4).", linha_de("nao agendar"), bloco.nome))
        if regra.frequencia and not grammar.frequencia_valida(regra.frequencia):
            rel.problemas.append(Problema(
                "E034", "erro", f"Frequência inválida: `{regra.frequencia}`.", linha_de("frequencia"), bloco.nome,
                ("frequencia de 14 dias", "frequencia semanal()", "frequencia do vendedor ou de 35 dias")))

        for modificador in regra.modificadores:
            primeira = modificador.splitlines()[0]
            if not grammar.modificador_valido(modificador):
                rel.problemas.append(Problema(
                    "A002", "aviso", f"Modificador fora da gramática conhecida: `{modificador}`.",
                    linha_de(primeira), bloco.nome,
                    ("preferencia nos dias uteis", "preferencia maior na segunda e menor na sexta", "fixo")))
            choque = grammar.SEM_CHOQUE_REFERENCIA.search(normalizar(primeira)) if normalizar(primeira).startswith("sem choque") else None
            if choque:
                nome = re.search(r'"([^"]*)"', primeira)
                self._referencia(choque.group("tipo"), nome.group(1) if nome else choque.group("nome"),
                                 linha_de(primeira), bloco, definidos, rel)

        for condicao in regra.condicoes:
            self._condicao(condicao.texto, linha_de(condicao.texto), bloco, rel)

    # ------------------------------------------------------------------
    # 3. Referências e vocabulário
    # ------------------------------------------------------------------

    def _referencia(self, tipo: str, nome: str, linha: int | None, bloco: Bloco,
                    definidos: dict[str, dict[str, int]], rel: RelatorioValidacao) -> None:
        chave = normalizar(nome)
        if chave in definidos.get(tipo, {}):
            return
        depois = [b for b in rel.blocos if b.canonica == tipo and normalizar(b.nome) == chave and b.linha > bloco.linha]
        if depois:
            mensagem = (f'{tipo} "{nome}" é usado antes de ser definido (linha {depois[0].linha}). '
                        "Mova a definição para antes do uso.")
            rel.problemas.append(Problema("E021", "erro", mensagem, linha, bloco.nome))
            return
        outros = [t for t, nomes in definidos.items() if chave in nomes and t != tipo]
        dica = f' Existe um {outros[0]} com esse nome; use `{_escopo(outros[0])}("{nome}")`.' if outros else ""
        rel.problemas.append(Problema(
            "E020", "erro", f'{tipo} "{nome}" não foi definido no arquivo.{dica}', linha, bloco.nome,
            tuple(f'{tipo} "{n}"' for n in list(definidos.get(tipo, {}))[:3])))

    def _condicao(self, texto: str, linha: int | None, bloco: Bloco, rel: RelatorioValidacao) -> None:
        if not self.vocabulario.conhece(texto):
            rel.problemas.append(Problema(
                "A001", "aviso", f"Condição fora do vocabulário conhecido: `{texto}`.", linha, bloco.nome,
                tuple(self.vocabulario.sugestoes(texto))))


# ----------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------

def _sem_comentario(linha: str) -> str:
    """Remove `# ...` que esteja fora de aspas."""
    dentro = False
    for indice, caractere in enumerate(linha):
        if caractere == '"':
            dentro = not dentro
        elif caractere == "#" and not dentro:
            return linha[:indice]
    return linha


def _saldo_parenteses(texto: str) -> int:
    saldo, dentro = 0, False
    for caractere in texto:
        if caractere == '"':
            dentro = not dentro
        elif not dentro and caractere == "(":
            saldo += 1
        elif not dentro and caractere == ")":
            saldo -= 1
    return saldo


def _linha_de(bloco: Bloco, trecho: str) -> int:
    alvo = normalizar(trecho)[:40]
    for numero, conteudo in bloco.corpo:
        if alvo and alvo in normalizar(conteudo):
            return numero
    return bloco.linha


def _escopo(tipo: str) -> str:
    return {"time": "do time", "carteira": "da carteira", "segmentacao": "da segmentacao"}[tipo]


def main() -> None:
    argumentos = argparse.ArgumentParser(description="Valida arquivos GOBR.")
    argumentos.add_argument("arquivos", nargs="+", type=Path)
    argumentos.add_argument("--json", action="store_true", help="Saída em JSON.")
    args = argumentos.parse_args()
    validador = GobrValidator()
    algum_erro = False
    for caminho in args.arquivos:
        relatorio = validador.validar_arquivo(caminho)
        algum_erro |= not relatorio.ok
        if args.json:
            print(json.dumps({"arquivo": str(caminho), **relatorio.para_dict()}, ensure_ascii=False, indent=2))
        else:
            print(f"== {caminho}\n{relatorio.como_texto()}\n")
    sys.exit(1 if algum_erro else 0)


if __name__ == "__main__":
    main()
