"""Fatia o manual do GOBR por seção (construção da linguagem), não por página.

O índice antigo cortava o PDF por página e por tamanho: a seção 6.11 ("Outros
filtros úteis") ficava espalhada em 3 trechos misturados com o fim da 6.10 e
o começo da 6.11.1, e o cabeçalho de página ("6 R e g r a s p a r a...")
entrava em todo trecho. Aqui cada documento é **uma seção numerada** do
manual (ex.: `6.11.1 Opções para tempo`), com:

- o caminho da seção no texto ("6 Regras para filtros de clientes >
  6.11 Outros filtros úteis > 6.11.1 Opções para tempo"), para o BM25 e os
  embeddings saberem do que o trecho trata;
- `construcao`: qual parte da linguagem a seção descreve (time, carteira,
  filtro_cliente, parametros, alerta...), para filtrar a busca;
- as páginas de origem, para citar o manual na resposta.

Seções longas (tabelas grandes, gramática) são divididas em partes nas
quebras de linha, repetindo o caminho em cada parte. A indentação dos
exemplos de código é preservada.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

TAMANHO_MAXIMO = 1600

CONSTRUCAO_POR_CAPITULO = {
    1: "visao_geral",
    2: "visao_geral",
    3: "time",
    4: "carteira_segmentacao",
    5: "filtro_temporal",
    6: "filtro_cliente",
    7: "parametros",
    8: "alerta",
    9: "tag",
    10: "gramatica",
}

_CABECALHO = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")


def _linha_espacada(linha: str) -> bool:
    """Cabeçalho de página que o PDF devolve letra a letra: `6 R e g r a s ...`."""
    tokens = linha.split()
    return len(tokens) >= 5 and sum(len(t) == 1 for t in tokens) / len(tokens) >= 0.55


def _ruido(linha: str) -> bool:
    s = linha.strip()
    return (not s or "©" in s or "R E S E R V E D" in s or set(s) <= {"."} or _linha_espacada(s))


def limpar_linha(linha: str) -> str:
    """Remove espaços repetidos no meio da linha, mas mantém a indentação."""
    linha = linha.replace("�", "").rstrip()
    recuo = len(linha) - len(linha.lstrip())
    return " " * recuo + re.sub(r"\s{2,}", " ", linha.strip())


def _numero(texto: str) -> tuple[int, ...]:
    return tuple(int(p) for p in texto.split("."))


def _sucessor(anterior: tuple[int, ...] | None, novo: tuple[int, ...]) -> bool:
    """Só aceita como título um número que continua a numeração do manual.

    Evita confundir com título uma linha como `5 dias` (exemplo de tempo) ou
    `1 Menor preferência` (tabela de preferências).
    """
    if anterior is None:
        return novo == (1,)
    if novo == anterior + (1,):
        return True
    return any(novo == anterior[: n - 1] + (anterior[n - 1] + 1,) for n in range(len(anterior), 0, -1))


def secoes(paginas: Iterable[tuple[int, str]]) -> list[dict[str, Any]]:
    """`(número da página, texto)` → seções `{numero, titulo, caminho, linhas, pagina}`.

    `linhas` guarda pares `(linha, página)`: cada parte de uma seção longa cita
    só as páginas de onde veio.
    """
    resultado: list[dict[str, Any]] = []
    atual: dict[str, Any] | None = None
    titulos: dict[tuple[int, ...], str] = {}
    for pagina, texto in paginas:
        for bruta in (texto or "").replace("\r", "").splitlines():
            if _ruido(bruta):
                continue
            linha = limpar_linha(bruta)
            m = _CABECALHO.match(linha.strip())
            if m and not linha.startswith(" "):
                numero = _numero(m.group(1))
                if _sucessor(atual["numero"] if atual else None, numero):
                    titulos[numero] = m.group(2).strip()
                    caminho = [f"{'.'.join(map(str, numero[:n]))} {titulos[numero[:n]]}"
                               for n in range(1, len(numero) + 1) if numero[:n] in titulos]
                    atual = {"numero": numero, "titulo": m.group(2).strip(), "caminho": caminho,
                             "linhas": [], "pagina": pagina}
                    resultado.append(atual)
                    continue
            if atual is None:
                continue  # capa e índice
            atual["linhas"].append((linha, pagina))
    return resultado


Linha = tuple[str, int]


def _partes(linhas: list[Linha], limite: int) -> list[list[Linha]]:
    partes: list[list[Linha]] = [[]]
    tamanho = 0
    for linha in linhas:
        if partes[-1] and tamanho + len(linha[0]) + 1 > limite:
            partes.append([])
            tamanho = 0
        partes[-1].append(linha)
        tamanho += len(linha[0]) + 1
    return partes


def documentos_do_manual(paginas: Iterable[tuple[int, str]], *, arquivo: str = "govendas-rules.pdf",
                         tamanho_maximo: int = TAMANHO_MAXIMO) -> list[dict[str, Any]]:
    documentos = []
    for secao in secoes(paginas):
        linhas = [(l, p) for l, p in secao["linhas"] if l.strip()]
        if not linhas:
            continue  # título sem texto próprio (ex.: 4.3, logo seguido de 4.3.1)
        numero = ".".join(map(str, secao["numero"]))
        caminho = " > ".join(secao["caminho"])
        partes = _partes(linhas, tamanho_maximo)
        for indice, parte in enumerate(partes, start=1):
            sufixo = f"_p{indice}" if len(partes) > 1 else ""
            corpo = "\n".join(l for l, _ in parte)
            paginas = {p for _, p in parte} | ({secao["pagina"]} if indice == 1 else set())
            documentos.append({
                "id": f"manual_{numero}{sufixo}",
                "text": f"{caminho}\n{corpo}",
                "metadata": {
                    "source": "gobr_manual",
                    "arquivo_origem": arquivo,
                    "secao": numero,
                    "titulo": secao["titulo"],
                    "caminho": caminho,
                    "capitulo": secao["numero"][0],
                    "construcao": CONSTRUCAO_POR_CAPITULO.get(secao["numero"][0], "outro"),
                    "parte": indice,
                    "partes": len(partes),
                    "paginas": sorted(paginas),
                },
            })
    return documentos


def ler_pdf(pdf: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    return [(n, p.extract_text() or "") for n, p in enumerate(PdfReader(str(pdf)).pages, start=1)]
