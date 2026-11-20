"""Gera `data/vocabulario/condicoes.json` a partir do acervo de referências.

Roda localmente, sem LLM:
    python -m src.pipelines.build_vocabulary

Coleta as condições `quando ...` (e suas continuações `e ...` / `ou ...`) dos
blocos `parametros` e também das definições de `carteira` e `segmentacao`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.gobr.extractor import GobrRuleExtractor
from src.gobr.grammar import normalizar
from src.gobr.parser import GobrParser
from src.gobr.vocabulary import CAMINHO_PADRAO, VocabularioCondicoes


def condicoes_de_definicao(texto: str) -> list[str]:
    """Condições dentro de `carteira`/`segmentacao` (o parser só lê parametros)."""
    condicoes: list[str] = []
    em_condicao = False
    for linha in GobrParser._linhas_logicas(texto)[1:]:  # Ignora o cabeçalho.
        normalizada = normalizar(linha)
        if normalizada.startswith("quando "):
            condicoes.append(linha.strip()[len("quando "):])
            em_condicao = True
        elif em_condicao and normalizada.startswith(("e ", "ou ")):
            condicoes.append(linha.strip().split(" ", 1)[1])
        else:
            em_condicao = False
    return condicoes


def construir(diretorio: Path) -> VocabularioCondicoes:
    extrator = GobrRuleExtractor()
    parser = GobrParser()
    vocabulario = VocabularioCondicoes()
    definicoes_vistas: set[tuple[str, int]] = set()
    for bloco in extrator.extract_directory(diretorio):
        for condicao in parser.parse(bloco.gobr).condicoes:
            vocabulario.registrar(condicao.texto)
        for definicao in bloco.contexto:
            chave = (bloco.arquivo_origem, definicao.linha_inicio)
            if definicao.tipo == "time" or chave in definicoes_vistas:
                continue
            definicoes_vistas.add(chave)
            for condicao in condicoes_de_definicao(definicao.texto):
                vocabulario.registrar(condicao)
    return vocabulario


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--input", type=Path, default=Path("data/referencias"))
    argumentos.add_argument("--output", type=Path, default=CAMINHO_PADRAO)
    args = argumentos.parse_args()
    vocabulario = construir(args.input)
    vocabulario.salvar(args.output)
    total = sum(e.ocorrencias for e in vocabulario.entradas.values())
    print(f"Condicoes lidas: {total}")
    print(f"Modelos distintos: {len(vocabulario.entradas)}")
    print(f"Arquivo gerado: {args.output.resolve()}")


if __name__ == "__main__":
    main()
