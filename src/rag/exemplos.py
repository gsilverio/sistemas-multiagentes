"""Documentos do índice de exemplos: regras do acervo com sua explicação.

Uso no agente: o usuário descreve um segmento ("clientes inativos há 90 dias,
a cada 14 dias, fora da rota"); o recuperador acha regras do acervo com
descrição parecida e entrega o GOBR delas ao gerador como exemplo (few-shot).

Cada documento é um **corpo distinto** de bloco `parametros` (1.204 dos
1.700 blocos; os repetidos só mudam o nome e ficam em `ids_iguais`).

Texto indexado:
- `text` (embeddings): nome da regra + explicação em linguagem comercial.
  Usa a explicação do LLM quando ela passou no guard; senão, o rascunho
  determinístico. Os dois dizem o mesmo (o guard garante), então o índice
  não fica "errado" com o rascunho, só menos variado na redação.
- `texto_lexico` (BM25): o mesmo texto + o corpo GOBR, para que termos da
  DSL citados pelo usuário ("roteirizavel", "nao agendar") também batam.

Blocos com cláusula que o glossário não traduz ficam de fora: não servem de
exemplo confiável.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.explicacao.rascunho import montar_rascunho
from src.gobr.assinatura import assinatura_estrutural, chave_corpo, construcoes
from src.gobr.parser import GobrParser


def carregar_jsonl(caminho: Path) -> list[dict[str, Any]]:
    with caminho.open(encoding="utf-8") as origem:
        return [json.loads(linha) for linha in origem if linha.strip()]


def explicacoes_aprovadas(registros: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`explicacoes.jsonl` da etapa 2 → {id: {texto, metodo}} só das aprovadas pelo LLM."""
    aprovadas = {}
    for r in registros:
        texto = r.get("explicacao")
        if isinstance(texto, dict):
            texto = texto.get("texto")
        if r.get("aprovada") and r.get("metodo") in ("llm", "llm_corrigido") and texto:
            aprovadas[r["id"]] = {"texto": texto.strip(), "metodo": r["metodo"]}
    return aprovadas


def _corpo(gobr: str) -> str:
    return "\n".join(gobr.strip().splitlines()[1:])


def documentos_de_exemplos(regras: Iterable[dict[str, Any]],
                           explicacoes: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    explicacoes = explicacoes or {}
    parser = GobrParser()
    por_corpo: dict[str, dict[str, Any]] = {}
    for regra in regras:
        chave = chave_corpo(regra["gobr"])
        if chave in por_corpo:
            doc = por_corpo[chave]
            doc["metadata"]["ids_iguais"].append(regra["id"])
            # Prefere a explicação do LLM, se o primeiro bloco não tinha.
            if doc["metadata"]["metodo"] == "rascunho" and regra["id"] in explicacoes:
                _aplicar_explicacao(doc, explicacoes[regra["id"]])
            continue
        ast = parser.parse(regra["gobr"])
        rascunho = montar_rascunho(ast)
        if rascunho.desconhecidos:
            continue
        doc = {
            "id": regra["id"],
            "text": "",
            "texto_lexico": "",
            "metadata": {
                "id": regra["id"],
                "nome": regra.get("nome") or ast.nome,
                "arquivo_origem": regra.get("arquivo_origem"),
                "gobr": regra["gobr"],
                "contexto": [d["texto"] for d in regra.get("contexto") or []],
                "rascunho": rascunho.texto,
                "explicacao": rascunho.texto,
                "metodo": "rascunho",
                "assinatura": assinatura_estrutural(ast),
                "construcoes": construcoes(ast),
                "ids_iguais": [],
            },
        }
        if regra["id"] in explicacoes:
            _aplicar_explicacao(doc, explicacoes[regra["id"]])
        else:
            _atualizar_textos(doc)
        por_corpo[chave] = doc
    return list(por_corpo.values())


def _aplicar_explicacao(doc: dict[str, Any], explicacao: dict[str, Any]) -> None:
    doc["metadata"]["explicacao"] = explicacao["texto"]
    doc["metadata"]["metodo"] = explicacao["metodo"]
    _atualizar_textos(doc)


def _atualizar_textos(doc: dict[str, Any]) -> None:
    m = doc["metadata"]
    doc["text"] = f'{m["nome"]}. {m["explicacao"]}'
    extra = f'\n{m["rascunho"]}' if m["explicacao"] != m["rascunho"] else ""
    doc["texto_lexico"] = f'{doc["text"]}{extra}\n{_corpo(m["gobr"])}'
