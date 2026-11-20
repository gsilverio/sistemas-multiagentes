"""Pares de treino e teste para o fine-tuning do Gerador: descrição → bloco GOBR.

Cada par usa exatamente o formato que o Gerador recebe no grafo
(`gerador_prompt.pedido`: nome, escopo e descrição), então o modelo treinado
entra no lugar do modelo base sem mudar nada no agente.

- **Descrição**: a explicação do bloco. No treino, quando há explicação do
  LLM aprovada pelo guard, entram dois pares (texto do LLM e rascunho
  determinístico): mesma resposta, redações diferentes.
- **Resposta**: o bloco do acervo, normalizado (cabeçalho com o nome,
  corpo com 4 espaços, sem linhas de comentário).
- **Separação treino/teste por forma de regra** (`assinatura_estrutural`):
  ~10% das formas ficam só no teste. Um bloco de teste nunca tem "irmão" no
  treino que só difere nos números, então o teste mede generalização, não
  memorização. Blocos repetidos (mesmo corpo) aparecem uma vez.
- **Exemplos reais** (`data/exemplos/*.gobr`, fora do acervo) formam um
  terceiro conjunto, só de avaliação.
- **Exemplos no prompt**: metade dos pares de treino leva 2 exemplos
  recuperados do próprio treino (como o Recuperador faz no grafo), para o
  modelo funcionar com e sem RAG.

    python -m src.finetuning.dataset --explicacoes data/explicacoes/explicacoes_v2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.agents.gerador import escopo_de
from src.explicacao.rascunho import montar_rascunho
from src.gobr.assinatura import assinatura_estrutural, chave_corpo
from src.gobr.extractor import GobrRuleExtractor
from src.gobr.parser import GobrParser
from src.rag.busca_hibrida import IndiceHibrido
from src.rag.exemplos import carregar_jsonl, explicacoes_aprovadas
from src.validation.gobr_validator import GobrValidator

RAIZ = Path(__file__).resolve().parents[2]
SAIDA_PADRAO = RAIZ / "outputs" / "finetuning" / "dataset"
PERCENTUAL_TESTE = 10
FRACAO_COM_EXEMPLOS = 0.5
K_EXEMPLOS = 2


def bloco_normalizado(gobr: str) -> str:
    linhas = [l.strip() for l in gobr.strip().splitlines()]
    corpo = [l for l in linhas[1:] if l and not l.startswith("#")]
    return "\n".join([linhas[0], *("    " + l for l in corpo)])


def split_da_assinatura(assinatura: str, percentual_teste: int = PERCENTUAL_TESTE) -> str:
    h = int(hashlib.sha256(assinatura.encode("utf-8")).hexdigest(), 16)
    return "teste" if h % 100 < percentual_teste else "treino"


def _par(bloco, *, nome: str, descricao: str, fonte: str, split: str, assinatura: str, contexto: list[str],
         corpo: str) -> dict[str, Any]:
    alvo = bloco_normalizado(bloco.gobr)
    return {
        "id": bloco.id, "split": split, "nome": nome, "escopo": escopo_de(alvo), "descricao": descricao,
        "fonte_descricao": fonte, "bloco": alvo, "contexto": contexto, "assinatura": assinatura,
        "chave_corpo": corpo, "exemplos": [],
    }


CODIGOS_DE_REFERENCIA = {"E007", "E020", "E021"}   # dependem de definições fora do bloco


def bloco_valido(bloco: str, contexto: list[str], validador: GobrValidator) -> bool:
    arquivo = "\n\n".join(["version 1.0", *contexto, bloco]) + "\n"
    return not [p for p in validador.validar(arquivo).erros if p.codigo not in CODIGOS_DE_REFERENCIA]


def pares_do_acervo(blocos: Iterable, explicacoes: dict[str, dict] | None = None, *,
                    split_fixo: str | None = None, percentual_teste: int = PERCENTUAL_TESTE) -> list[dict[str, Any]]:
    """Um par por corpo distinto; ficam de fora blocos com cláusula sem tradução ou GOBR inválido."""
    explicacoes = explicacoes or {}
    parser, validador = GobrParser(), GobrValidator()
    vistos: set[str] = set()
    pares: list[dict[str, Any]] = []
    for bloco in blocos:
        corpo = chave_corpo(bloco.gobr)
        if corpo in vistos:
            continue
        vistos.add(corpo)
        ast = parser.parse(bloco.gobr)
        rascunho = montar_rascunho(ast)
        if rascunho.desconhecidos:
            continue
        if not bloco_valido(bloco_normalizado(bloco.gobr), [d.texto for d in bloco.contexto], validador):
            continue  # o modelo não aprende com GOBR quebrado
        assinatura = assinatura_estrutural(ast)
        split = split_fixo or split_da_assinatura(assinatura, percentual_teste)
        comum = dict(nome=bloco.nome, split=split, assinatura=assinatura, corpo=corpo,
                     contexto=[d.texto for d in bloco.contexto])
        llm = explicacoes.get(bloco.id)
        if split == "treino":
            pares.append(_par(bloco, descricao=rascunho.texto, fonte="rascunho", **comum))
            if llm and llm["texto"] != rascunho.texto:
                pares.append(_par(bloco, descricao=llm["texto"], fonte=llm["metodo"], **comum))
        else:
            descricao, fonte = (llm["texto"], llm["metodo"]) if llm else (rascunho.texto, "rascunho")
            pares.append(_par(bloco, descricao=descricao, fonte=fonte, **comum))
    return pares


def indice_de_exemplos(treino: list[dict[str, Any]]) -> IndiceHibrido:
    """BM25 sobre as descrições do treino: fonte dos exemplos do prompt (treino e avaliação)."""
    docs = [{"id": f"{p['id']}#{i}", "text": f"{p['nome']}. {p['descricao']}", "metadata": p}
            for i, p in enumerate(treino)]
    return IndiceHibrido(docs, tipo="exemplos_finetuning")


def exemplos_para(par: dict[str, Any], indice: IndiceHibrido, k: int = K_EXEMPLOS) -> list[list[str]]:
    """k exemplos parecidos, nunca do mesmo corpo nem da mesma forma de regra do par."""
    resultados = indice.buscar(f"{par['nome']}. {par['descricao']}", k=k, modo="bm25",
                               filtro=lambda d: d["metadata"]["chave_corpo"] != par["chave_corpo"]
                               and d["metadata"]["assinatura"] != par["assinatura"])
    return [[r["metadata"]["nome"], r["metadata"]["escopo"], r["metadata"]["descricao"], r["metadata"]["bloco"]]
            for r in resultados]


def montar(referencias: Path, exemplos_reais: Path, explicacoes: Path | None, *, seed: int = 13,
           percentual_teste: int = PERCENTUAL_TESTE) -> dict[str, list[dict[str, Any]]]:
    aprovadas = explicacoes_aprovadas(carregar_jsonl(explicacoes)) if explicacoes and explicacoes.exists() else {}
    extrator = GobrRuleExtractor()
    pares = pares_do_acervo(extrator.extract_directory(referencias), aprovadas, percentual_teste=percentual_teste)
    reais = pares_do_acervo(extrator.extract_directory(exemplos_reais), split_fixo="real")
    treino = [p for p in pares if p["split"] == "treino"]
    teste = [p for p in pares if p["split"] == "teste"]

    indice = indice_de_exemplos(treino)
    aleatorio = random.Random(seed)
    for par in treino:
        if aleatorio.random() < FRACAO_COM_EXEMPLOS:
            par["exemplos"] = exemplos_para(par, indice)
    aleatorio.shuffle(treino)
    return {"treino": treino, "teste": teste, "real": reais}


def salvar(conjuntos: dict[str, list[dict[str, Any]]], saida: Path, **extra: Any) -> dict[str, Any]:
    saida.mkdir(parents=True, exist_ok=True)
    for nome, pares in conjuntos.items():
        with (saida / f"{nome}.jsonl").open("w", encoding="utf-8") as destino:
            for par in pares:
                destino.write(json.dumps(par, ensure_ascii=False) + "\n")
    resumo = {
        nome: {"pares": len(pares), "formas": len({p["assinatura"] for p in pares}),
               "fontes": dict(Counter(p["fonte_descricao"] for p in pares)),
               "com_exemplos": sum(bool(p["exemplos"]) for p in pares)}
        for nome, pares in conjuntos.items()
    }
    formas_treino = {p["assinatura"] for p in conjuntos["treino"]}
    resumo["vazamento_formas_teste_no_treino"] = len({p["assinatura"] for p in conjuntos["teste"]} & formas_treino)
    resumo.update(extra)
    (saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resumo


def carregar(diretorio: Path, nome: str) -> list[dict[str, Any]]:
    return carregar_jsonl(diretorio / f"{nome}.jsonl")


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--referencias", type=Path, default=RAIZ / "data" / "referencias")
    args.add_argument("--exemplos-reais", type=Path, default=RAIZ / "data" / "exemplos")
    args.add_argument("--explicacoes", type=Path, default=None,
                      help="explicacoes.jsonl da etapa 2 (padrão: data/explicacoes/explicacoes_v2.jsonl, se existir)")
    args.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    a = args.parse_args()
    explicacoes = a.explicacoes or RAIZ / "data" / "explicacoes" / "explicacoes_v2.jsonl"
    conjuntos = montar(a.referencias, a.exemplos_reais, explicacoes)
    resumo = salvar(conjuntos, a.saida, explicacoes=str(explicacoes) if explicacoes.exists() else None)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
