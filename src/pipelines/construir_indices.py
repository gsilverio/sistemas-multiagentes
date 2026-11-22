"""Gera os índices de recuperação da etapa 3 em `outputs/rag/`.

    outputs/rag/manual/     uma entrada por seção do manual GOBR
    outputs/rag/exemplos/   uma entrada por corpo distinto de regra do acervo

Local ou no Colab. Com sentence-transformers instalado, calcula também os
embeddings (`vetores.npy`); com `--sem-vetores`, só o BM25 (instantâneo).

    python -m src.pipelines.construir_indices
    python -m src.pipelines.construir_indices --explicacoes data/explicacoes/explicacoes_v2.jsonl

Sem `--explicacoes`, usa `data/explicacoes/explicacoes_v2.jsonl` se existir;
senão indexa os rascunhos determinísticos (mesmo conteúdo, redação menos
variada). Rodar de novo depois que as explicações do Colab chegarem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.gobr.extractor import GobrRuleExtractor
from src.rag.busca_hibrida import MODELO_EMBEDDINGS, IndiceHibrido, embeddings_disponiveis
from src.rag.exemplos import carregar_jsonl, descricoes_cliente, documentos_de_exemplos, explicacoes_aprovadas
from src.rag.manual import documentos_do_manual, ler_pdf

RAIZ = Path(__file__).resolve().parents[2]
EXPLICACOES_PADRAO = RAIZ / "data" / "explicacoes" / "explicacoes_v2.jsonl"
DESCRICOES_PADRAO = RAIZ / "data" / "explicacoes" / "descricoes_cliente.jsonl"


def construir(*, pdf: Path, referencias: Path, explicacoes: Path | None, saida: Path, modelo: str,
              com_vetores: bool, descricoes: Path | None = None) -> dict:
    if com_vetores and not embeddings_disponiveis():
        print("sentence-transformers não instalado: gerando só o índice BM25.")
        com_vetores = False

    manual_docs = documentos_do_manual(ler_pdf(pdf), arquivo=pdf.name)
    manual = IndiceHibrido.construir(manual_docs, tipo="gobr_manual_secoes", modelo=modelo, com_vetores=com_vetores)
    m1 = manual.salvar(saida / "manual", source_pdf=pdf.name)

    regras = [
        {"id": b.id, "nome": b.nome, "arquivo_origem": b.arquivo_origem, "gobr": b.gobr,
         "contexto": [{"texto": d.texto} for d in b.contexto]}
        for b in GobrRuleExtractor().extract_directory(referencias)
    ]
    aprovadas = explicacoes_aprovadas(carregar_jsonl(explicacoes)) if explicacoes and explicacoes.exists() else {}
    curtas = descricoes_cliente(carregar_jsonl(descricoes)) if descricoes and descricoes.exists() else {}
    exemplo_docs = documentos_de_exemplos(regras, aprovadas, curtas)
    exemplos = IndiceHibrido.construir(exemplo_docs, tipo="gobr_exemplos", modelo=modelo, com_vetores=com_vetores)
    metodos: dict[str, int] = {}
    for d in exemplo_docs:
        metodos[d["metadata"]["metodo"]] = metodos.get(d["metadata"]["metodo"], 0) + 1
    m2 = exemplos.salvar(saida / "exemplos", blocos=len(regras), metodos=metodos,
                         explicacoes=explicacoes.name if aprovadas else None,
                         descricoes_cliente=sum(bool(d["metadata"]["descricao_cliente"]) for d in exemplo_docs))
    return {"manual": m1, "exemplos": m2}


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--pdf", type=Path, default=RAIZ / "data" / "manual" / "govendas-rules.pdf")
    args.add_argument("--referencias", type=Path, default=RAIZ / "data" / "referencias")
    args.add_argument("--explicacoes", type=Path, default=None,
                      help="explicacoes.jsonl da etapa 2 (padrão: data/explicacoes/explicacoes_v2.jsonl, se existir)")
    args.add_argument("--saida", type=Path, default=RAIZ / "outputs" / "rag")
    args.add_argument("--modelo", default=MODELO_EMBEDDINGS)
    args.add_argument("--sem-vetores", action="store_true")
    a = args.parse_args()
    explicacoes = a.explicacoes or (EXPLICACOES_PADRAO if EXPLICACOES_PADRAO.exists() else None)
    resultado = construir(pdf=a.pdf, referencias=a.referencias, explicacoes=explicacoes, saida=a.saida,
                          modelo=a.modelo, com_vetores=not a.sem_vetores,
                          descricoes=DESCRICOES_PADRAO if DESCRICOES_PADRAO.exists() else None)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
