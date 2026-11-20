"""Executa o grafo pela linha de comando (precisa do Ollama: no Colab, ver o notebook).

    python -m src.grafo.executar --arquivo data/exemplos/RegraMaxlub.pdf
    python -m src.grafo.executar --texto "Ativos: compra até 60 dias, a cada 14 dias ..."
    python -m src.grafo.executar --arquivo tabela.png --interativo   # responde às perguntas no terminal

Sem `--interativo`, o grafo não pausa: perguntas não bloqueantes viram
suposições registradas na resposta.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.percepcao import ler_anexo
from src.grafo.estado import Agentes
from src.grafo.grafo import config_thread, construir_grafo, entrada


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--texto", default="")
    args.add_argument("--arquivo", type=Path)
    args.add_argument("--interativo", action="store_true")
    args.add_argument("--orquestrador-sem-llm", action="store_true", help="só as regras determinísticas")
    args.add_argument("--explicar-com-llm", action="store_true")
    args.add_argument("--trace", action="store_true", help="mostra as decisões do orquestrador")
    a = args.parse_args()

    texto, imagens = a.texto, []
    if a.arquivo:
        conteudo, imagens = ler_anexo(a.arquivo)
        texto = f"{texto}\n\n{conteudo}".strip()
    agentes = Agentes.padrao(orquestrador_com_llm=not a.orquestrador_sem_llm, explicar_com_llm=a.explicar_com_llm,
                             interativo=a.interativo)
    grafo = construir_grafo(agentes, checkpointer=InMemorySaver())
    config = config_thread(uuid.uuid4().hex)
    saida = grafo.invoke(entrada(texto, imagens), config)
    while "__interrupt__" in saida:
        print(saida["__interrupt__"][0].value["mensagem"])
        saida = grafo.invoke(Command(resume=input("\nSua resposta (Enter para seguir com as suposições): ")), config)
    if a.trace:
        for passo in saida.get("trace", []):
            print(json.dumps(passo, ensure_ascii=False))
    print(saida.get("resposta"))


if __name__ == "__main__":
    main()
