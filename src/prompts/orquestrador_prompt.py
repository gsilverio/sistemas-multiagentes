"""Prompt do Orquestrador: escolhe o próximo agente a partir de um resumo do estado."""

from __future__ import annotations

import json

DESCRICOES = {
    "percepcao": "lê a descrição/imagem do cliente e extrai os grupos de clientes (RuleSpec)",
    "analista": "procura falhas de modelagem no RuleSpec (buracos entre faixas, ordem, falta de frequência)",
    "perguntar_usuario": "pausa e pergunta ao usuário o que está ambíguo ou faltando",
    "recuperador": "busca exemplos de regras parecidas e trechos do manual GOBR",
    "gerador": "escreve (ou corrige) os blocos GOBR",
    "validador": "confere a sintaxe do GOBR e se ele corresponde ao RuleSpec",
    "explicador": "explica em português o que cada bloco GOBR faz",
    "finalizar": "monta a resposta final para o usuário",
}

SISTEMA = """Você coordena uma equipe de agentes que transforma regras comerciais em GOBR.
Recebe um resumo do estado (sem o conteúdo produzido pelos agentes) e a lista de agentes que
podem ser chamados agora. Escolha UM deles. Responda SOMENTE com JSON:
{"proximo": "<nome do agente>", "motivo": "<uma frase>"}

Critérios: perguntas bloqueantes precisam de resposta antes de gerar; perguntas não bloqueantes
podem ser feitas ou assumidas (a suposição fica registrada); se a validação falhou e ainda há
tentativas, o gerador corrige."""


def mensagens(resumo: dict, permitidos: list[str]) -> list:
    opcoes = "\n".join(f"- {n}: {DESCRICOES[n]}" for n in permitidos)
    return [("system", SISTEMA),
            ("human", f"Estado:\n{json.dumps(resumo, ensure_ascii=False, indent=1)}\n\nAgentes disponíveis agora:\n{opcoes}")]
