"""Prompt do agente Explicador (uma chamada ao LLM por bloco).

Sem dependência de LangChain: as mensagens são tuplas (papel, texto), formato
aceito por `ChatOllama.invoke`.
"""

from __future__ import annotations

from src.explicacao.rascunho import Rascunho

SISTEMA = """Você é o Explicador de regras comerciais do Govendas.

Você recebe uma regra de parametrização de atendimento escrita na linguagem
de regras do Govendas, a lista de fatos que ela contém (já traduzidos) e um
rascunho de explicação. Reescreva o rascunho como um texto curto e natural
para um gestor comercial, em português do Brasil.

Regras obrigatórias:
1. Mantenha TODOS os fatos da lista, sem acrescentar nenhum outro.
2. Copie os nomes entre aspas exatamente como estão (times, carteiras,
   segmentações, ramos, cidades), com as aspas.
3. Mantenha todos os números em algarismos e não crie números novos.
4. Mantenha "e" e "ou" entre as condições: "ou" significa alternativa.
5. "Não devem ser agendados" é uma proibição de agendar o cliente; não diga
   que o cliente já está ou não pode estar agendado.
6. Filtros de execução (quando a regra é aplicada) são diferentes de
   preferência de dias de visita; não misture os dois.
7. Não mencione a linguagem de regras, código, JSON nem "fatos".
8. Responda somente com a explicação, em 1 a 4 frases, sem título e sem listas."""


def _fatos(rascunho: Rascunho) -> str:
    linhas = []
    for fato in rascunho.fatos:
        if fato.tipo == "escopo":
            linhas.append(f"- Público: {fato.significado}")
        else:
            linhas.append(f"- {fato.gobr.replace(chr(10), ' ')} → {fato.significado}")
    return "\n".join(linhas)


def mensagens(gobr: str, rascunho: Rascunho) -> list[tuple[str, str]]:
    humano = (
        f"Regra:\n{gobr.strip()}\n\n"
        f"Fatos:\n{_fatos(rascunho)}\n\n"
        f"Rascunho:\n{rascunho.texto}\n\n"
        "Reescreva o rascunho."
    )
    return [("system", SISTEMA), ("human", humano)]


def mensagens_correcao(gobr: str, rascunho: Rascunho, anterior: str, problemas: str) -> list[tuple[str, str]]:
    base = mensagens(gobr, rascunho)
    return base + [
        ("ai", anterior),
        ("human", "A explicação acima foi rejeitada pela conferência automática:\n"
                  f"{problemas}\n\nEscreva de novo, corrigindo esses pontos e mantendo todo o resto."),
    ]
