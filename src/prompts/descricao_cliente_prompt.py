"""Prompt da descrição no estilo das tabelas dos clientes (dica do enunciado).

As imagens de `data/exemplos` descrevem cada grupo numa linha curta:
"Clientes com compra entre 0 e 60 dias | 14 dias | Somente clientes fora da rota".
O modelo reescreve cada bloco do acervo nesse estilo, a partir do GOBR e do
significado (rascunho determinístico). Os exemplos abaixo vêm das próprias
tabelas reais.
"""

from __future__ import annotations

SISTEMA = """Você reescreve uma regra de atendimento de clientes no estilo das tabelas que os clientes
enviam: uma ou duas frases curtas, como uma linha de planilha comercial.

Formato: "<quem são os clientes>. <o que fazer>." Exemplos de "o que fazer": "Atender a cada 14
dias.", "Não agendar.", "Atender com a frequência do vendedor.", "Atender a cada 30 dias, de
preferência na segunda-feira."

Regras:
1. Use a linguagem do dia a dia comercial ("clientes com compra nos últimos 60 dias",
   "clientes sem pedidos há mais de 90 dias"), não a da documentação.
2. Mantenha TODOS os números e TODOS os valores entre aspas das condições e modificadores, entre
   aspas, e não invente outros (ex.: sem choque no time("campo") → 'sem choque com o time "campo"').
3. Não cite o nome do time ou da carteira do escopo (a primeira linha da regra) nem o nome da regra.
4. "nao agendar" vira "Não agendar ..."; nunca escreva "não agendar" se a regra tem frequência.
5. Não explique o motivo da regra. Responda só com a descrição."""


EXEMPLOS = [
    ('parametros "x":\n    do time("t")\n    quando recencia de positivacao em 60 dias\n    frequencia de 14 dias',
     "Clientes com compra nos últimos 60 dias. Atender a cada 14 dias."),
    ('parametros "x":\n    do time("t")\n    quando recencia de positivacao em 90 dias\n'
     "    e fora da recencia de positivacao em 60 dias\n    frequencia de 7 dias",
     "Clientes que compraram nos últimos 90 dias, mas estão há 60 dias sem comprar. Atender a cada 7 dias."),
    ('parametros "x":\n    do time("t")\n    quando fora da recencia de positivacao em 45 dias\n    frequencia de 14 dias',
     "Clientes sem compra há mais de 45 dias. Atender a cada 14 dias."),
    ('parametros "x":\n    do time("t")\n    quando no ciclo de vida prospect com a empresa\n    frequencia de 7 dias',
     "Clientes com ciclo de vida prospect na empresa. Atender a cada 7 dias."),
    ('parametros "x":\n    do time("t")\n    quando status na receita federal definido como baixada\n    nao agendar',
     "Não agendar clientes baixados na Receita Federal."),
    ('parametros "x":\n    da carteira("c")\n    quando ramo de atividade igual("MATCON", "FERRAGISTA")\n'
     "    frequencia de 30 dias\n    preferencia na segunda",
     'Clientes do ramo de atividade "MATCON" ou "FERRAGISTA". Atender a cada 30 dias, de preferência na segunda-feira.'),
]

CORRECAO = """A descrição tem problemas:
{problemas}

Escreva a descrição corrigida, só ela."""


def pedido(gobr: str, significado: str) -> str:
    corpo = "\n".join(gobr.strip().splitlines()[1:])
    return f"Regra:\n{corpo}\n\nSignificado completo:\n{significado}"


def mensagens(gobr: str, significado: str, exemplos: list[tuple[str, str, str]]) -> list:
    """`exemplos`: (gobr, significado, descrição) — o significado vem do rascunho de cada exemplo."""
    msgs: list = [("system", SISTEMA)]
    for ex_gobr, ex_significado, ex_descricao in exemplos:
        msgs += [("human", pedido(ex_gobr, ex_significado)), ("ai", ex_descricao)]
    msgs.append(("human", pedido(gobr, significado)))
    return msgs
