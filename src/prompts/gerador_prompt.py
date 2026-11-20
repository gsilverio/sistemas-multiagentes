"""Prompt do agente Gerador: descrição de um segmento → um bloco `parametros` GOBR.

A mensagem do usuário tem sempre o mesmo formato (`pedido()`), que é também
o formato dos pares de fine-tuning da etapa 6: o modelo treinado recebe
exatamente o que o modelo base recebe aqui.
"""

from __future__ import annotations

SISTEMA = """Você escreve UM bloco `parametros` da linguagem GOBR (goVendas) a partir da descrição
comercial de um grupo de clientes. Responda SOMENTE com o bloco, sem explicações e sem ```.

Formato:
parametros "<nome da regra>":
    <escopo>                      # ex.: do time("televendas")
    quando <condição>             # opcional; várias condições ligadas por "e" / "ou"
    frequencia de <N> dias        # ou: nao agendar  (nunca os dois)
    <modificadores opcionais>     # ex.: preferencia na segunda, sem choque no time("campo") na rota

Regras:
1. Use exatamente o nome e o escopo pedidos.
2. Indente o corpo com 4 espaços. Não escreva `version`, `time` nem outros blocos.
3. Use só construções do GOBR que aparecem nos exemplos ou no manual abaixo. Não invente palavras-chave.
4. Recência: "comprou nos últimos 60 dias" → `recencia de positivacao em 60 dias`;
   "há mais de 90 dias sem comprar" → `fora da recencia de positivacao em 90 dias`.
   Os blocos são avaliados em ordem e o cliente fica com o primeiro que o atende; por isso uma
   faixa "entre 60 e 90 dias" que vem depois da faixa de 60 pode ser só `recencia de positivacao em 90 dias`.
5. Todo número da descrição precisa estar no bloco, e nenhum número novo.{manual}"""

CORRECAO = """O bloco abaixo tem problemas:
{erros}

Bloco:
{bloco}

Escreva o bloco corrigido, SOMENTE o bloco."""


def pedido(nome: str, escopo: str, descricao: str) -> str:
    return f'Nome da regra: "{nome}"\nEscopo: {escopo}\nDescrição: {descricao}'


def mensagens(nome: str, escopo: str, descricao: str, *, exemplos: list[tuple[str, str, str, str]] = (),
              manual: list[tuple[str, str]] = ()) -> list:
    """`exemplos`: (nome, escopo, descrição, bloco GOBR); `manual`: (título, texto)."""
    trechos = "".join(f"\n\n### Manual: {titulo}\n{texto}" for titulo, texto in manual)
    msgs: list = [("system", SISTEMA.format(manual=("\n\nTrechos do manual:" + trechos) if trechos else ""))]
    for ex_nome, ex_escopo, ex_descricao, ex_bloco in exemplos:
        msgs.append(("human", pedido(ex_nome, ex_escopo, ex_descricao)))
        msgs.append(("ai", ex_bloco.strip()))
    msgs.append(("human", pedido(nome, escopo, descricao)))
    return msgs


def mensagens_correcao(base: list, bloco: str, erros: list[str]) -> list:
    return [*base, ("ai", bloco), ("human", CORRECAO.format(erros="\n".join(f"- {e}" for e in erros), bloco=bloco))]
