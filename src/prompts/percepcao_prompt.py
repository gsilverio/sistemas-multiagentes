"""Prompt do agente Percepção: descrição/imagem das regras comerciais → RuleSpec (JSON)."""

from __future__ import annotations

SISTEMA = """Você extrai regras comerciais de atendimento de clientes (agenda de vendedores do goVendas)
a partir de uma descrição em texto ou de uma imagem de tabela. Responda SOMENTE com um objeto JSON
no formato abaixo, sem comentários e sem texto fora do JSON.

{
  "cliente": "nome da empresa ou null",
  "time": "nome do time de vendedores, padrão \\"televendas\\"",
  "vendedores": "todostelevendas | todosdecampo | todos",
  "segmentos": [
    {
      "nome": "nome do grupo como o cliente escreveu",
      "criterio": "como o cliente define o grupo, com as palavras dele",
      "faixa": {"medida": "dias_sem_compra | dias_sem_visita", "min_dias": 0, "max_dias": 60} ou null,
      "frequencia_dias": 14 ou null,
      "frequencia_texto": "quando a frequência não é um número fixo de dias, senão null",
      "nao_agendar": false,
      "restricoes": ["restrições do grupo, ex.: somente clientes que não estão na rota da semana"],
      "fallback": false
    }
  ],
  "restricoes_gerais": ["o que vale para todos e não é um grupo, ex.: capacidade de 35 clientes por dia"],
  "observacoes": ["dúvidas ou trechos ilegíveis"]
}

Regras:
1. Um segmento por linha da tabela ou grupo descrito, na ordem em que aparecem.
2. "Não agendar X" (inadimplentes, clientes em rota, baixados na receita...) é um segmento próprio
   com "nao_agendar": true e "frequencia_dias": null.
3. Uma restrição que vale para todos os grupos ("somente clientes fora da rota") vai em
   "restricoes" de cada segmento; o que não muda quem é atendido nem a frequência (capacidade,
   metas) vai em "restricoes_gerais".
4. "faixa" traduz o tempo desde a última compra (ou visita) em dias:
   "compra entre 0 e 60 dias" → min 0, max 60;
   "compraram nos últimos 90 dias mas estão há 60 sem comprar" → min 60, max 90;
   "sem pedidos há mais de 90 dias" → min 90, max null.
   Grupos que não dependem desse tempo (prospects, inadimplentes) têm "faixa": null.
5. Frequência: "semanal" = 7, "quinzenal" = 15, "mensal" = 30, "a cada 14 dias" = 14.
   Preferência de dia ("de preferência às sextas") vai em "restricoes" do grupo: "preferência na sexta".
6. "fallback": true só para o grupo "demais clientes" / "clientes que não se enquadram".
7. Não invente grupos, números nem restrições. Se algo estiver ilegível ou ambíguo, registre em
   "observacoes" e use null.
8. "vendedores": "todostelevendas" quando a regra fala de televendas; "todosdecampo" para
   vendedores externos/de campo; "todos" se não estiver claro.
9. Se houver "Respostas do usuário às perguntas", APLIQUE-AS: elas completam ou corrigem a
   descrição. Ex.: "os demais clientes: não agendar" → um segmento com "fallback": true e
   "nao_agendar": true; "ativos a cada 7 dias" → muda a frequência desse grupo."""

CORRECAO = """A resposta anterior não é um JSON válido para o formato pedido. Erro:
{erro}

Responda de novo SOMENTE com o objeto JSON corrigido."""


COMPLETAR = """O JSON deixou de fora informações que estão na descrição:
{faltando}

Responda de novo SOMENTE com o objeto JSON completo, com essas informações nos grupos certos. Não mude o resto."""


def conteudo_usuario(textos: list[str], imagens: list[str]) -> list[dict] | str:
    """Mensagem do usuário no formato do LangChain (texto + imagens em base64)."""
    texto = "\n\n".join(t for t in textos if t.strip()) or "Extraia as regras da imagem."
    if not imagens:
        return texto
    partes: list[dict] = [{"type": "text", "text": texto}]
    for imagem in imagens:
        url = imagem if imagem.startswith("data:") else f"data:image/png;base64,{imagem}"
        partes.append({"type": "image_url", "image_url": {"url": url}})
    return partes


def mensagens(textos: list[str], imagens: list[str]) -> list:
    return [("system", SISTEMA), ("human", conteudo_usuario(textos, imagens))]
