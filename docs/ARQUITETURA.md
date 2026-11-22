# Arquitetura multiagente GOBR

Objetivo (ver [`enunciado.pdf`](enunciado.pdf)): um agente conversacional que recebe uma
imagem ou descrição das regras comerciais de um cliente, aponta falhas na modelagem
e devolve o conjunto de regras na DSL **GOBR** do goVendas.

Restrição do projeto: **somente modelos open source** (servidos pelo Ollama, localmente
ou no Colab).

## Visão geral

```
   START ─► inicio ─► ┌────────────────────┐
                      │   ORQUESTRADOR     │◄──────────────────────────────┐
                      │ (guardrails + LLM) │                               │
                      └─────────┬──────────┘                               │
                Command(goto=?) │                                          │
   ┌──────────┬──────────┬──────┴─────┬─────────┬──────────┬───────────┐   │
   ▼          ▼          ▼            ▼         ▼          ▼           ▼   │
percepcao  analista  perguntar_   recuperador gerador  validador  explicador
  (LLM)    (regras)   usuario      (RAG)      (LLM)    (regras)   (rascunho/LLM)
                    (interrupt)                                            │
   └──────────┴──────────┴────────────┴─────────┴──────────┴───────────┴───┘
                         todos voltam ao orquestrador
                                  │
                              finalizar ──► END
```

Código em `src/grafo/` (grafo, estado, orquestrador, resposta final) e `src/agents/`
(um módulo por agente, contratos em `contratos.py`). O grafo é um **supervisor em
estrela** no LangGraph:

- **Guardrails em código** (`destinos_permitidos`) dizem quais agentes podem ser chamados
  em cada momento: só gera GOBR com contexto recuperado e sem pergunta bloqueante aberta;
  só explica depois da validação; só finaliza com explicação pronta, erro ou impasse.
- O **orquestrador LLM** escolhe entre os destinos permitidos a partir de um *resumo* do
  estado (contagens e flags, sem o conteúdo produzido pelos agentes) e responde
  `{"proximo", "motivo"}`. Só é chamado quando há mais de uma opção válida; a principal é
  "perguntar ao usuário ou seguir com as suposições". Escolha inválida ou resposta ilegível
  → vale a regra, e o `trace` registra `origem: "guardrail"`.
- Cada agente é um **nó isolado** com entrada e saída tipadas (Pydantic): o nó monta a
  entrada a partir do estado e devolve só os campos que produz. O estado guarda tipos
  simples (dict/list/str) para o checkpointer serializar.
- **Verificação cega**: o explicador recebe apenas o GOBR (não a descrição original); o
  analista recebe apenas o `RuleSpec` (não o GOBR). Quem verifica não é influenciado por
  quem produziu.
- **Laço de correção**: o validador agrupa os erros por bloco; o gerador refaz só os blocos
  com erro, recebendo os erros (até 3 gerações).
- **Memória**: checkpointer por `thread_id` (`SqliteSaver` na interface, `InMemorySaver`
  nos testes). Uma mensagem nova na mesma conversa (ex.: "mude a frequência dos ativos")
  passa por `inicio`, que zera os campos derivados, e a Percepção relê o histórico todo.
  `interrupt()` pausa o grafo para perguntas; retoma-se com `Command(resume=...)`.

## Agentes

| Agente | Entrada | Saída | Modelo |
|---|---|---|---|
| Orquestrador | resumo do estado + destinos permitidos | próximo nó + motivo | `GOBR_MODELO_ORQUESTRADOR` (padrão: o mesmo modelo) |
| Percepção | imagem / PDF / texto + respostas do usuário | `RuleSpec` (segmentos: nome, critério, faixa de dias, frequência, restrições) | gemma3:12b (visão) |
| Analista de negócio | `RuleSpec` | falhas de modelagem, perguntas, ajustes (reordenação) e `RuleSpec` ajustado | determinístico |
| Recuperador | cada segmento | exemplos similares + trechos do manual | busca híbrida (BM25 + denso) |
| Gerador GOBR | um segmento + contexto RAG (+ erros, na correção) | um bloco `parametros`; o arquivo é montado em código | `GOBR_MODELO_GERADOR`, depois fine-tuning QLoRA |
| Validador | GOBR + `RuleSpec` | erros por bloco (sintaxe + conferência com o `RuleSpec`) | determinístico |
| Explicador | GOBR | explicação comercial | gemma3:12b / qwen3:8b |

## Estrutura do código

Descrição de cada pasta, módulo e dos códigos de erro do validador:
[`ESTRUTURA_DO_CODIGO.md`](ESTRUTURA_DO_CODIGO.md).

## Etapas

| # | Etapa | Resultado | Status |
|---|---|---|---|
| 0 | Limpeza e reestruturação (tag `entregas-1-a-4` preserva o estado anterior) | repositório enxuto, testes passando | ✅ |
| 1 | Validador GOBR completo (arquivo inteiro, referências, vocabulário de condições) | `GobrValidator.validar(gobr)` com erros acionáveis | ✅ |
| 2 | Explicador novo (AST → rascunho determinístico → 1 chamada LLM → guard) e geração por bloco | 1.698 explicações com `gemma3:12b` (1.466 de primeira, 182 após correção, 50 no rascunho), 100% aprovadas pelo guard: `data/explicacoes/explicacoes_v2.jsonl` | ✅ |
| 3 | RAG híbrido, re-chunking do manual por construção da linguagem | `buscar_exemplos()` / `buscar_manual()`; índices com rascunhos, refazer com as explicações do LLM | ✅ |
| 4 | Grafo multiagente com orquestrador supervisor | fluxo ponta a ponta (testado com LLM falso; falta rodar no Colab) | ⏳ |
| 5 | Memória de contexto + interface Gradio | `SqliteSaver` por conversa, chat com anexos, progresso e retomada (testado com LLM falso; falta rodar no Colab) | ⏳ |
| 6 | Fine-tuning QLoRA do gerador + avaliação comparativa | código ✅ (dataset, treino, avaliação, plugue no agente); treino ⏳ no Colab | ⏳ |

## Linha de base

`data/avaliacao/explicacoes_baseline_v1.jsonl` guarda as 59 explicações geradas pelo
pipeline antigo (Interpretador → Explicador, `gemma3:12b`, regras 001 a 015). Elas **não**
são usadas como verdade: servem para comparar com o explicador novo da etapa 2. Problemas
conhecidos dessa versão: `frequencia de inativos da empresa` interpretada como "atendimento
para inativos" (é a frequência configurada para inativos), frequência inventada
(`regra_005_003` cita "14 dias" que não existe na regra) e 10 casos que caíram no
fallback determinístico.

O guard da etapa 2 aprova só **20 das 59** (o guard antigo aprovava 49): 33 reprovações
por frequência mal explicada, 11 por número inventado e 7 por condição perdida. Para
reproduzir: `python -m src.pipelines.avaliar_explicacoes data/avaliacao/explicacoes_baseline_v1.jsonl`.

## Recuperação (etapa 3)

O manual é indexado **por seção numerada** (98 documentos) em vez de por página (64
trechos que misturavam seções e repetiam o cabeçalho da página). Os exemplos são os
1.202 corpos distintos de regra do acervo, com GOBR e explicação. A busca funde BM25 e
embeddings (`intfloat/multilingual-e5-small`) por Reciprocal Rank Fusion.

Medido com `python -m src.pipelines.avaliar_recuperacao` (30 consultas ao manual, 18 de
exemplos, parte delas tiradas das descrições reais de clientes em `data/exemplos`):

| Índice | Modo | acerto@1 | acerto@k | MRR |
|---|---|---|---|---|
| manual (k=4) | BM25 | 0,533 | 0,867 | 0,681 |
| | denso | 0,533 | 0,767 | 0,633 |
| | **híbrido** | **0,667** | **0,867** | **0,756** |
| exemplos (k=3) | BM25 | 0,722 | 0,889 | 0,796 |
| | denso | 0,722 | 0,833 | 0,778 |
| | **híbrido** | **0,833** | **0,889** | **0,861** |

O BM25 sozinho é melhor que os embeddings neste domínio (termos exatos da DSL, como
`roteirizavel` e `nao agendar`); o híbrido junta os dois e é o melhor em todas as métricas.
O texto dos exemplos começa pelo nome da regra: testado sem ele, o MRR do denso cai de
0,778 para 0,685 (os nomes do acervo descrevem o segmento, ex.: "Inativos (90 dias)").
A tabela acima foi medida com os rascunhos. Refeito com as explicações do LLM (etapa 2), o
índice de exemplos melhora na busca por significado: denso acerto@1 0,833, acerto@3 0,944,
MRR 0,880 (antes 0,722 / 0,833 / 0,778); o híbrido fica igual (0,833 / 0,889 / 0,861).

Limitação observada: as explicações são fiéis (100% no guard), mas pouco variadas: em 46%
delas o LLM praticamente copiou o rascunho, e todas têm o tom formal de documentação, não o
das tabelas dos clientes (`data/exemplos`).

## Teste local com modelo pequeno (qwen3:4b-instruct, RTX 2060)

O grafo rodou ponta a ponta numa GPU de 6 GB (~80 s para a Maxlub) e expôs falhas que os
testes com LLM falso não pegavam, todas corrigidas e cobertas por testes:

- a Percepção pedia JSON só no prompt; um modelo que "pensa em voz alta" estourava o
  contexto, e com o esquema sem campos obrigatórios fechava o JSON vazio. Agora a saída é
  restrita a um esquema com todos os campos obrigatórios (`esquema_estrito`);
- "Não agendar X" ia para restrições gerais e bloqueios eram marcados como `fallback`: a
  Percepção e o Analista corrigem isso em código;
- o Gerador inventava modificadores (`preferencia nos dias uteis`): a conferência agora
  reprova modificador que o segmento não pede;
- o tipo de vendedor saía incoerente com o nome do time: o Analista ajusta.

## Grafo (etapa 4)

O **Analista** é determinístico porque as falhas que importam são aritméticas sobre as
faixas de dias, mais a semântica do GOBR (manual, 7.6: o cliente fica com a **primeira**
regra que o atende):

| Achado | Exemplo (Maxlub) |
|---|---|
| reordenação automática | bloqueios `nao agendar` primeiro, faixas da menor para a maior, "demais clientes" por último |
| buraco entre faixas | grupos de 0–30 e 45–90 dias: quem está entre 30 e 45 dias fica sem regra |
| sobreposição | 90–120 e 100–130: a regra de cima vence entre 100 e 120 |
| sem fallback | última faixa termina em 180 dias e não há regra para os demais (pergunta, com suposição) |
| sem ação / frequência inválida | pergunta **bloqueante**: não dá para gerar |
| fora do GOBR | "capacidade de 35 clientes por dia" não se escreve em `parametros` |
| bloqueio comum ausente | inadimplentes, baixados na Receita Federal (sugestão) |

O **Gerador** escreve um bloco por segmento (a mesma unidade do fine-tuning da etapa 6) e
o arquivo (`version 1.0`, `time`, blocos na ordem do Analista) é montado em código. O
**Validador** soma o `GobrValidator` à conferência pela AST: um bloco por segmento,
`nao agendar` só onde foi pedido, a frequência pedida e os limites da faixa nas condições.

Execução: `python -m src.grafo.executar --arquivo data/exemplos/RegraMaxlub.pdf --trace`
(precisa do Ollama), as etapas 14 e 15 do notebook, a interface Gradio (etapa 16) ou o
**LangGraph Studio** (etapa 16b), que mostra o grafo e executa nó a nó, com o estado de cada
passo e as interrupções para perguntas.

## Fine-tuning (etapa 6)

O Gerador escreve um bloco `parametros` por segmento, então o fine-tuning aprende essa
unidade: **descrição comercial → bloco GOBR**, com o mesmo prompt do agente.

- **Dados**: um par por corpo distinto do acervo (1.122 no treino). A descrição é a
  explicação do bloco: o rascunho determinístico e, quando aprovada pelo guard, a
  explicação do LLM (dois pares com a mesma resposta). Blocos com GOBR inválido ficam fora.
- **Teste sem vazamento**: a separação é por forma de regra (`assinatura_estrutural`):
  as 44 formas do teste (79 pares) não aparecem no treino, nem com outros números. Os 17
  blocos de `data/exemplos` (clientes reais, fora do acervo) são um segundo teste.
- **Modelo**: `Qwen/Qwen3-4B` em 4 bits + LoRA r=16 numa T4. O 4B (e não o 8B) cabe na GPU
  junto com o `gemma3:12b` do Ollama, que os outros agentes continuam usando.
- **Comparação justa**: base e treinado rodam com o mesmo código, a mesma quantização e o
  mesmo prompt, com e sem exemplos do RAG. A métrica principal é `ast_igual` (mesma
  estrutura da referência); também `valido`, `vocabulario_ok`, `numeros_ok` e F1 por linha.

Resultados: `python -m src.finetuning.avaliar --comparar <dir>` (etapa 20 do notebook).
