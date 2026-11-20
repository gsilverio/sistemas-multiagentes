# Arquitetura multiagente GOBR

Objetivo (ver [`enunciado.pdf`](enunciado.pdf)): um agente conversacional que recebe uma
imagem ou descrição das regras comerciais de um cliente, aponta falhas na modelagem
e devolve o conjunto de regras na DSL **GOBR** do goVendas.

Restrição do projeto: **somente modelos open source** (servidos pelo Ollama, localmente
ou no Colab).

## Visão geral

```
                        ┌────────────────────┐
              ┌────────►│   ORQUESTRADOR     │◄────────┐
              │         │ (LLM + guardrails) │         │
              │         └─────────┬──────────┘         │
              │   Command(goto=?) │                    │
   ┌──────────┼──────────┬────────┼─────────┬──────────┼──────────┐
   ▼          ▼          ▼        ▼         ▼          ▼          ▼
percepcao  analista  recuperador gerador validador explicador perguntar_usuario
(subgrafo) (subgrafo) (subgrafo) (subgrafo)  (nó)     (nó)      (interrupt)
   └──────────┴──────────┴────────┴─────────┴──────────┴──────────┘
                todos voltam com Command(goto="orquestrador")
                                  │
                              finalizar ──► END
```

O grafo é um **supervisor em estrela** no LangGraph:

- O **orquestrador** é um LLM que escolhe o próximo nó (`Decisao{proximo, motivo}`) a
  partir de um *resumo* do estado (flags, contagens, erros), sem ver o conteúdo produzido
  pelos agentes.
- **Guardrails em código** verificam pré-condições de cada destino (ex.: só gera GOBR com
  `RuleSpec` pronto e sem perguntas abertas; só finaliza com validação OK). Se o LLM
  escolhe um destino inválido, o guardrail corrige e registra no `trace`.
- Cada agente é um **nó ou subgrafo isolado** com entrada/saída tipadas (Pydantic). O
  estado interno de um subgrafo não vaza para o grafo principal.
- **Verificação cega**: o explicador recebe apenas o GOBR (não a descrição original); o
  analista recebe apenas o `RuleSpec` (não o GOBR). Quem verifica não é influenciado por
  quem produziu.
- **Memória**: checkpointer (`SqliteSaver`) por `thread_id`; `interrupt()` pausa o grafo
  para perguntas ao usuário.

## Agentes

| Agente | Entrada | Saída | Modelo |
|---|---|---|---|
| Orquestrador | resumo do estado + mensagem | próximo nó + motivo | qwen3:8b |
| Percepção | imagem / PDF / texto | `RuleSpec` (segmentos: nome, critério, frequência, restrições) | gemma3:12b (visão) |
| Analista de negócio | `RuleSpec` | problemas + perguntas | verificações determinísticas + LLM para redigir |
| Recuperador | `RuleSpec` | exemplos similares + trechos do manual | busca híbrida (BM25 + denso) |
| Gerador GOBR | `RuleSpec` + contexto RAG | GOBR | Qwen3-8B, depois fine-tuning QLoRA |
| Validador | GOBR | erros acionáveis | determinístico (`src/gobr`) |
| Explicador | GOBR | explicação comercial | gemma3:12b / qwen3:8b |

## Estrutura do código

Descrição de cada pasta, módulo e dos códigos de erro do validador:
[`ESTRUTURA_DO_CODIGO.md`](ESTRUTURA_DO_CODIGO.md).

## Etapas

| # | Etapa | Resultado | Status |
|---|---|---|---|
| 0 | Limpeza e reestruturação (tag `entregas-1-a-4` preserva o estado anterior) | repositório enxuto, testes passando | ✅ |
| 1 | Validador GOBR completo (arquivo inteiro, referências, vocabulário de condições) | `GobrValidator.validar(gobr)` com erros acionáveis | ✅ |
| 2 | Explicador novo (AST → rascunho determinístico → 1 chamada LLM → guard) e geração por bloco | código ✅; 1.698 rascunhos gerados localmente; versão com LLM ⏳ no Colab | ⏳ |
| 3 | RAG híbrido, re-chunking do manual por construção da linguagem | `buscar_exemplos()` / `buscar_manual()`; índices com rascunhos, refazer com as explicações do LLM | ✅ |
| 4 | Grafo multiagente com orquestrador supervisor e subgrafos | fluxo ponta a ponta | |
| 5 | Memória de contexto + interface Gradio | entrega final | |
| 6 | Fine-tuning QLoRA do gerador + avaliação comparativa | modelo próprio plugado | |

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
Os índices atuais usam os rascunhos determinísticos; serão refeitos com as explicações do
LLM quando o lote da etapa 2 terminar.
