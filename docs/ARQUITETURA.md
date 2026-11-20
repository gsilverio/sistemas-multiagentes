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

```
src/
  gobr/         núcleo determinístico da DSL: AST, parser, extrator de blocos
  validation/   verificações determinísticas (guard das explicações; validador GOBR)
  agents/       agentes com LLM
  prompts/      prompts dos agentes
  models/       modelos Pydantic de saída dos agentes
  rag/          indexação e busca (manual e explicações)
  pipelines/    scripts de lote: preparar referências, gerar explicações
  llm_config.py configuração dos modelos (Ollama)
data/
  referencias/  246 arquivos .gobr (1.699 blocos `parametros`)
  exemplos/     3 regras com PDF da regra comercial + GOBR (conjunto de teste)
  manual/       manual da linguagem
  avaliacao/    linhas de base para comparação
```

## Etapas

| # | Etapa | Resultado |
|---|---|---|
| 0 | Limpeza e reestruturação (tag `entregas-1-a-4` preserva o estado anterior) | repositório enxuto, testes passando |
| 1 | Validador GOBR completo (arquivo inteiro, referências, vocabulário de condições) | `validar(gobr)` com erros acionáveis |
| 2 | Explicador novo (AST → rascunho determinístico → 1 chamada LLM → guard) e geração por bloco | ~1.700 explicações revisadas |
| 3 | RAG híbrido, re-chunking do manual por construção da linguagem | `buscar_exemplos()` / `buscar_manual()` |
| 4 | Grafo multiagente com orquestrador supervisor e subgrafos | fluxo ponta a ponta |
| 5 | Memória de contexto + interface Gradio | entrega final |
| 6 | Fine-tuning QLoRA do gerador + avaliação comparativa | modelo próprio plugado |

## Linha de base

`data/avaliacao/explicacoes_baseline_v1.jsonl` guarda as 59 explicações geradas pelo
pipeline antigo (Interpretador → Explicador, `gemma3:12b`, regras 001 a 015). Elas **não**
são usadas como verdade: servem para comparar com o explicador novo da etapa 2. Problemas
conhecidos dessa versão: `frequencia de inativos da empresa` interpretada como "atendimento
para inativos" (é a frequência configurada para inativos), frequência inventada
(`regra_005_003` cita "14 dias" que não existe na regra) e 10 casos que caíram no
fallback determinístico.
