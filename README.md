# GOBR — agente multiagente para parametrização do goVendas

Agente conversacional que recebe uma imagem ou descrição de regras comerciais e gera as
regras na DSL GOBR, apontando falhas na modelagem. Usa apenas modelos open source
(Ollama) e LangGraph.

- Arquitetura, agentes e etapas: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md)
- Estrutura do código, módulo a módulo: [`docs/ESTRUTURA_DO_CODIGO.md`](docs/ESTRUTURA_DO_CODIGO.md)
- Enunciado: [`docs/enunciado.pdf`](docs/enunciado.pdf)
- Execução no Colab: [`notebooks/gobr_colab.ipynb`](notebooks/gobr_colab.ipynb) e
  [`docs/COLAB_PASSO_A_PASSO.md`](docs/COLAB_PASSO_A_PASSO.md)
- Estado das entregas 1 a 4 antes da reestruturação: tag `entregas-1-a-4`

## Instalação

```powershell
pip install -r requirements.txt
ollama pull gemma3:12b
```

Variáveis opcionais: `GOBR_LLM_MODEL` (padrão `gemma3:12b`), `GOBR_LLM_NUM_CTX`
(padrão `4096`) e `OLLAMA_BASE_URL` (padrão `http://localhost:11434`).

## Comandos

Testes (sem GPU e sem Ollama):

```powershell
python -m unittest discover -s tests -v
```

Validar um arquivo GOBR (erros de sintaxe, referências e vocabulário):

```powershell
python -m src.validation.gobr_validator data/exemplos/RegraMaxlub.gobr
```

Regenerar o vocabulário de condições a partir do acervo:

```powershell
python -m src.pipelines.build_vocabulary
```

Extrair os 1.700 blocos `parametros` das referências e diagnosticar o parser:

```powershell
python -m src.pipelines.prepare_references
```

Gerar explicações em lote, com retomada incremental. Sem LLM (só o rascunho
determinístico, roda local em segundos):

```powershell
python -m src.pipelines.generate_explanations --sem-llm --limit 0
```

Com LLM (no Colab, ver o notebook):

```powershell
python -m src.pipelines.generate_explanations --limit 5
```

Conferir explicações com o guard (aqui, a linha de base antiga):

```powershell
python -m src.pipelines.avaliar_explicacoes data/avaliacao/explicacoes_baseline_v1.jsonl
```

Gerar os índices de recuperação (manual por seção + exemplos do acervo) em
`outputs/rag/`. Usa `data/explicacoes/explicacoes_v2.jsonl` se existir; com
`--sem-vetores`, só BM25:

```powershell
python -m src.pipelines.construir_indices
```

Medir a recuperação (BM25 × denso × híbrido):

```powershell
python -m src.pipelines.avaliar_recuperacao --erros
```

Buscar no código: `from src.rag.recuperador import buscar_manual, buscar_exemplos`.

Rodar o grafo multiagente (precisa do Ollama; no Colab, etapas 14 e 15 do notebook):

```powershell
python -m src.grafo.executar --arquivo data/exemplos/RegraMaxlub.pdf --trace
```

`--interativo` pausa para perguntas; `--orquestrador-sem-llm` usa só as regras;
`--explicar-com-llm` usa o LLM nas explicações.

Interface Gradio (entrega final; precisa do Ollama, no Colab é a etapa 16 do notebook). As
conversas ficam em `outputs/conversas.sqlite` e podem ser retomadas:

```powershell
python -m src.interface.app
```
