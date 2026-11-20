# GOBR — agente multiagente para parametrização do goVendas

Agente conversacional que recebe uma imagem ou descrição de regras comerciais e gera as
regras na DSL GOBR, apontando falhas na modelagem. Usa apenas modelos open source
(Ollama) e LangGraph.

- Arquitetura, agentes e etapas: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md)
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

Extrair os 1.699 blocos `parametros` das referências e diagnosticar o parser:

```powershell
python -m src.pipelines.prepare_references
```

Gerar explicações em lote (pensado para o Colab, com retomada incremental):

```powershell
python -m src.pipelines.generate_explanations --input outputs/generated/regras_extraidas.jsonl --limit 3
```

Indexar o manual e as explicações:

```powershell
python -m src.rag.manual_index --pdf data/manual/govendas-rules.pdf --output-dir outputs/manual_index
```

```powershell
python -m src.rag.explanation_index --items-dir <pasta items> --output-dir <pasta do índice>
```

O índice do manual versionado em `outputs/manual_index` usa
`paraphrase-multilingual-MiniLM-L12-v2` e um trecho por página; será refeito na etapa 3.
