# GOBR — Geração de explicações (primeira entrega)

Este projeto usa `GobrParser`, `GobrInterpreterAgent` e `BusinessExplainerAgent` para traduzir parametrizações GOBR em explicações comerciais.

## Pré-requisitos

- Python com `pydantic`, `langchain-core`, `langchain-ollama` e `langgraph` instalados.
- Ollama em execução.

A codebase original não acompanha arquivo de dependências com versões fixadas; esta alteração mantém as importações existentes.

## Modelo

Por padrão, os dois agentes usam **`gemma3:12b`**. Para baixar no Ollama:

```powershell
ollama pull gemma3:12b
```

Parâmetros opcionais (PowerShell, na sessão atual):

```powershell
$env:GOBR_LLM_MODEL = "gemma3:12b"
$env:GOBR_LLM_NUM_CTX = "4096"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

Se a máquina tiver cerca de 6 GB de VRAM e o 12B estiver lento, uma alternativa maior que o antigo 4B é:

```powershell
ollama pull qwen3:8b
$env:GOBR_LLM_MODEL = "qwen3:8b"
```

O modelo 12B ocupa mais de 8 GB somente em pesos quantizados, além de memória de execução; pode utilizar RAM/CPU fora da GPU.

## Demonstração

Na raiz do projeto:

```powershell
python -m src.demo_first_delivery
```

Mantêm-se os seis exemplos e os arquivos de saída em `outputs/first_delivery/`. A codebase também contém 246 referências `.gobr` em `data/referencias/` e três exemplos adicionais com PDFs e suas regras. Nenhuma indexação foi implementada nesta etapa.

## Escopo da limpeza

- Removidos os scripts de avaliação quantitativa de `src/evaluation/` e os dados de teste de `tests/fixtures/`.
- Preservados os scripts de demonstração e teste manual de fluxo (`src/test_*.py`).
- A implementação de LangGraph em `src/graph/explanation_graph.py` já tinha uma divergência de tipos anterior às alterações (passa uma string ao interpretador, que espera um objeto `GobrParsedRule`); deve ser corrigida antes de utilizá-la na próxima entrega.

## Execucao no Colab (mantendo Windows + GitHub)

- Notebook: [`notebooks/gobr_colab.ipynb`](notebooks/gobr_colab.ipynb).
- Guia para iniciantes: [`docs/COLAB_PASSO_A_PASSO.md`](docs/COLAB_PASSO_A_PASSO.md).
- Dependencias: `pip install -r requirements.txt`.
- As saídas do Colab são salvas diretamente em `Meu Drive/gobr-agent/outputs/first_delivery_colab`.
- O Colab apenas executa o pipeline; **o código-fonte continua sendo mantido no Windows e enviado ao GitHub**.
- A indexação das explicações (segunda entrega) permanece pendente.
