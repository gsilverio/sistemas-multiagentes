"""Configuração central dos LLMs (Ollama) usados pelos agentes GOBR.

Variáveis de ambiente: GOBR_LLM_MODEL (padrão gemma3:12b), GOBR_LLM_NUM_CTX
(padrão 4096), OLLAMA_BASE_URL (padrão http://localhost:11434) e, por agente,
GOBR_MODELO_PERCEPCAO / _GERADOR / _ORQUESTRADOR / _EXPLICADOR. Para usar o
Gerador treinado: GOBR_MODELO_GERADOR=hf:Qwen/Qwen3-4B e
GOBR_GERADOR_ADAPTADOR=<diretório do adaptador>.
"""

import os


def modelo_padrao() -> str:
    return os.getenv("GOBR_LLM_MODEL", "gemma3:12b")


def contexto_padrao() -> int:
    return int(os.getenv("GOBR_LLM_NUM_CTX", "4096"))


PAPEIS = ("percepcao", "gerador", "orquestrador", "explicador")


def modelo_para(papel: str) -> str:
    """Modelo de um agente: `GOBR_MODELO_<PAPEL>` (ex.: GOBR_MODELO_GERADOR) ou o padrão.

    Por padrão todos usam o mesmo modelo, para caber numa T4 sem trocar de
    modelo a cada chamada (OLLAMA_MAX_LOADED_MODELS=1 no Colab).
    """
    if papel not in PAPEIS:
        raise ValueError(f"Papel desconhecido: {papel}")
    return os.getenv(f"GOBR_MODELO_{papel.upper()}") or modelo_padrao()


_MODELOS_HF: dict = {}


def create_llm_para(papel: str, **kwargs):
    """LLM de um agente. `hf:<modelo>` serve pelo Hugging Face em vez do Ollama;
    com GOBR_GERADOR_ADAPTADOR=<dir>, aplica o adaptador LoRA do fine-tuning (etapa 6)."""
    modelo = modelo_para(papel)
    if modelo.startswith("hf:"):
        adaptador = os.getenv("GOBR_GERADOR_ADAPTADOR") if papel == "gerador" else None
        chave = (modelo, adaptador)
        if chave not in _MODELOS_HF:  # carrega uma vez por processo
            from src.finetuning.modelo_hf import GeradorHF

            _MODELOS_HF[chave] = GeradorHF(modelo.removeprefix("hf:"), adaptador)
        return _MODELOS_HF[chave]
    return create_llm(modelo, **kwargs)


def create_llm(model: str | None = None, *, num_ctx: int | None = None, temperature: float = 0,
               formato: str | dict | None = None):
    """`formato`: "json" ou um JSON Schema; o Ollama restringe a saída a ele (sem texto fora do JSON)."""
    from langchain_ollama import ChatOllama

    nome = model or modelo_padrao()
    extras = {}
    if nome.startswith("qwen3") and "instruct" not in nome:
        extras["reasoning"] = False  # Sem bloco <think>: resposta direta.
    if formato is not None:
        extras["format"] = formato
    return ChatOllama(
        model=nome,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        num_ctx=num_ctx or contexto_padrao(),
        **extras,
    )
