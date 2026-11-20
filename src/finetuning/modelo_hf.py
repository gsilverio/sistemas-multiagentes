"""Modelo do Gerador servido pelo Hugging Face (base em 4 bits + adaptador LoRA).

`GeradorHF` tem o mesmo `invoke(mensagens)` que o ChatOllama usado pelos
agentes, então entra no `AgenteGerador` sem mudança. Serve o modelo base e o
modelo treinado com o MESMO código e a mesma quantização, que é o que torna a
comparação da avaliação justa.

Também define como um par vira tokens de treino (`tokenizar_par`): o prompt
é renderizado pelo chat template com `enable_thinking=False` (Qwen3 responde
direto, sem `<think>`), e só os tokens da resposta entram na perda.
"""

from __future__ import annotations

from typing import Any

from src.prompts import gerador_prompt

MODELO_BASE = "Qwen/Qwen3-4B"
PAPEIS = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}


def mensagens_do_par(par: dict[str, Any]) -> list[tuple[str, str]]:
    """Prompt do par no formato do agente (system + exemplos + pedido)."""
    return gerador_prompt.mensagens(par["nome"], par["escopo"], par["descricao"],
                                    exemplos=[tuple(e) for e in par.get("exemplos") or []])


def para_chat(mensagens: list) -> list[dict[str, str]]:
    """Tuplas do LangChain (`("human", "...")`) ou mensagens → formato do chat template."""
    chat = []
    for m in mensagens:
        if isinstance(m, tuple):
            papel, conteudo = m
        else:
            papel, conteudo = getattr(m, "type", "user"), getattr(m, "content", str(m))
        chat.append({"role": PAPEIS.get(papel, "user"), "content": str(conteudo)})
    return chat


def texto_do_prompt(tokenizer: Any, mensagens: list) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(para_chat(mensagens), enable_thinking=False, **kwargs)
    except TypeError:  # template sem enable_thinking
        return tokenizer.apply_chat_template(para_chat(mensagens), **kwargs)


def fim_de_turno(tokenizer: Any) -> str:
    return "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else (tokenizer.eos_token or "")


def tokenizar_par(par: dict[str, Any], tokenizer: Any, *, max_tokens: int = 1536) -> dict[str, list[int]] | None:
    """input_ids e labels (-100 no prompt). None se não couber em `max_tokens`."""
    prompt = tokenizer(texto_do_prompt(tokenizer, mensagens_do_par(par)), add_special_tokens=False)["input_ids"]
    resposta = tokenizer(par["bloco"] + fim_de_turno(tokenizer) + "\n", add_special_tokens=False)["input_ids"]
    if len(prompt) + len(resposta) > max_tokens:
        return None
    return {"input_ids": prompt + resposta, "attention_mask": [1] * (len(prompt) + len(resposta)),
            "labels": [-100] * len(prompt) + resposta}


def carregar_base(modelo: str = MODELO_BASE, *, quatro_bits: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(modelo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantizacao = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16) if quatro_bits else None
    model = AutoModelForCausalLM.from_pretrained(modelo, quantization_config=quantizacao, device_map="auto",
                                                 torch_dtype=torch.float16)
    return model, tokenizer


class GeradorHF:
    """`invoke(mensagens) -> str`, como um chat model do LangChain."""

    def __init__(self, modelo: str = MODELO_BASE, adaptador: str | None = None, *, max_novos_tokens: int = 320,
                 model: Any = None, tokenizer: Any = None):
        if model is None:
            model, tokenizer = carregar_base(modelo)
            if adaptador:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adaptador)
        model.eval()
        self.model, self.tokenizer = model, tokenizer
        self.nome = f"{modelo}+{adaptador}" if adaptador else modelo
        self.max_novos_tokens = max_novos_tokens

    def invoke(self, mensagens: list) -> str:
        import torch

        prompt = texto_do_prompt(self.tokenizer, mensagens)
        entrada = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        fim = self.tokenizer.convert_tokens_to_ids(fim_de_turno(self.tokenizer))
        with torch.no_grad():
            saida = self.model.generate(**entrada, max_new_tokens=self.max_novos_tokens, do_sample=False,
                                        eos_token_id=[t for t in {fim, self.tokenizer.eos_token_id} if t is not None],
                                        pad_token_id=self.tokenizer.pad_token_id)
        novos = saida[0][entrada["input_ids"].shape[1]:]
        return self.tokenizer.decode(novos, skip_special_tokens=True).strip()
