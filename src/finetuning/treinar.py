"""Fine-tuning QLoRA do Gerador (Colab, GPU T4).

    python -m src.finetuning.treinar --dataset outputs/finetuning/dataset \\
        --saida /content/drive/MyDrive/gobr-agent/finetuning/qwen3-4b-gobr

- Base `Qwen/Qwen3-4B` em 4 bits (NF4) + LoRA r=16 em todas as projeções
  lineares. Cabe numa T4 (15 GB) junto com folga para o contexto.
- Perda só nos tokens da resposta (o bloco GOBR); o prompt é idêntico ao do
  agente (`modelo_hf.tokenizar_par`).
- fp16 (a T4 não tem bf16), gradient checkpointing, otimizador paginado de
  8 bits.
- Checkpoints no Drive a cada `--salvar-a-cada` passos; se a sessão cair,
  rodar o mesmo comando retoma do último checkpoint.
- No fim, grava o adaptador em `<saida>/adaptador` e as curvas de perda em
  `<saida>/historico.json`.

Antes de treinar, pare o Ollama (`pkill -f "ollama serve"`) para liberar a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.finetuning.dataset import SAIDA_PADRAO, carregar
from src.finetuning.modelo_hf import MODELO_BASE, carregar_base, tokenizar_par

ALVOS_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def preparar(dataset: Path, tokenizer, *, max_tokens: int, validacao: float, seed: int):
    pares = carregar(dataset, "treino")
    exemplos = [t for t in (tokenizar_par(p, tokenizer, max_tokens=max_tokens) for p in pares) if t]
    descartados = len(pares) - len(exemplos)
    random.Random(seed).shuffle(exemplos)
    n_val = max(1, int(len(exemplos) * validacao))
    return exemplos[n_val:], exemplos[:n_val], descartados


def argumentos_compativeis(classe, **kwargs):
    """Adapta os argumentos à versão instalada do `transformers`.

    A 5.x removeu `warmup_ratio` (a fração vai em `warmup_steps`, que aceita
    float < 1). Argumentos opcionais que a versão não conhece são descartados
    com aviso, em vez de derrubar o treino.
    """
    import inspect

    aceitos = inspect.signature(classe.__init__).parameters
    if "warmup_ratio" in kwargs and "warmup_ratio" not in aceitos:
        kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")
    descartados = sorted(k for k in kwargs if k not in aceitos)
    if descartados:
        print(f"Aviso: esta versão do transformers não tem {', '.join(descartados)}; seguindo sem.")
    return classe(**{k: v for k, v in kwargs.items() if k in aceitos})


def ultimo_checkpoint(diretorio: Path) -> str | None:
    """Checkpoint mais recente que está inteiro.

    Se a sessão do Colab cai durante a gravação, o último fica pela metade no
    Drive; nesse caso vale o anterior.
    """
    candidatos = [p for p in diretorio.glob("checkpoint-*") if p.name.removeprefix("checkpoint-").isdigit()]
    for pasta in sorted(candidatos, key=lambda p: int(p.name.removeprefix("checkpoint-")), reverse=True):
        try:
            json.loads((pasta / "trainer_state.json").read_text(encoding="utf-8"))
            inteiro = all((pasta / n).stat().st_size > 0
                          for n in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt"))
        except (OSError, ValueError):
            inteiro = False
        if inteiro:
            return str(pasta)
        print(f"Aviso: {pasta.name} está incompleto; ignorando.")
    return None


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--dataset", type=Path, default=SAIDA_PADRAO)
    args.add_argument("--saida", type=Path, required=True)
    args.add_argument("--modelo", default=MODELO_BASE)
    args.add_argument("--epocas", type=float, default=1)
    args.add_argument("--lr", type=float, default=2e-4)
    args.add_argument("--rank", type=int, default=16)
    # Lote 1 × 16 acumulados (lote efetivo 16): com lote 4, a saída do modelo (vocabulário de
    # 152 mil tokens × sequência) estourava os 15 GB da T4 no backward.
    args.add_argument("--lote", type=int, default=1)
    args.add_argument("--acumular", type=int, default=16)
    args.add_argument("--max-tokens", type=int, default=1536)
    args.add_argument("--validacao", type=float, default=0.05)
    args.add_argument("--salvar-a-cada", type=int, default=25)
    args.add_argument("--seed", type=int, default=13)
    a = args.parse_args()

    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # menos fragmentação
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer = carregar_base(a.modelo)
    treino, validacao, descartados = preparar(a.dataset, tokenizer, max_tokens=a.max_tokens,
                                              validacao=a.validacao, seed=a.seed)
    print(f"Pares de treino: {len(treino)} | validação: {len(validacao)} | longos demais: {descartados}")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=ALVOS_LORA, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    a.saida.mkdir(parents=True, exist_ok=True)
    config = argumentos_compativeis(
        TrainingArguments, output_dir=str(a.saida / "checkpoints"), num_train_epochs=a.epocas, learning_rate=a.lr,
        per_device_train_batch_size=a.lote, per_device_eval_batch_size=a.lote,
        gradient_accumulation_steps=a.acumular, lr_scheduler_type="cosine", warmup_ratio=0.03,
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", logging_steps=10,
        eval_strategy="steps", eval_steps=2 * a.salvar_a_cada, save_strategy="steps", save_steps=a.salvar_a_cada,
        save_total_limit=2, report_to="none", seed=a.seed, group_by_length=True,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model, args=config, train_dataset=Dataset.from_list(treino), eval_dataset=Dataset.from_list(validacao),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    ultimo = ultimo_checkpoint(a.saida / "checkpoints")
    if ultimo:
        print(f"Retomando de {ultimo}")
    trainer.train(resume_from_checkpoint=ultimo)

    model.save_pretrained(str(a.saida / "adaptador"))
    tokenizer.save_pretrained(str(a.saida / "adaptador"))
    resumo = {"modelo_base": a.modelo, "pares_treino": len(treino), "pares_validacao": len(validacao),
              "descartados": descartados, "epocas": a.epocas, "lr": a.lr, "rank": a.rank,
              "lote_efetivo": a.lote * a.acumular, "historico": trainer.state.log_history}
    (a.saida / "historico.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Adaptador salvo em {a.saida / 'adaptador'}")


if __name__ == "__main__":
    main()
