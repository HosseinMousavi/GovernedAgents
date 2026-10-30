"""LoRA fine-tune of a small instruct model on the distillation set (PEFT + transformers).

    python -m store_support.lora.train                       # Qwen2.5-1.5B-Instruct, r=16, 2 epochs, micro-batch 1
    python -m store_support.lora.train --base Qwen/Qwen2.5-0.5B-Instruct --epochs 1   # ~4x faster, fits 16 GB
    python -m store_support.lora.train --batch 4 --no-grad-checkpointing              # CUDA with >=24 GB

Runs on Apple silicon (MPS), CUDA, or CPU. Only the assistant turn is trained on (prompt tokens
are masked), so the student learns to emit the JSON/list answer, not to reproduce the prompt.
Writes models/lora/<name>/adapter (the LoRA weights) and models/lora/<name>/merged (full weights
for Ollama import).
"""

from __future__ import annotations

import json
import sys
import time

import typer

from store_support.evals.runner import ROOT

app = typer.Typer(add_completion=False)
DATA = ROOT / "data" / "lora"
MODELS = ROOT / "models" / "lora"


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _encode(tokenizer, messages: list[dict[str, str]], max_len: int) -> dict[str, list[int]] | None:  # noqa: ANN001
    """Tokenize with the model's chat template; label only the assistant tokens.

    Returns None when the prompt alone fills max_len: such an example has no learnable token, and an
    all -100 label row makes the loss NaN (fatal with micro-batch 1)."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + messages[-1]["content"] + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    if len(f_ids) <= len(p_ids):
        return None
    labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}


def encode_all(tokenizer, rows: list[dict], max_len: int) -> tuple[list[dict[str, list[int]]], int]:  # noqa: ANN001
    """Encode every row; returns (kept, dropped_count)."""
    encoded = [_encode(tokenizer, r["messages"], max_len) for r in rows]
    kept = [e for e in encoded if e is not None]
    return kept, len(encoded) - len(kept)


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-1.5B-Instruct",
    name: str = "ssa-grader",
    epochs: float = 2.0,
    lr: float = 2e-4,
    r: int = 16,
    alpha: int = 32,
    max_len: int = 1024,
    batch: int = 1,
    grad_checkpointing: bool = True,
) -> None:
    """Defaults fit a 16-24 GB Apple-silicon machine: micro-batch 1 x 8 accumulation = effective batch 8.
    Qwen's 152k vocab makes the fp32 logits ~0.6 GB per 1024-token sequence, so bigger micro-batches swap on MPS."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    device = _device()
    out = MODELS / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"base={base} device={device} out={out}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    model.to(device)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.config.use_cache = False
    if grad_checkpointing:  # recompute activations in backward: ~3x less memory for ~30% more compute
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text().splitlines() if line.strip()]
    val_rows = [json.loads(line) for line in (DATA / "val.jsonl").read_text().splitlines() if line.strip()]
    train_enc, dropped_t = encode_all(tok, rows, max_len)
    val_enc, dropped_v = encode_all(tok, val_rows[:64], max_len)
    if dropped_t or dropped_v:
        print(f"dropped {dropped_t} train / {dropped_v} val examples longer than max_len={max_len}", file=sys.stderr)
    if not train_enc:
        raise SystemExit("no trainable examples: raise --max-len")
    train_ds = Dataset.from_list(train_enc)
    val_ds = Dataset.from_list(val_enc)

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=max(1, 8 // batch),
        gradient_checkpointing=grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=device == "cuda",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=device == "cuda",
        report_to=[],
        warmup_steps=10,
        lr_scheduler_type="cosine",
    )
    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"trained in {time.time() - t0:.0f}s  eval_loss={metrics.get('eval_loss'):.3f}", file=sys.stderr)

    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    (out / "Modelfile").write_text(
        f"FROM ./{name}.gguf\nPARAMETER temperature 0\nPARAMETER num_ctx 4096\n"
        f"# Ollama cannot import Qwen2 safetensors; `make lora-serve` converts ./merged to ./{name}.gguf first.\n"
    )
    (out / "train_metrics.json").write_text(
        json.dumps(
            {
                "base": base,
                "r": r,
                "alpha": alpha,
                "epochs": epochs,
                "train_examples": len(rows),
                "eval_loss": metrics.get("eval_loss"),
                "seconds": round(time.time() - t0),
            },
            indent=2,
        )
    )
    print(f"adapter -> {out / 'adapter'}\nmerged  -> {out / 'merged'}\nnext: make lora-serve", file=sys.stderr)


if __name__ == "__main__":
    app()
