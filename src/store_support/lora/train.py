"""LoRA fine-tune of a small instruct model on the distillation set (PEFT + transformers).

    python -m store_support.lora.train                       # Qwen2.5-1.5B-Instruct, r=16, 2 epochs
    python -m store_support.lora.train --base Qwen/Qwen2.5-0.5B-Instruct --epochs 1   # faster smoke run

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


def _encode(tokenizer, messages: list[dict[str, str]], max_len: int) -> dict[str, list[int]]:  # noqa: ANN001
    """Tokenize with the model's chat template; label only the assistant tokens."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + messages[-1]["content"] + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids) :]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-1.5B-Instruct",
    name: str = "ssa-grader",
    epochs: float = 2.0,
    lr: float = 2e-4,
    r: int = 16,
    alpha: int = 32,
    max_len: int = 2048,
    batch: int = 4,
) -> None:
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
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text().splitlines() if line.strip()]
    val_rows = [json.loads(line) for line in (DATA / "val.jsonl").read_text().splitlines() if line.strip()]
    train_ds = Dataset.from_list([_encode(tok, r_["messages"], max_len) for r_ in rows])
    val_ds = Dataset.from_list([_encode(tok, r_["messages"], max_len) for r_ in val_rows[:64]])

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=max(1, 8 // batch),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=device == "cuda",
        use_mps_device=device == "mps",
        report_to=[],
        warmup_ratio=0.05,
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
        f"FROM ./merged\nPARAMETER temperature 0\nPARAMETER num_ctx 4096\n"
        f'SYSTEM ""\n# import: ollama create {name} -f {out / "Modelfile"}\n'
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
