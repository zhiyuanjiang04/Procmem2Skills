"""Compute Qwen3 embeddings for task instructions."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).parent
TASKS = ROOT / "outputs" / "tasks.jsonl"
EMB_OUT = ROOT / "outputs" / "embeddings.npy"
IDS_OUT = ROOT / "outputs" / "task_ids.json"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_LEN = 1024
BATCH_SIZE = 2


def last_token_pool(last_hidden_states, attention_mask):
    """Qwen3-Embedding uses last-token pooling (per HF model card)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), seq_lengths
    ]


def main():
    rows = [json.loads(l) for l in open(TASKS)]
    print(f"loaded {len(rows)} tasks")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()

    instructions = [r["instruction"] for r in rows]
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(instructions), BATCH_SIZE):
            batch = instructions[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            pooled = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
            del out, pooled, enc
            if device == "mps":
                torch.mps.empty_cache()
            print(f"  batch {i // BATCH_SIZE + 1}/{(len(instructions) + BATCH_SIZE - 1) // BATCH_SIZE}")

    emb = np.vstack(embeddings).astype(np.float32)
    np.save(EMB_OUT, emb)
    with open(IDS_OUT, "w") as f:
        json.dump([r["task_id"] for r in rows], f)

    print(f"\nsaved {emb.shape} → {EMB_OUT}")
    print(f"saved {len(rows)} task_ids → {IDS_OUT}")


if __name__ == "__main__":
    main()
