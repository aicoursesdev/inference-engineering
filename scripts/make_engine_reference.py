"""Make the reference files the engine's tests compare against.

Run once, only when the tests' expectations change (it needs PyTorch and
transformers, which the course repo does not install):

    uv run --with torch --with transformers python scripts/make_engine_reference.py

Writes to servinglab/tests/data/:
  tiny-llama/            a 2-layer random Llama (config.json, model.safetensors)
                         and reference.json: its logits for a fixed input
  smollm2-135m.npz       SmolLM2-135M-Instruct's logits for the next token
                         after a fixed prompt, plus 20 greedy tokens and the
                         chat template's token ids (needs models/SmolLM2-135M-Instruct)
Every number comes from transformers' own LlamaForCausalLM in float32.
"""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "servinglab" / "tests" / "data"

# ---- a tiny random model: runs in milliseconds, tests every code path ----
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, rms_norm_eps=1e-5,
                  rope_theta=10000.0, tie_word_embeddings=False,
                  max_position_embeddings=64)
tiny = LlamaForCausalLM(cfg).eval()
with torch.no_grad():  # random weights large enough that logits are not all ~0
    for p in tiny.parameters():
        p.copy_(torch.randn_like(p) * 0.3 + (1.0 if p.ndim == 1 else 0.0))
tiny.save_pretrained(OUT / "tiny-llama", safe_serialization=True)
ids = [5, 17, 3, 99, 42, 8, 8, 61, 0, 23]
with torch.no_grad():
    logits = tiny(torch.tensor([ids])).logits[0].numpy()
(OUT / "tiny-llama" / "reference.json").write_text(json.dumps({
    "made_with": f"transformers {transformers.__version__}, torch {torch.__version__}",
    "ids": ids, "logits": logits.round(5).tolist()}))

# ---- SmolLM2-135M-Instruct ----
path = ROOT / "models" / "SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
prompt = "The capital of France is"
p_ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    last = model(p_ids).logits[0, -1].numpy()
    out = model.generate(p_ids, max_new_tokens=20, do_sample=False)
msgs = [{"role": "user", "content": "Can I return a rain jacket?"}]
chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
chat = chat["input_ids"] if hasattr(chat, "keys") else chat
np.savez_compressed(OUT / "smollm2-135m.npz",
                    prompt_ids=p_ids[0].numpy(), last_logits=last,
                    greedy=out[0, p_ids.shape[1]:].numpy(),
                    chat_ids=np.array(chat))
print("done", transformers.__version__, torch.__version__)
