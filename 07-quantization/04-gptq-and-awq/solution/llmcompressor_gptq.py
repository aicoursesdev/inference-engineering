"""GPTQ 4-bit with LLM Compressor 0.15.0, adapted from the project's own
examples/quantization_w4a16/llama3_example.py (model id changed). Needs a GPU
(a Colab T4 can hold Qwen3-0.6B); not run on the laptop this was written on.

    pip install llmcompressor==0.15.0
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4-bit weights, groups of 128, found with GPTQ; the output layer stays 16-bit
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(model=model, dataset="perfectblend", splits="train[:512]",
        recipe=recipe, max_seq_length=2048, num_calibration_samples=512)

SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
