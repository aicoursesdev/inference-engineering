"""AWQ 4-bit with LLM Compressor 0.15.0, adapted from the project's own
examples/awq/llama_example.py (model id changed). Needs a GPU; not run here.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_ID = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM",
                         targets=["Linear"]),
]

oneshot(model=model, dataset="perfectblend", splits="train[:512]",
        recipe=recipe, max_seq_length=512, num_calibration_samples=256)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-asym"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
