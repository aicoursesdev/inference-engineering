"""Offline: the formula against the counts Hugging Face reports for each
model's safetensors files (api/models/<repo>, 'safetensors.total', 2026-11-12;
Qwen3-4B's index.json total_size, 8,044,936,192 bytes, is exactly 2 bytes each)."""

import pytest

from servinglab.data import model_config
from size_model import param_count

REPORTED = {
    "SmolLM2-135M-Instruct": 134_515_008,
    "Qwen3-4B": 4_022_468_096,
    "Llama-3.1-8B-Instruct": 8_030_261_248,
    "Llama-3.1-70B-Instruct": 70_553_706_496,
}


@pytest.mark.parametrize("name", REPORTED)
def test_formula_matches_reported_count(name):
    assert param_count(model_config(name)) == REPORTED[name]
