"""Given: load the course's data files. Every lesson may use these.

    from servinglab.data import traces, hardware, model_config, prices

    traces("tidewater")        # a list of request dicts (see data/README.md)
    hardware("H100-SXM")       # one chip's spec dict; hardware() lists them all
    model_config("Qwen3-4B")   # the model's real config.json as a dict
    prices()                   # data/prices.json as a dict
"""

import json
from functools import cache
from pathlib import Path

# data/ sits beside this package in the repo, so this works from any folder once installed with uv sync.
DATA = Path(__file__).resolve().parent.parent / "data"

TRACES = ("tidewater", "kodekart", "nyaya", "batch", "mixed")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def _traces(name):
    with open(DATA / "traces" / f"{name}.jsonl", encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)


def traces(name):
    """The requests of one trace, in arrival order. `name` is one of TRACES."""
    if name not in TRACES:
        raise ValueError(f"no trace {name!r}; choose from {', '.join(TRACES)}")
    return [dict(r) for r in _traces(name)]  # copies, so a lesson can edit its own


@cache
def _hardware():
    return {h["name"]: h for h in _json(DATA / "hardware.json")["hardware"]}


def hardware(name=None):
    """One chip's spec by name (e.g. "T4", "H100-SXM", "M5-Max"); with no name, every chip."""
    chips = _hardware()
    if name is None:
        return [dict(c) for c in chips.values()]
    if name not in chips:
        raise ValueError(f"no hardware {name!r}; choose from {', '.join(chips)}")
    return dict(chips[name])


def model_config(name):
    """A model's config.json, by the name data/models/<name>.json has (e.g. "SmolLM2-135M-Instruct")."""
    path = DATA / "models" / f"{name}.json"
    if not path.is_file():
        names = sorted(p.stem for p in path.parent.glob("*.json") if p.stem != "sources")
        raise ValueError(f"no model {name!r}; choose from {', '.join(names)}")
    return _json(path)


def prices():
    """GPU hourly rates and API prices, each with its source URL and date."""
    return _json(DATA / "prices.json")
