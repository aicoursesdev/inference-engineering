"""Ridge points of every chip in data/hardware.json, and what the sparsity trap would do to them.

    uv run python ridge_table.py
"""

from servinglab.data import hardware


def ridge(tflops, bandwidth_gbs):
    """FLOPs per byte at which a chip stops being memory-bound."""
    return tflops * 1000 / bandwidth_gbs


def headline(chip, key="fp16_tflops"):
    """The number a datasheet prints in big type: twice the dense figure when the chip's figure was
    stored halved ("derived": the vendor quotes "with sparsity"), otherwise the dense figure itself."""
    dense = chip[key]
    return dense * 2 if chip["sources"][key].get("derived") and "spars" in chip["sources"][key]["note"].lower() else dense


def rows():
    out = []
    for chip in hardware():
        if chip["fp16_tflops"] is None or chip["bandwidth_gbs"] is None:
            continue
        dense, bw = chip["fp16_tflops"], chip["bandwidth_gbs"]
        out.append({"name": chip["name"], "gbs": bw, "dense": dense, "headline": headline(chip),
                    "ridge": ridge(dense, bw), "wrong_ridge": ridge(headline(chip), bw)})
    return out


if __name__ == "__main__":
    print(f"{'chip':<16}{'GB/s':>6}{'dense':>8}{'headline':>10}{'ridge':>7}{'if trusted':>11}")
    for r in rows():
        print(f"{r['name']:<16}{r['gbs']:>6}{r['dense']:>8.1f}{r['headline']:>10.1f}{r['ridge']:>7.0f}{r['wrong_ridge']:>11.0f}")
