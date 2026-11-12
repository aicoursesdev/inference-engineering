"""Ridge points of every chip in data/hardware.json, and what the sparsity trap would do to them.

    uv run python ridge_table.py
"""

from servinglab.data import hardware


def ridge(tflops, bandwidth_gbs):
    """FLOPs per byte at which a chip stops being memory-bound."""
    raise NotImplementedError(
        "TODO 02.6: return FLOPs per byte at which a chip stops being memory-bound: TFLOPS * 1000 / GB/s, see the lesson's 'The ridge points of the whole course' section")


def headline(chip, key="fp16_tflops"):
    """The number a datasheet prints in big type: twice the dense figure when the chip's figure was
    stored halved ("derived": the vendor quotes "with sparsity"), otherwise the dense figure itself."""
    raise NotImplementedError(
        "TODO 02.6: return the number a datasheet prints in big type: twice the dense figure when the stored one was derived from a with-sparsity number, else the dense figure, see the lesson's 'The star on the big number' section")


def rows():
    raise NotImplementedError(
        "TODO 02.6: return a dict per chip that publishes both numbers: name, gbs, dense, headline, ridge and wrong_ridge, see the lesson's 'The ridge points of the whole course' section")


if __name__ == "__main__":
    print(f"{'chip':<16}{'GB/s':>6}{'dense':>8}{'headline':>10}{'ridge':>7}{'if trusted':>11}")
    for r in rows():
        print(f"{r['name']:<16}{r['gbs']:>6}{r['dense']:>8.1f}{r['headline']:>10.1f}{r['ridge']:>7.0f}{r['wrong_ridge']:>11.0f}")
