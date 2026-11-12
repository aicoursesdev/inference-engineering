"""Rebuild the two text samples in this folder from wikitext-2 (Wikipedia articles).

    uv run python data/text/make_samples.py

Downloads wikitext-2-raw-v1 (the zip llama.cpp's own scripts use, from
Hugging Face's ggml-org/ci dataset), and writes the first 40,000 characters of
its train split and the first 6,000 of its test split, with the blank lines
and "= Heading =" lines left out. Lessons 07.3 and 07.4 read them.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
HERE = Path(__file__).parent


def body(raw, chars):
    """The first `chars` characters of the article text, headings dropped."""
    lines = [l.strip() for l in raw.splitlines()]
    keep = [l for l in lines if l and not l.startswith("=")]
    return " ".join(keep)[:chars]


if __name__ == "__main__":
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    for split, chars in (("train", 40_000), ("test", 6_000)):
        raw = z.read(f"wikitext-2-raw/wiki.{split}.raw").decode("utf-8")
        (HERE / f"wikitext2-{split}-sample.txt").write_text(body(raw, chars), encoding="utf-8")
