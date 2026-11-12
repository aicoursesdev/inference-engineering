"""Compare two canary runs and cheap reply checks: the gate a rollout should pass.

    uv run python gate.py canary-Q8_0.json canary-Q2_K.json --allow-drop 3
"""
import json, re, sys

STRAY = re.compile(r"[؀-ۿ฀-๿぀-ヿ一-鿿]")  # Arabic, Thai, Japanese kana, Chinese


def load(path):
    d = json.load(open(path))
    return d["score"], sum(bool(STRAY.search(r["reply"])) for r in d["results"]), d["total"]


base, new = load(sys.argv[1]), load(sys.argv[2])
allow = int(sys.argv[sys.argv.index("--allow-drop") + 1]) if "--allow-drop" in sys.argv else 3
print(f"canary score     {base[0]}/{base[2]} -> {new[0]}/{new[2]}")
print(f"stray-script replies {base[1]} -> {new[1]}")
raise NotImplementedError("TODO 12.5: ok when the score fell by no more than --allow-drop and the stray-script count did not rise, see the lesson's 'Watch the replies, not only the score' section")
print("GATE:", "pass" if ok else "FAIL: do not roll out")
sys.exit(0 if ok else 1)
