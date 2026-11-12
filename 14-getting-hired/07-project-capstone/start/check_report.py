"""Check a final report against the template: every section present, and every
module row filled with an artefact, a number and a label.
Usage: python check_report.py final-report.md"""
import re
import sys

SECTIONS = ["Headline", "The serving lab, module by module", "The deployed endpoint",
            "Cost and capacity", "Incidents", "Capstone piece", "Limits", "Reproduce"]
LABELS = {"measured", "simulated", "sourced"}
LESSONS = ["00.6", "01.8", "02.7", "03.9", "04.8", "05.9", "06.6", "07.9", "08.5",
           "09.8", "10.9", "11.6", "12.7", "13.8"]


def problems(text):
    out = [f"missing section: {s}" for s in SECTIONS if f"## {s}" not in text]
    rows = {m.group(1): [c.strip() for c in m.group(2).split("|")]
            for m in re.finditer(r"^\| (\d\d\.\d) [^|]*\|(.*)\|\s*$", text, re.M)}
    for k in LESSONS:
        cells = rows.get(k)
        if cells is None:
            out.append(f"{k}: row missing")
        elif len(cells) < 3 or not all(cells[:3]):
            out.append(f"{k}: artefact, number or label is empty")
        elif cells[2].lower() not in LABELS:
            out.append(f"{k}: label must be one of {sorted(LABELS)}")
    return out


if __name__ == "__main__":
    found = problems(open(sys.argv[1]).read())
    print("\n".join(found) or "ok: the report has every section and every row")
    sys.exit(1 if found else 0)
