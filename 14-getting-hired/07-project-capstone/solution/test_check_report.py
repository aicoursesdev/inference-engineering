import check_report as c


def report(label="measured"):
    rows = "\n".join(f"| {k} | file.py | 12 ms | {label} |" for k in c.LESSONS)
    return "\n".join(f"## {s}" for s in c.SECTIONS) + "\n" + rows + "\n"


def test_a_complete_report_passes():
    assert c.problems(report()) == []


def test_a_row_without_a_known_label_is_named():
    out = c.problems(report("guessed"))
    assert len(out) == len(c.LESSONS) and "label must be" in out[0]


def test_a_missing_section_and_row_are_named():
    text = report().replace("## Limits", "").replace("| 05.9 ", "| xx ")
    out = c.problems(text)
    assert "missing section: Limits" in out and "05.9: row missing" in out
