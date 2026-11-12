"""The lesson runner: list lessons, check your work, compare with the solution.

    uv run python scripts/lesson.py list
    uv run python scripts/lesson.py check 02.4 [--solution]
    uv run python scripts/lesson.py diff 02.4
    uv run python scripts/lesson.py all

`make check 02.4`, `make diff 02.4` and `make test` run these. A lesson id is
the module number and the lesson number: 02.4 (2.4 and 02.04 work too).

Uses only Python's standard library, so it runs before anything else works.
"""

import argparse
import difflib
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folders the comparison and the test hunt never look into.
SKIP = {"cassettes", "results", "outputs", "__pycache__", "node_modules",
        ".venv", ".git", ".pytest_cache"}
# Files `diff` shows: code and text. Anything else (images, databases) is not.
TEXT = {".py", ".md", ".ts", ".tsx", ".json", ".toml", ".yml", ".yaml",
        ".txt", ".html", ".css", ".sql"}
PYTEST = [sys.executable, "-m", "pytest", "-q",
          "-p", "no:cacheprovider"]


class Lesson:
    """One lesson folder, <module>/<lesson>/, with start/ and solution/."""

    def __init__(self, module_dir, lesson_dir):
        self.module = int(module_dir.name[:2])
        self.number = int(lesson_dir.name[:2])
        self.id = f"{self.module:02d}.{self.number}"
        self.path = lesson_dir
        self.start = lesson_dir / "start"
        self.solution = lesson_dir / "solution"


def numbered(folder):
    """The sub-folders named like 02-something, in order."""
    return sorted(p for p in folder.iterdir()
                  if p.is_dir() and re.match(r"\d\d-", p.name))


def lessons():
    return [Lesson(m, l) for m in numbered(ROOT) for l in numbered(m)]


def find(text):
    """The lesson for an id like 02.4, 2.4 or 02.04."""
    match = re.fullmatch(r"(\d+)\.(\d+)", text.strip())
    if match:
        wanted = (int(match[1]), int(match[2]))
        for lesson in lessons():
            if (lesson.module, lesson.number) == wanted:
                return lesson
    sys.exit(f"No lesson {text!r}. Lesson ids look like 02.4; "
             "`uv run python scripts/lesson.py list` shows them all.")


def walk(folder, skip=SKIP):
    """Every file under folder, as paths relative to it, minus skipped dirs."""
    found = []
    for here, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        found += [Path(here, f).relative_to(folder) for f in files]
    return sorted(found)


def is_test(path):
    return path.suffix == ".py" and (path.name.startswith("test_")
                                     or path.name.endswith("_test.py"))


def tests_in(folder):
    return [p for p in walk(folder) if is_test(p)] if folder.is_dir() else []


def shared_files(folder):
    """The files start/ and solution/ must hold identically: tests, their
    conftest.py or pytest.ini, and every recorded cassette."""
    if not folder.is_dir():
        return {}
    return {p: folder / p for p in walk(folder, skip=SKIP - {"cassettes"})
            if is_test(p) or p.name in ("conftest.py", "pytest.ini")
            or "cassettes" in p.parts}


def differing_shared_files(lesson):
    """Tests and cassettes that differ between start/ and solution/."""
    a, b = shared_files(lesson.start), shared_files(lesson.solution)
    return [p.as_posix() for p in sorted(set(a) | set(b))
            if p not in a or p not in b
            or a[p].read_bytes() != b[p].read_bytes()]


def missing_tools():
    return [name for name in ("pytest",)
            if importlib.util.find_spec(name) is None]


def run_pytest(folder, extra=(), echo=True):
    """Run pytest in folder; return its exit code and everything it printed."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(
        [*PYTEST, *extra], cwd=folder, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace")
    lines = []
    for line in process.stdout:
        lines.append(line)
        if echo:
            print(line, end="", flush=True)
    return process.wait(), "".join(lines)


def failures(output):
    """How many tests failed or errored, read from pytest's summary line
    ("10 failed, 2 passed in 0.30s"), the last line that ends in a time:
    a library may print a warning after it."""
    summary = next((line for line in reversed(output.strip().splitlines())
                    if re.search(r" in [\d.]+s\b", line)), "")
    return sum(int(n) for n, _ in re.findall(r"(\d+) (failed|errors?)\b",
                                             summary))


# ---- list ------------------------------------------------------------------

def cmd_list(_args):
    print("lesson  start  solution  tests  folder")
    for lesson in lessons():
        tests = len(tests_in(lesson.start)) or len(tests_in(lesson.solution))
        print(f"{lesson.id:<7} {'yes' if lesson.start.is_dir() else '-':<6} "
              f"{'yes' if lesson.solution.is_dir() else '-':<9} "
              f"{tests or '-':<6} {lesson.path.relative_to(ROOT).as_posix()}")
    return 0


# ---- check -----------------------------------------------------------------

def cmd_check(args):
    lesson = find(args.lesson)
    where = lesson.solution if args.solution else lesson.start
    if not where.is_dir():
        print(f"{lesson.id}: there is no {where.name}/ folder yet. "
              "Nothing to check.")
        return 0
    if not tests_in(where):
        print(f"{lesson.id}: this lesson has no tests. Nothing to check.")
        return 0
    if missing_tools():
        print("The test tools are not installed. Run: uv sync --extra tests")
        return 1
    code, output = run_pytest(where)
    print()
    failing = failures(output)
    if code == 0 and args.solution:
        print(f"{lesson.id}: all tests pass in solution/.")
    elif code == 0:
        print(f"{lesson.id}: all tests pass. Done.")
    elif failing and args.solution:
        print(f"{lesson.id}: {failing} failing in solution/.")
    elif failing:
        print(f"{lesson.id}: {failing} failing. Read the first TODO above; "
              f"stuck? `make diff {lesson.id}`")
    else:
        print(f"{lesson.id}: pytest stopped before it could run the tests "
              f"in {where.name}/ (exit code {code}). Read the message above.")
    return code


# ---- diff ------------------------------------------------------------------

def text_of(path):
    """The lines of a code or text file; None for a binary one."""
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def cmd_diff(args):
    lesson = find(args.lesson)
    if not lesson.start.is_dir() or not lesson.solution.is_dir():
        print(f"{lesson.id}: needs both start/ and solution/ to compare.")
        return 0
    changed = []
    for name in sorted(set(walk(lesson.start)) | set(walk(lesson.solution))):
        if name.suffix not in TEXT:
            continue
        a, b = lesson.start / name, lesson.solution / name
        old = text_of(a) if a.is_file() else []
        new = text_of(b) if b.is_file() else []
        if old is None or new is None or old == new:
            continue
        changed.append(name.as_posix())
        for line in difflib.unified_diff(
                old, new, f"start/{name.as_posix()}",
                f"solution/{name.as_posix()}", lineterm=""):
            print(line)
    if changed:
        print(f"\n{lesson.id}: solution/ differs from start/ in "
              f"{len(changed)} file{'s' if len(changed) != 1 else ''}: "
              f"{', '.join(changed)}")
    else:
        print(f"{lesson.id}: start/ and solution/ are the same.")
    return 0


# ---- all -------------------------------------------------------------------

def check_lesson(lesson):
    """Everything wrong with one lesson's start/, solution/ and tests."""
    problems = []
    if not lesson.solution.is_dir():
        return ["start/ has tests but there is no solution/"]
    problems += [f"{name} differs between start/ and solution/"
                 for name in differing_shared_files(lesson)]
    code, output = run_pytest(lesson.solution, echo=False)
    if code != 0:
        problems.append("solution/ fails its tests")
        print(f"--- {lesson.id} solution/\n{output}")
    # Collecting runs no test but imports every file: stubs must import.
    code, output = run_pytest(lesson.start, ["--collect-only"], echo=False)
    if code != 0:
        problems.append("start/ cannot be collected (an import error?)")
        print(f"--- {lesson.id} start/\n{output}")
    return problems


def cmd_all(_args):
    if missing_tools():
        print("The test tools are not installed. Run: uv sync --extra tests")
        return 1
    rows, waiting, nothing = [], [], 0
    for lesson in lessons():
        if lesson.start.is_dir() and (tests_in(lesson.start)
                                      or tests_in(lesson.solution)):
            began = time.perf_counter()
            rows.append((lesson, check_lesson(lesson),
                         time.perf_counter() - began))
        elif tests_in(lesson.solution):
            waiting.append(lesson.id)
        else:
            nothing += 1

    print(f"\n{'lesson':<7} {'result':<7} {'seconds':>7}  folder")
    for lesson, problems, seconds in rows:
        print(f"{lesson.id:<7} {'FAIL' if problems else 'pass':<7} "
              f"{seconds:7.1f}  {lesson.path.relative_to(ROOT).as_posix()}")
        for problem in problems:
            print(f"        - {problem}")
    failed = sum(1 for _, problems, _ in rows if problems)
    print(f"\n{len(rows)} lessons checked: {len(rows) - failed} pass, "
          f"{failed} fail.")
    print(f"{nothing + len(waiting)} other lessons have no tests yet"
          + (f" ({len(waiting)} of them have tests in solution/, waiting "
             "for a start/ folder)." if waiting else "."))
    return 1 if failed else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="List lessons, check your work, compare with the "
                    "solution.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every lesson folder and what it has")
    check = commands.add_parser("check", help="run a lesson's tests")
    check.add_argument("lesson", help="a lesson id such as 02.4")
    check.add_argument("--solution", action="store_true",
                       help="run the tests on solution/ instead of start/")
    diff = commands.add_parser("diff", help="show start/ against solution/")
    diff.add_argument("lesson", help="a lesson id such as 02.4")
    commands.add_parser("all", help="check every solution and its tests")
    args = parser.parse_args()
    sys.exit({"list": cmd_list, "check": cmd_check, "diff": cmd_diff,
              "all": cmd_all}[args.command](args))


if __name__ == "__main__":
    main()
