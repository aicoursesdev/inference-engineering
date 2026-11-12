import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def test_the_kernel_is_written():
    text = (HERE / "vec_add_kernel.cuh").read_text()
    todo = re.search(r"TODO 16\.2.*", text)
    if todo:
        pytest.fail(todo.group(), pytrace=False)


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_the_kernel_lines_are_right_on_the_cpu(tmp_path):
    exe = tmp_path / "check"
    subprocess.run(["c++", "-std=c++17", "-I", str(HERE),
                    str(HERE / "cpu_check.cpp"), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "4 blocks of 256 threads = 1024 threads for 1000 elements; wrong: 0" in out.stdout
