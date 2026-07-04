from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SUBPROCESS_TIMEOUT = 30


def test_setup_targets_use_selected_python_interpreter():
    custom_python = "/opt/custom/bin/python"
    result = subprocess.run(
        [
            "make",
            "-n",
            f"PYTHON={custom_python}",
            "install-torch-stack",
            "codespace-setup",
            "nlp-assets",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TEST_SUBPROCESS_TIMEOUT,
    )
    lines = result.stdout.splitlines()

    assert f"{custom_python} -m pip install --upgrade pip" in lines
    assert f"{custom_python} -m pip install -r torch-core-requirements.txt" in lines
    assert f"{custom_python} -m pip install --no-build-isolation -r torch-requirements.txt" in lines
    assert f"{custom_python} -m pip install -r requirements.txt" in lines
    assert f"{custom_python} -m spacy download en_core_web_sm" in lines
    assert any(line.startswith(f"{custom_python} -c ") for line in lines)
    assert not any(line.startswith("pip install") or line.startswith("python ") for line in lines)
