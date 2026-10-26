"""DAT19/DAT20: a refused BYOD input renders exactly one line, ``Refused: <file>: <reason and how to fix it>``.

The Section 4 cell is executed from the committed notebook (the generator's output), with the model-side helpers
stubbed, so these tests exercise the code a learner runs.
"""

# ruff: noqa: E501  -- quoted notebook sentences are kept on one line for grep-ability
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


TEMPLATE = _load("notebook_template_dat20", ROOT / "tools" / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
STEM = TEMPLATE["stem"]
REFUSAL_CELL_ID = "pix2struct_textcaps-17"


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _section4() -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found = [c for c in nb["cells"] if c["cell_type"] == "code" and "BYOD_PATH = ''  # @param" in _src(c)]
    assert len(found) == 1
    return found[0]


def _byod_block() -> str:
    source = _src(_section4())
    defs = "class RunInvalidated" + source.split("class RunInvalidated", 1)[1].split("invalid_marker = ", 1)[0]
    tail = "invalid_marker = " + source.split("invalid_marker = ", 1)[1].split("\nelse:\n", 1)[0]
    return defs + "\n" + tail


def _run(tmp_path: Path, byod_path: str, into: dict | None = None, **stubs):
    (tmp_path / "outputs").mkdir(exist_ok=True)
    ns = into if into is not None else {}
    ns |= {
        "Path": Path, "os": os, "io": io, "json": json, "time": time, "zipfile": zipfile, "shutil": shutil,
        "USE_BYOD": True, "BYOD_PATH": byod_path, "SPLIT_SEED": 42, "MAX_RECORDS": 5000, "MIN_RECORDS": 8,
        "load_byod_dataset": lambda path: [], "split_dataset": lambda records, seed, base_dir: {},
        "validate_dataset": lambda part: {}, "RUN_STATE": {"valid": True}, "splits": {"train": []},
    }
    ns.update(stubs)
    with contextlib.chdir(tmp_path):
        exec(compile(_byod_block(), "<section 4>", "exec"), ns)
    return ns


def _refusal(tmp_path: Path, byod_path: str, **stubs) -> tuple[BaseException, dict]:
    ns: dict = {}
    with pytest.raises(Exception) as caught:
        _run(tmp_path, byod_path, into=ns, **stubs)
    return caught.value, ns


def _assert_one_refused_line(exc: BaseException, file_name: str) -> None:
    assert type(exc).__name__ == "RunInvalidated" and getattr(exc, "dimer_stop", False) is True
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    assert rendered[0] == str(exc)
    assert rendered[0].startswith(f"Refused: {file_name}: "), rendered[0]
    assert rendered[0].count("Refused:") == 1 and "BYOD refused" not in rendered[0]
    assert rendered[0].endswith("fix the named file or rule and run this cell again.")


def _zip(tmp_path: Path, name: str = "mine.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("records.jsonl", '{"id": "a"}\n')
    return path


def test_the_refusal_cell_id_is_the_section_4_cell() -> None:
    assert _section4().get("id") == REFUSAL_CELL_ID


def test_scripted_refusal_etc_hostname_is_one_refused_line(tmp_path: Path) -> None:
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a file that is not a .zip."""
    hostname = tmp_path / "etc" / "hostname"
    hostname.parent.mkdir()
    hostname.write_text("colab\n", encoding="utf-8")
    exc, _ = _refusal(tmp_path, str(hostname))
    _assert_one_refused_line(exc, "hostname")
    assert "ending in .zip" in str(exc)


def test_a_directory_path_is_refused_naming_the_path(tmp_path: Path) -> None:
    folder = tmp_path / "etc"
    folder.mkdir()
    exc, _ = _refusal(tmp_path, str(folder))
    _assert_one_refused_line(exc, str(folder))
    assert "BYOD_PATH does not exist or is not a file" in str(exc)


def test_an_empty_path_outside_colab_is_refused_as_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "google.colab", None)
    exc, _ = _refusal(tmp_path, "")
    _assert_one_refused_line(exc, "(no file)")


def test_a_validate_dataset_refusal_names_the_zip(tmp_path: Path) -> None:
    zip_path = _zip(tmp_path)

    def bad(part):
        raise ValueError("record 3: caption is empty; give one or more non-empty captions")

    splits = {name: [{"id": str(i)} for i in range(8)] for name in ("train", "validation", "test")}
    exc, _ = _refusal(tmp_path, str(zip_path), split_dataset=lambda records, seed, base_dir: splits, validate_dataset=bad)
    _assert_one_refused_line(exc, "mine.zip")
    assert "record 3: caption is empty" in str(exc)


def test_a_refusal_drops_earlier_results_and_invalidates_the_run(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / f"{STEM}_result.json").write_text("{}", encoding="utf-8")
    ns: dict = {}
    with pytest.raises(Exception) as caught:
        _run(tmp_path, str(tmp_path / "missing.zip"), into=ns, train_records=[1])
    assert "splits" not in ns and "train_records" not in ns and ns["RUN_STATE"]["valid"] is False
    _assert_one_refused_line(caught.value, str(tmp_path / "missing.zip"))
    marker = json.loads((tmp_path / "outputs" / f"{STEM}_run_invalidated.json").read_text(encoding="utf-8"))
    assert marker["run_valid"] is False and marker["refusal"].startswith(str(tmp_path / "missing.zip") + ": ")


def test_a_genuine_error_keeps_its_type_and_traceback(tmp_path: Path) -> None:
    zip_path = _zip(tmp_path)

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(tmp_path, str(zip_path), load_byod_dataset=broken)
    assert type(caught.value).__name__ == "RuntimeError" and not getattr(caught.value, "dimer_stop", False)


def test_the_troubleshooting_entry_quotes_the_new_prefix() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(_src(c) for c in nb["cells"])
    assert "BYOD refused:" not in text
    assert "- **Refused: <zip>:" in text
