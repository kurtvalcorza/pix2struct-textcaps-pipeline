"""Regression tests for the 2026-10-17 row-64 review (T64-M1, T64-m1..m6, T64-S1): the pre-registered learning-rate
grid, the fresh test and the score-once rule, the coco-caption ROUGE-L aggregation, spec 2.5, the DAT9 wording and
the DAT20 BYOD refusal path. Offline: no model download and no network."""

# ruff: noqa: E501  -- quoted notebook sentences and fixture rows are kept on one line for grep-ability
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
from pathlib import Path

import pytest

from pix2struct_textcaps_pipeline import metrics, samples
from pix2struct_textcaps_pipeline.metrics import METRIC_DEFINITIONS, rouge_l
from pix2struct_textcaps_pipeline.pipeline import caption_tokens

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


build = _load("build_notebook_row64", TOOLS / "build_notebook.py")
TEMPLATE = _load("notebook_template_row64", TOOLS / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
STEM = TEMPLATE["stem"]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


# --- T64-m3: ROUGE-L follows the coco-caption aggregation; tokenisation is labelled honestly ------------------------


def _coco_rouge(hyp: str, refs: list[str], beta: float = 1.2) -> float:
    """Independent re-implementation of pycocoevalcap Rouge.calc_score (max P and max R taken separately)."""
    h = caption_tokens(hyp)
    precisions, recalls = [], []
    for ref in refs:
        r = caption_tokens(ref)
        lcs = metrics._lcs(h, r)
        precisions.append(lcs / len(h) if h else 0.0)
        recalls.append(lcs / len(r) if r else 0.0)
    p, rc = max(precisions), max(recalls)
    return ((1 + beta**2) * p * rc) / (rc + beta**2 * p) if p and rc else 0.0


def test_t64_m3_rouge_l_uses_max_precision_and_max_recall() -> None:
    """The review's constructed case: best-per-reference F gave 0.569, coco-caption gives 1.000."""
    hyp = "a red soup can on a table"
    refs = ["a red soup can on a kitchen table next to a window with a long label", "soup can"]
    assert rouge_l(hyp, refs) == pytest.approx(1.0)
    assert rouge_l(hyp, refs) == pytest.approx(_coco_rouge(hyp, refs))
    best_per_reference_f = max(
        (1 + 1.44)
        * (lcs / 7)
        * (lcs / len(caption_tokens(r)))
        / ((lcs / len(caption_tokens(r))) + 1.44 * (lcs / 7))
        for r in refs
        for lcs in [metrics._lcs(caption_tokens(hyp), caption_tokens(r))]
    )
    assert best_per_reference_f == pytest.approx(0.569, abs=5e-4)  # the old aggregation, for the record


@pytest.mark.parametrize(
    ("hyp", "refs"),
    [
        ("a dog", ["a dog on grass", "dog"]),
        ("nothing in common", ["a can of soup"]),
        ("a bottle of water on a desk", ["bottle", "a water bottle on a wooden desk", "a desk"]),
    ],
)
def test_t64_m3_rouge_l_matches_the_independent_coco_implementation(hyp: str, refs: list[str]) -> None:
    assert rouge_l(hyp, refs) == pytest.approx(_coco_rouge(hyp, refs))


def test_t64_m3_metric_text_names_the_aggregation_and_the_tokeniser(nb: dict) -> None:
    assert "best precision and the best" in METRIC_DEFINITIONS["rouge_l"]
    assert "not the PTB tokenizer" in metrics.__doc__
    md = _markdown(nb)
    assert "own implementations of the `coco-caption` definitions" not in md
    assert "own pure-Python implementations of the `coco-caption` definitions" not in md
    assert "**not the PTB tokenizer**" in md and "not the PTB tokenizer" in md


# --- T64-M1: grid selection rules ----------------------------------------------------------------------------------


def test_t64_M1_default_learning_rate_is_kept_inside_the_margin() -> None:
    sel = samples.select_learning_rate({1e-5: 0.275, 2e-5: 0.290, 5e-5: 0.2949})
    assert sel["chosen_lr"] == 1e-5 and sel["best_challenger"]["lr"] == 5e-5


def test_t64_M1_challenger_at_the_margin_wins_and_the_higher_challenger_wins() -> None:
    assert samples.select_learning_rate({1e-5: 0.275, 2e-5: 0.295, 5e-5: 0.280})["chosen_lr"] == 2e-5
    assert samples.select_learning_rate({1e-5: 0.275, 2e-5: 0.300, 5e-5: 0.310})["chosen_lr"] == 5e-5
    assert (
        samples.select_learning_rate({1e-5: 0.275, 2e-5: 0.310, 5e-5: 0.310})["chosen_lr"] == 2e-5
    )  # tie: smaller
    with pytest.raises(ValueError, match="default"):
        samples.select_learning_rate({2e-5: 0.3, 5e-5: 0.3})


def test_t64_M1_grid_score_ignores_the_frozen_epoch() -> None:
    history = [
        {"epoch": 0, "val": {"cider_d": 0.9}},
        {"epoch": 1, "val": {"cider_d": 0.2}},
        {"epoch": 2, "val": {"cider_d": 0.25}},
    ]
    assert samples.best_adapted_validation(history) == 0.25


def test_t64_M1_preregistration_doc_matches_the_module() -> None:
    doc = (ROOT / "docs" / "learning-rate-grid-preregistration.md").read_text(encoding="utf-8")
    assert samples.LR_GRID == (1e-5, 2e-5, 5e-5) and samples.LR_DEFAULT == 1e-5
    assert samples.LR_MARGIN == 0.02 and samples.FROZEN_MARGIN == 0.02
    assert "at least 0.02" in doc and "row group 2" in doc and "Hamming" in doc and "distance **10**" in doc
    assert samples.NEAR_DUPLICATE_MAX_DISTANCE == 10
    assert (
        samples.PREREGISTERED_SETTINGS["run_grid"] is True and samples.PREREGISTERED_SETTINGS["epochs"] == 4
    )
    amendment = doc.split("## Amendment 1 (2026-10-17)", 1)[1]
    assert "208 training photographs" in amendment and "recorded manual review" in amendment
    assert (
        "Secondary:** the full unscreened captioned pool" in amendment
        and "paired bootstrap 95 %" in amendment
    )
    for key in (
        "weight_decay",
        "grad_clip",
        "scheduler",
        "decoding",
        "validation_max_new_tokens",
        "model_revision",
        "fresh_test_ids_sha256",
    ):
        assert key in samples.PREREGISTERED_SETTINGS


def test_t64_M1_fresh_test_pins_are_disjoint_and_consistent() -> None:
    assert samples.FRESH_TEST_ROW_GROUP == 2 and samples.SHARD_ROW_GROUP_ROWS == (336, 336, 336, 336, 206)
    assert len(samples.FRESH_IMAGE_PINS) == 336 and not set(samples.FRESH_IMAGE_PINS) & set(
        samples.IMAGE_PINS
    )
    assert (
        set(samples.FRESH_TEST_EXCLUDED)
        <= set(samples.FRESH_TEST_CANDIDATES)
        <= set(samples.FRESH_IMAGE_PINS)
    )
    assert len(samples.FRESH_TEST_UNCAPTIONED) == 12 and samples.FRESH_POOL_SIZE == 336 - 12
    assert len(samples.FRESH_TEST_CANDIDATES) == 41 and samples.FRESH_TEST_EXCLUDED == ()
    assert samples.FRESH_POOL_SIZE - len(samples.FRESH_TEST_EXCLUDED) == samples.FRESH_TEST_SIZE
    assert set(samples.FRESH_TEST_SCREEN_REVIEW) == set(samples.FRESH_TEST_CANDIDATES)
    assert all(isinstance(ok, bool) and reason for ok, reason in samples.FRESH_TEST_SCREEN_REVIEW.values())
    assert re.fullmatch(r"[0-9a-f]{64}", samples.FRESH_POOL_IDS_SHA256)
    assert re.fullmatch(r"[0-9a-f]{64}", samples.FRESH_TEST_IDS_SHA256)


def test_t64_S1_near_duplicate_screen_flags_a_retake(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    def scene(name: str, shift: int, colour: str) -> Path:
        image = Image.new("RGB", (160, 120), "white")
        ImageDraw.Draw(image).rectangle([20 + shift, 20, 90 + shift, 100], fill=colour)
        path = tmp_path / name
        image.save(path)
        return path

    reference = {"r1": scene("r1.png", 0, "black")}
    candidates = {"retake": scene("c1.png", 1, "black"), "other": scene("c2.png", 60, "black")}
    screen = samples.near_duplicate_screen(candidates, reference, max_distance=10)
    assert "retake" in screen["flagged"] and "other" not in screen["flagged"]


def test_t64_M1_build_fresh_test_refuses_a_screen_that_disagrees_with_the_pin(
    tmp_path: Path, monkeypatch
) -> None:
    from PIL import Image

    ids = sorted(samples.FRESH_IMAGE_PINS)[:3]
    path = tmp_path / "x.png"
    Image.new("RGB", (32, 32), "grey").save(path)
    annotations = [{"id": i, "captions": ["a"], "text_detected": False} for i in ids]
    monkeypatch.setattr(samples, "FRESH_IMAGE_PINS", {i: ("0" * 64, 1) for i in ids})
    with pytest.raises(ValueError, match="the candidate screen found"):
        samples.build_fresh_test(annotations, {i: path for i in ids}, {"29631": path})


# --- T64-m5: score-once rule -----------------------------------------------------------------------------------------


def _settings(**changes):
    return {**samples.PREREGISTERED_SETTINGS, **changes}


@pytest.mark.parametrize(
    "changes",
    [
        {"epochs": 3},
        {"learning_rate": 2e-5},
        {"batch_size": 4},
        {"split_seed": 7},
        {"byod": True},
        {"data_source": "BYOD (mine.zip)"},
        {"run_grid": False},
        {"trainable_decoder_layers": 1},
        {"caption_max_tokens": 40},
        {"weight_decay": 0.0},
        {"model_revision": "main"},
        {"validation_max_new_tokens": 40},
        {"fresh_test_ids_sha256": "0" * 64},
    ],
)
def test_t64_m5_a_changed_setting_never_scores_the_fresh_test(changes, tmp_path: Path) -> None:
    check = samples.check_preregistration(_settings(**changes))
    assert check["preregistered"] is False and set(check["settings_differing"]) == set(changes)
    look = samples.plan_test_look(check, tmp_path, STEM)
    assert look["test_look"] == "not scored (exploratory)" and look["fresh_test_scored"] is False
    assert look["suffix"] == "_exploratory_1"


def test_t64_m5_a_bool_never_stands_in_for_a_number() -> None:
    assert samples.check_preregistration(_settings(adapt_seed=False))["preregistered"] is False


def test_t64_m5_identical_reruns_are_reproductions(tmp_path: Path) -> None:
    check = samples.check_preregistration(_settings())
    assert check["preregistered"] is True
    first = samples.plan_test_look(check, tmp_path, STEM)
    assert first == {
        **first,
        "test_look": "pre-registered evaluation (this runtime)",
        "suffix": "",
        "fresh_test_scored": True,
    }
    (tmp_path / f"{STEM}_evaluation_report.json").write_text("{}", encoding="utf-8")
    again = samples.plan_test_look(check, tmp_path, STEM)
    assert again["test_look"] == "reproduction" and again["suffix"] == "_reproduction_1"
    later = samples.plan_test_look(
        check, tmp_path / "fresh-runtime", STEM, recorded_evaluation="release-verification row"
    )
    assert later["test_look"] == "reproduction" and later["suffix"] == ""


def test_t64_m5_missing_or_unknown_settings_are_refused() -> None:
    settings = _settings()
    settings.pop("epochs")
    with pytest.raises(ValueError, match="missing"):
        samples.check_preregistration(settings)


def _make_pipe(calls, state):
    pipe = _Pipe(calls, "adapted")
    pipe.state = state
    return pipe


class _Pipe:
    def __init__(self, calls, label):
        self.calls, self.label, self.adapter = calls, label, {"best_epoch": 4, "lr": 1e-5}
        self.state = "abc"

    def adapter_state_sha256(self):
        return self.state

    def evaluate(self, records, max_new_tokens=30):
        self.calls.append((self.label, len(records)))
        return {
            "bleu4": 0.1,
            "rouge_l": 0.3,
            "cider_d": 0.3,
            "unigram_f1": 0.4,
            "mean_words": 8.0,
            "n": len(records),
            "adapted": self.label != "frozen",
            "predictions": {r["id"]: "a caption" for r in records},
        }


def _section8(nb: dict, tmp_path: Path, settings: dict, state: str = "abc") -> tuple[dict, list]:
    source = _cell_with(nb, "selected_state = {")
    calls: list = []
    recs = [
        {"id": f"t{i}", "category": "text" if i % 2 else "no-text", "captions": ["a caption"]}
        for i in range(4)
    ]
    fresh = [
        {"id": f"f{i}", "category": "text" if i % 2 else "no-text", "captions": ["a caption"]}
        for i in range(5)
    ]
    base = {"bleu4": 0.0, "rouge_l": 0.0, "cider_d": 0.0, "unigram_f1": 0.0, "mean_words": 3.0, "n": 4}
    fields = {"text": {"n": 2, "cider_d": 0.1}, "no-text": {"n": 2, "cider_d": 0.1}}

    class _Pipeline:
        @staticmethod
        def from_pretrained(weights_dir):
            return _Pipe(calls, "frozen")

    ns = {
        "Path": Path,
        "json": json,
        "require_valid_run": lambda: None,
        "plan_test_look": samples.plan_test_look,
        "prereg": samples.check_preregistration(settings),
        "pipe": _make_pipe(calls, state),
        "test_records": recs,
        "val_records": recs[:2],
        "fresh_records": fresh,
        "fresh_pool_records": fresh,
        "fresh_report": {"row_group": 2, "excluded": {}, "uncaptioned": ["30317"]},
        "frozen_test": {**base, "adapted": False},
        "baseline_constant": base,
        "baseline_neighbour": base,
        "constant_fields": fields,
        "frozen_fields": fields,
        "frozen_predictions": {r["id"]: "x" for r in recs},
        "by_category": lambda preds, records: fields,
        "adapt_result": {"best_epoch": 4, "history": []},
        "chosen_lr": 1e-5,
        "METRICS": ("bleu4", "rouge_l", "cider_d", "unigram_f1"),
        "CAPTION_MAX_TOKENS": 30,
        "constant_caption_baseline": lambda tr, te: base,
        "colour_neighbour_baseline": lambda tr, te: base,
        "train_records": recs,
        "medoid": "m",
        "Pix2StructTextCapsPipeline": _Pipeline,
        "WEIGHTS_DIR": "w",
        "system": "adapted model",
        "USE_BYOD": settings["byod"],
        "TEST_LABEL": "exposed development test",
        "MODEL_ID": "m",
        "MODEL_REVISION": "r",
        "MODEL_KEY": "k",
        "data_source": "d",
        "dataset_manifests": {},
        "disjoint": {},
        "categories": {},
        "grid_runs": {1e-5: {"state_sha256": "abc", "result": {"history": []}}},
        "selection": {},
        "adapt_seconds": 1.0,
        "frozen_beats_constant": True,
        "reference_captions": lambda r: r["captions"],
        "grid_dir": Path("g"),
        "set_profile": samples.set_profile,
        "caption_metrics": metrics.caption_metrics,
        "cider_d": metrics.cider_d,
        "paired_bootstrap_ci": metrics.paired_bootstrap_ci,
    }
    with contextlib.chdir(tmp_path), contextlib.redirect_stdout(io.StringIO()):
        (tmp_path / "outputs").mkdir(exist_ok=True)
        exec(compile(source, "<section 8>", "exec"), ns)
    return ns, calls


@pytest.mark.parametrize(
    "changes", [{"epochs": 3}, {"learning_rate": 2e-5}, {"batch_size": 4}, {"split_seed": 7}]
)
def test_t64_m5_section_8_does_not_score_the_fresh_test_when_exploratory(
    nb: dict, tmp_path: Path, changes
) -> None:
    ns, calls = _section8(nb, tmp_path, _settings(**changes))
    assert ("frozen", 5) not in calls and ("adapted", 5) not in calls
    assert ns["fresh"] is None and ns["evaluation_report_payload"]["test_look"] == "not scored (exploratory)"
    assert (tmp_path / "outputs" / f"{STEM}_evaluation_report_exploratory_1.json").is_file()
    assert not (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").exists()


def test_t64_m5_section_8_scores_the_fresh_test_once_under_the_preregistration(
    nb: dict, tmp_path: Path
) -> None:
    ns, calls = _section8(nb, tmp_path, _settings())
    assert calls.count(("frozen", 5)) == 1 and calls.count(("adapted", 5)) == 1
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert (
        report["test_look"] == "pre-registered evaluation (this runtime)"
        and report["settings_differing"] == []
    )
    ns, calls = _section8(nb, tmp_path, _settings())
    assert ns["evaluation_report_payload"]["test_look"] == "reproduction"
    assert (tmp_path / "outputs" / f"{STEM}_evaluation_report_reproduction_1.json").is_file()


def test_t64_m5_grid_members_start_from_the_frozen_weights_and_the_experiment_is_exploratory(
    nb: dict,
) -> None:
    seven = _cell_with(nb, "RUN_GRID = True  # @param")
    loop = seven.split("for lr in grid_lrs:", 1)[1]
    assert loop.index(
        "candidate = Pix2StructTextCapsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)"
    ) < loop.index("candidate.adapt(")
    assert "prereg = check_preregistration(run_settings)" in seven
    experiment = _cell_with(nb, "RUN_BLOCK_COMPARISON = False  # @param")
    assert "check_preregistration({**run_settings, 'trainable_decoder_layers': 1})" in experiment
    assert "fresh_records" not in experiment


# --- T64-m6: BYOD refusals (DAT20) ----------------------------------------------------------------------------------


def _section4_byod_block(nb: dict) -> str:
    source = _cell_with(nb, "BYOD_PATH = ''  # @param")
    head = source.split("def byod_file(", 1)[0]
    defs = "class RunInvalidated" + head.split("class RunInvalidated", 1)[1]
    tail = "invalid_marker = " + source.split("invalid_marker = ", 1)[1].split("\nelse:\n", 1)[0]
    return defs + "\n" + tail


def test_t64_m6_a_refused_byod_zip_stops_without_a_traceback_and_invalidates_the_run(
    nb: dict, tmp_path: Path
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / f"{STEM}_result.json").write_text("{}", encoding="utf-8")

    def refuse():
        raise ValueError("mine.zip: split(s) {'test': 6} hold fewer than MIN_RECORDS = 8 records")

    ns = {
        "Path": Path,
        "json": json,
        "time": __import__("time"),
        "USE_BYOD": True,
        "load_byod_splits": refuse,
        "validate_dataset": None,
        "splits": {"train": []},
        "train_records": [1],
        "RUN_STATE": {"valid": True},
    }
    with contextlib.chdir(tmp_path), pytest.raises(Exception) as caught:
        exec(compile(_section4_byod_block(nb), "<section 4>", "exec"), ns)
    exc = caught.value
    assert type(exc).__name__ == "RunInvalidated" and getattr(exc, "dimer_stop", False) is True
    assert exc._render_traceback_() == [str(exc)] and str(exc).startswith("BYOD refused: mine.zip: split(s)")
    marker = json.loads((outputs / f"{STEM}_run_invalidated.json").read_text(encoding="utf-8"))
    assert marker["run_valid"] is False and marker["earlier_outputs_kept_not_results_for_this_input"] == [
        f"{STEM}_result.json"
    ]
    assert (outputs / f"{STEM}_result.json").is_file()  # kept, and labelled as not a result for this input
    assert "splits" not in ns and "train_records" not in ns and ns["RUN_STATE"]["valid"] is False
    with pytest.raises(Exception, match="This run is invalidated"):
        ns["require_valid_run"]()


def test_t64_m6_every_later_section_requires_a_valid_run(nb: dict) -> None:
    for marker in (
        "CAPTION_MAX_TOKENS = 30  # @param",
        "baseline_constant = constant_caption_baseline(",
        "RUN_GRID = True  # @param",
        "def fresh_summary(",
        "import csv",
    ):
        assert "require_valid_run()" in _cell_with(nb, marker)


def test_t64_m6_router_shows_a_deliberate_stop_as_its_message_only() -> None:
    router = (
        build._ISOLATED_ROUTER if isinstance(build._ISOLATED_ROUTER, str) else "".join(build._ISOLATED_ROUTER)
    )
    assert 'if getattr(exc, "dimer_stop", False):' in router and '_send.send(("stop", str(exc)))' in router
    assert 'elif kind == "stop":' in router and "raise IsolatedCellStop(message[1]) from None" in router
    assert "def _render_traceback_(self):" in router.split("class IsolatedCellStop", 1)[1]


# --- T64-m2: spec 2.5 migration --------------------------------------------------------------------------------------


def test_t64_m2_kind_notice_env_and_ieee_references(nb: dict) -> None:
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    assert any(_src(c).startswith("> **⚠️ Notice.**") for c in nb["cells"][:first_code])
    md = _markdown(nb)
    assert "`DIMER_ISOLATED_ENV`" in md
    refs = md.split("## References", 1)[1]
    assert all(f"[{n}]" in refs for n in range(1, 7)) and "Lee, K., Joshi" not in refs
    assert "Pix2Struct model [1]" in md


def test_t64_m2_stamp_names_a_commit_that_contains_the_carried_modules(nb: dict) -> None:
    revision = nb["metadata"]["dimer"]["generated_from"]["revision"]
    try:
        for module in TEMPLATE["modules"]:
            rel = f"src/{TEMPLATE['package']}/{module}"
            committed = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{revision}:{rel}"], stderr=subprocess.DEVNULL
            )
            assert committed.replace(b"\r\n", b"\n") == (ROOT / rel).read_bytes().replace(b"\r\n", b"\n"), rel
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the generated_from revision is not in this clone (shallow CI checkout)")


# --- T64-M1 / T64-m1 / T64-m4: wording ------------------------------------------------------------------------------


def test_t64_M1_no_markdown_calls_the_exposed_test_unused_for_selection(nb: dict) -> None:
    md = _markdown(nb)
    assert "never used for training or epoch selection" not in md
    assert "records never used for training or selection" not in md
    assert "exposed development test" in md and "BLIP captioning row" in md and "2b72c3f7" in md
    assert "fresh test" in md and "test_look" in md


def test_t64_m1_quoted_numbers_name_their_blob(nb: dict) -> None:
    md = _markdown(nb)
    for stale in ("has no recorded number yet", "325.5 s", "868.0 s", "25 September 2026"):
        assert stale not in md, stale
    for answer in re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", md, re.S):
        if re.search(r"\d\.\d{3}", answer):
            assert "0e4fcb00" in answer or "2b72c3f7" in answer, answer[:80]


def test_t64_m4_dat9_wording(nb: dict) -> None:
    md = _markdown(nb)
    assert "never saw" not in md
    assert "photographs from VizWiz, which is not in the checkpoint's TextCaps" in md
    assert "pretraining corpus is not public" in md


def test_t64_M1_adapt_keeps_the_frozen_epoch_unless_beaten_by_the_margin(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from pix2struct_textcaps_pipeline.pipeline import DEFAULT_WEIGHTS_DIR

    if not (DEFAULT_WEIGHTS_DIR / "tokenizer_config.json").is_file():
        pytest.skip("the pinned processor files are not staged")
    helpers = _load("adaptation_model_helpers_row64", ROOT / "tests" / "test_adaptation_model.py")
    recs = [
        {
            "id": f"s{i}",
            "image": str(
                helpers._scene(tmp_path / f"s{i}.png", helpers.COLOURS[i % 4], "circle", f"SIGN {i}")
            ),
            "captions": [f"a sign that says sign {i}"],
            "category": "text",
        }
        for i in range(8)
    ]
    for min_gain, expected_epoch, kept_frozen in ((0.02, 0, True), (0.0, 2, False)):
        pipe = helpers._tiny_pipeline()
        before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
        scores = iter([0.200, 0.210, 0.215])
        monkeypatch.setattr(
            pipe,
            "evaluate",
            lambda records, scores=scores, **kw: {
                "bleu4": 0.0,
                "rouge_l": 0.0,
                "cider_d": next(scores),
                "unigram_f1": 0.0,
                "mean_words": 1.0,
                "n": len(records),
            },
        )
        result = pipe.adapt(recs, recs[:2], epochs=2, lr=1e-3, batch_size=4, min_gain=min_gain)
        assert result["best_epoch"] == expected_epoch and result["kept_frozen_by_margin"] is kept_frozen
        changed = any(not (v == before[k]).all() for k, v in pipe._model.state_dict().items())
        assert changed is (not kept_frozen)


def test_rule8_section_8_refuses_the_fresh_test_when_the_model_is_not_the_selected_one(
    nb: dict, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="not the selected configuration"):
        _section8(nb, tmp_path, _settings(), state="something else")
    assert not (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").exists()


def test_round2_paired_bootstrap_interval_reads_the_sign() -> None:
    up = metrics.paired_bootstrap_ci([0.1, 0.2, 0.15, 0.05, 0.12] * 10)
    assert up["verdict"] == "adapted above frozen" and 0 < up["low"] <= up["mean"] <= up["high"]
    flat = metrics.paired_bootstrap_ci([0.3, -0.3, 0.2, -0.2, 0.0] * 10)
    assert flat["verdict"] == "no difference detected" and flat["low"] < 0 < flat["high"]
    assert metrics.paired_bootstrap_ci([0.3, -0.3, 0.2] * 5) == metrics.paired_bootstrap_ci(
        [0.3, -0.3, 0.2] * 5
    )
    assert metrics.paired_bootstrap_ci([-0.1] * 20)["verdict"] == "adapted below frozen"


def test_round2_section_8_reports_both_fresh_sets_with_intervals(nb: dict, tmp_path: Path) -> None:
    ns, calls = _section8(nb, tmp_path, _settings())
    assert (
        calls.count(("frozen", 5)) == 1 and calls.count(("adapted", 5)) == 1
    )  # one caption pass serves both sets
    fresh = ns["evaluation_report_payload"]["fresh_test"]
    for name in ("primary (screened)", "secondary (unscreened captioned pool)"):
        assert set(fresh[name]["cider_d_difference_interval"]) >= {"overall", "text", "no-text"}
        assert fresh[name]["profile"]["n"] == 5
    assert (
        fresh["verdict"] == fresh["primary (screened)"]["cider_d_difference_interval"]["overall"]["verdict"]
    )


def test_round2_no_scored_once_wording_and_one_screen_description(nb: dict) -> None:
    md = _markdown(nb)
    assert not re.search(r"scored once|score[sd]? (the fresh test |a fresh[^.]*?)?once", md, re.I)
    assert "near-duplicate screen" not in md
    assert "41 candidates" in md and "0 confirmed exclusions" in md
    assert "Compare models within one set, never absolute CIDEr-D across sets" in md
    assert "practical tolerance" in md


def test_round2_exploratory_grid_never_deletes_the_preregistered_grid(nb: dict) -> None:
    seven = _cell_with(nb, "RUN_GRID = True  # @param")
    assert "grid_dir = Path(f\"outputs/pix2struct_textcaps_grid{grid_look['suffix']}\")" in seven
    assert seven.index("grid_look = plan_test_look(prereg") < seven.index("shutil.rmtree(grid_dir")
    assert "val_max_new_tokens=CAPTION_MAX_TOKENS" in seven and "'history.json'" in seven
