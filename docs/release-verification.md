# Release verification

`tutorials/pix2struct_textcaps_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/pix2struct_textcaps_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Pix2StructTextCapsPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `caption`,
  `evaluation_report`), the ceiling print (`MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, `MAX_PATCHES`,
  `MAX_PREFIX_CHARS`, `MAX_NEW_TOKENS`, `DEFAULT_MAX_NEW_TOKENS`, `DECODING`), the exports, the
  learner-facing statements (caller-owned budget and prefix, no score in generated text, the `truncated`
  flag, `text_recall` as sanity check, `not-measurable` without known text, the model captions any image,
  capability exclusions) and the gated-off BYOD default listed in the validator; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `Pix2StructForConditionalGeneration` /
  `Pix2StructProcessor` / `model.generate(` / `from huggingface_hub import` use **outside the carried
  module cell**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebook is standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `max_new_tokens = 30`, `caption_prefix = ''`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `Pix2StructTextCapsPipeline`, `validate_inputs`,
     `evaluation_report`, `text_recall`, `keyword_hits`, `unigram_f1`, `verify_snapshot`,
     `stage_missing_files`) with no import of the repository package;
   - three synthetic cartoon scenes with rendered text drawn in code with their RGB SHA-256 printed and the
     ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_PATCHES` 2048, `MAX_PREFIX_CHARS` 128,
     `MAX_NEW_TOKENS` 64, `DEFAULT_MAX_NEW_TOKENS` 30, `DECODING` greedy) surfaced;
   - pinned `google/pix2struct-textcaps-base` acquisition at the immutable revision through the carried
     module: the inline `MANIFEST` is asserted against the module identity and written to
     `weights/pix2struct-textcaps-base/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports
     all 8 manifest entries on a clean runtime, `verify_snapshot` returns its summary dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads from the verified directory with no further Hub
     access (any download in the logs after staging — a font fetch in particular — is a finding);
   - `validate_inputs` writes `outputs/pix2struct_textcaps_input_manifest.json` (verdict `accepted`, three
     inputs, one recorded rejection finding from the tiny-image probe);
   - `caption` returning one caption per image with `truncated` false on the default budget; record the
     captions (the card-pass CPU smoke produced `A stop sign is on a pole.`, `A blue sign that says Blue
     Fern Bakery.`, `A blue shirt with the number 42 on it.`; a materially different result is a finding
     to record, not a failure by itself, because no metric is asserted — greedy decoding on a different
     device can diverge);
   - `evaluation_report` writes `outputs/pix2struct_textcaps_evaluation_report.json` with verdict
     `sample-sanity`, a `text_recall` entry and three per-image entries with found/missing tokens on the
     synthetic sample (`not-measurable` on BYOD), stated as such;
   - `outputs/pix2struct_textcaps_result.json`, `outputs/pix2struct_textcaps_captions.csv` and
     `outputs/pix2struct_textcaps_annotated.png` written with `NOTEBOOK_SOURCE`, model revision, model
     licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pix2struct_textcaps_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pix2struct_textcaps_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `2591c6272d77` (commit `faeb013`, generated at `6daa835`; `NOTEBOOK_SOURCE.repository_revision` = `6daa835…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (1.13 GB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font or other download in the log after staging, three `caption` calls → `A stop sign is on a pole.`, `A blue sign that says Blue Fern Bakery.`, `A blue shirt with the number 42 on it.` (9/11/13 tokens, none truncated, 2.87–2.99 s), `evaluation_report` `sample-sanity` (`text_recall` 0.75 = 1.0/0.75/0.5; `open` and `lions` missing — the recorded misses reproduced), scene digests `9432a2ba…` / `dd0853ce…` / `b1e4d658…` (Pillow 11.3.0 bundled font), 5 outputs written | 139.3 s | PASS — pre-flight only; not promotion evidence |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — no Colab/Kaggle run yet |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded yet; the run is
**pending**. What exists: static validation (`tools/validate_release_assets.py`), the generator parity
checks (`--check` OK), the offline unit suite, and one **local fresh-kernel execution** of the generated
notebook (table above) that exercised the standalone carrier, the real `hf_hub_download` staging path
into an empty `weights/` directory, verification, captioning, the evaluation report and every export —
which is necessary but not promotion evidence because the workstation is not a supported runtime. The
registry status remains **Candidate** until a reviewer confirms a recorded supported-runtime run against
the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the CUDA
path has not been executed; the tutorial samples are flat cartoons with text in Pillow's bundled font, not
photographs, and the model's two recorded misses on them (`OPEN` and `LIONS` not quoted; a green jersey
called blue) are kept in the scored set on purpose; on blank white images the model invented a bottle
"that says 'the man's'" and on uniform noise a shirt "with the number 10", so an image without text yields
a fluent caption quoting invented text with no signal; a conditional prefix can be continued mid-word or
followed by an immediate end-of-sequence, so the default path is unconditional; and each image costs a
2048-patch encoder pass (~2.8–3.4 s on the reference CPU), so a long image list scales linearly.
