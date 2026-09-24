# Release verification

`tutorials/pix2struct_textcaps_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  VizWiz-Captions dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Pix2StructTextCapsPipeline.from_pretrained(weights_dir=...)`, `fetch_annotations` and `fetch_images` from the
  pinned cache path, `build_sample_dataset(seed=SPLIT_SEED, image_paths=...)` / `load_byod_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_jsonl`, the ceiling print, `validate_inputs`
  with the tiny-image refusal probe, `pipe.caption` with the sanity checks, `evaluation_report` with `text_recall`
  on the drawn scenes, `constant_caption_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` on the frozen
  model and on the validation and test splits after adaptation, the per-category breakdown through `cider_d` and
  `reference_captions`, `pipe.adapt` with its explicit hyperparameters, `evaluation_report` on the scenes after
  adaptation, `pipe.save_artifact`, `Pix2StructTextCapsPipeline.from_artifact` and the reload-parity assertion,
  the recorded `adapted_beats_frozen` flag, and the provenance fields `weight_format`, `weight_sha256` and the
  `corpus` block), the six expected `outputs/` paths, the learner-facing statements and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `from transformers import` / `Pix2StructForConditionalGeneration` /
  `Pix2StructProcessor` / `.generate(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` /
  `safetensors` / `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline suites (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, annotation and image fetchers,
tiny PIL drawings, temporary manifests, no weights). `tests/test_adaptation_model.py` builds a 3-layer, 32-wide
random Pix2Struct from the committed config, processor and tokenizer and runs the real `evaluate` / `adapt` /
`save_artifact` / `load_artifact` path on it offline; its pinned-checkpoint and CUDA cases skip unless
`model.safetensors` is staged and a GPU is visible. These are source/provenance and unit checks. They are **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; a GPU runtime is recommended for Section 7) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/pix2struct-textcaps-base/` or the corpus cache `weights/vizwiz-captions/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `CAPTION_MAX_TOKENS = 30`, `EPOCHS = 4`, `LEARNING_RATE = 1e-5`,
   `BATCH_SIZE = 8`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; an interpreter restart after the install is expected where the runtime's preinstalled
   torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `google/pix2struct-textcaps-base` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files), and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with no further Hub access (a
     font fetch in the logs after staging is a finding);
   - Section 4: `fetch_annotations` checking the VizWiz-Captions shard's declared size and SHA-256 against the
     pins and matching the pinned text digest; `fetch_images` reading the 336 photographs of row group 0 with every
     size and SHA-256 matching; the seeded split of the 318 captioned photographs into 208 / 40 / 70 with
     `check_split_disjoint` reporting no shared image and the three dataset digests printed;
     `outputs/pix2struct_textcaps_train.jsonl` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings surfaced; the three scenes drawn; `validate_inputs` writing
     `outputs/pix2struct_textcaps_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     tiny-image probe); `pipe.caption` on the three scenes with every sanity check `True` and the per-image
     `evaluation_report` verdict `sample-sanity` with `text_recall` (the inference-only notebook's runs quoted
     `STOP`, `Blue Fern Bakery` and `42` and missed `OPEN` and `LIONS`; a different caption on another runtime is
     a finding to record, not a failure);
   - Section 6: the constant-caption and colour-neighbour baselines and the frozen model's test score with the
     per-category breakdown, and the cell's assertion that the frozen CIDEr-D is above the constant caption's;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 18,879,744 trainable of 282,285,696
     parameters (29 tensors: two decoder blocks and the final layer norm), 208 training photographs and their
     (image, caption) pairs, and the epoch history with validation CIDEr-D;
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on four metrics,
     the per-category breakdown, `adapted_beats_frozen` printed and `outputs/pix2struct_textcaps_evaluation_report.json`
     written (the gain is **recorded, not asserted**, until a measured recipe is on file);
   - Section 9: the three scenes captioned by the adapted model with the `sample-sanity` report,
     `outputs/pix2struct_textcaps_captions.csv` written; `pipe.save_artifact` writing
     `outputs/pix2struct_textcaps_adapter/{adapter.safetensors,manifest.json}` and
     `Pix2StructTextCapsPipeline.from_artifact` reloading it with identical captions (the cell asserts it);
     `outputs/pix2struct_textcaps_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block, the `corpus` block, the inference-contract items, the comparison, the artifact digest, the
     reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark), the value of
   `adapted_beats_frozen` and any warning or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pix2struct_textcaps_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pix2struct_textcaps_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### `E2E` notebook

No execution of the `E2E` notebook is recorded yet. The rows below are the earlier inference-only notebook's runs;
they are history and are **not** evidence for the `E2E` blob.

### Superseded `TASK-INFERENCE` notebook — local pre-flight (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `2591c6272d77` (commit `faeb013`, generated at `6daa835`; `NOTEBOOK_SOURCE.repository_revision` = `6daa835…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (1.13 GB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font or other download in the log after staging, three `caption` calls → `A stop sign is on a pole.`, `A blue sign that says Blue Fern Bakery.`, `A blue shirt with the number 42 on it.` (9/11/13 tokens, none truncated, 2.87–2.99 s), `evaluation_report` `sample-sanity` (`text_recall` 0.75 = 1.0/0.75/0.5; `open` and `lions` missing — the recorded misses reproduced), scene digests `9432a2ba…` / `dd0853ce…` / `b1e4d658…` (Pillow 11.3.0 bundled font), 5 outputs written | 139.3 s | PASS — pre-flight only; not promotion evidence |

### Superseded `TASK-INFERENCE` notebook — manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `2b77eb7` / `f94940dd8eab` | Kaggle CPU (`kurtvalcorza/dimer-nb2-pix2struct-textcaps` v1) | Default sample path | 307.8 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 1133 MB staged |

## Current status

**Candidate.** No execution of the `E2E` notebook is recorded. What exists: static validation
(`tools/validate_release_assets.py`), the generator parity checks (`--check` OK), the offline suites and the
adaptation suite on a small random Pix2Struct. The Kaggle CPU and local runs above were of the earlier
`TASK-INFERENCE` notebook, whose code path (staging, verification, captioning, the scene report) the `E2E` notebook
still carries as Section 5, but they do not carry over to the new blob.

Facts a reviewer should weigh before promotion: the fine-tuning recipe (`LEARNING_RATE = 1e-5`, four epochs, two
blocks) is carried over from the BLIP captioning row and has not been run on this checkpoint, so the notebook
records `adapted_beats_frozen` instead of asserting a gain — restore the assertion once a measured recipe is
recorded here; `google/pix2struct-textcaps-base` was fine-tuned on TextCaps, whose captions quote the text in the
image, while most VizWiz references do not, so the adaptation may trade quoted text for the corpus's style (the
per-category breakdown is where that shows); each image costs a 2048-patch encoder pass (~2.8–3.4 s on the reference
CPU in the inference-only runs), which the encoder cache pays once per image but the frozen and adapted evaluations
pay again, so a GPU runtime is recommended; the 40-photograph validation split makes epoch selection noisy and the
70-photograph test split carries no dispersion estimate; the drawn scenes are three images of evidence about
behaviour outside the corpus, not a measurement.
