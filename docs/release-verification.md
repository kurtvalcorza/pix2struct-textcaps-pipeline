# Release verification

`tutorials/pix2struct_textcaps_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
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
  `tools/build_notebook.py` output for its recorded revision; the single kernel cell that builds (or reuses, by lock digest) the isolated hash-locked uv environment and routes every later cell to it, with no `pip install` into the kernel and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
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
   (= `pyproject.toml`; they are installed into the isolated environment Section 1 builds, so no interpreter restart is
   expected);
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
     per-category breakdown, and the recorded verdict `frozen_beats_constant` (the frozen CIDEr-D against the constant caption's);
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

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-08 (03:14–03:28) | `63912b4` / `0e4fcb004ed9` (`NOTEBOOK_SOURCE.repository_revision` = `505d93f…`, the source revision the generator embedded; `505d93f..63912b4` changes only the isolated worker's `google.colab` stubs, which now carry a module spec, plus their regression test; embedded `module_sha256` `c828b40435e4…`, generator `build_notebook.py/2.2`) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the 40-char SHA and Git-blob verified; kernel Python 3.13.15; isolated uv environment from the 48-package hash lock (Python 3.12.12, 57 s), torch 2.14.0+cu130, transformers 4.57.6, device `cuda:0`, float32, source `local-snapshot` | Default path, form parameters at their defaults, 13/13 code cells in order (cells 3–5 are the carried module definitions and print nothing). Staging: 8 manifest files (1,133,312,209 B) from `google/pix2struct-textcaps-base` @ `61bee0d7…`, `verify_snapshot` 8 files. Section 4: VizWiz-Captions @ `c4a6d897` (1,550 annotations, 336 photographs pinned), split 208 / 40 / 70 photographs (test 40 `text` + 30 `no-text`), the four refusal probes rejected. Section 5: three drawn scenes, mean text recall 0.75. Section 6 (n = 70): constant caption CIDEr-D 0.052, colour neighbour 0.042, frozen 0.309 (BLEU-4 0.096, ROUGE-L 0.329; `text` 0.490, `no-text` 0.068). Section 7: 18,879,744 of 282,285,696 parameters trainable, 889 training pairs; validation CIDEr-D 0.219 → 0.237 → 0.261 → 0.264 → 0.275, best epoch 4; 354.9 s. Section 8: adapted CIDEr-D 0.325 (+0.016), BLEU-4 0.099, ROUGE-L 0.364; `text` 0.504, `no-text` 0.086; `adapted_beats_frozen` `True` (recorded, not asserted). Section 9: drawn-scene text recall frozen 0.75 / adapted 0.583; adapter 29 tensors, 75,522,608 B, sha256 `1e979f50d563…`; reload parity 8/8 identical, 8 reloaded captions differ from the frozen ones. Not exercised: BYOD upload, the block-comparison experiment (skipped by default) and the other activities. Byte-exact evidence in `docs/execution-evidence/2026-10-08-63912b4/`: executed notebook sha256 `3fe3915b79cf…`, `exec.log` `1dc24626ad94…`, `run_summary.json` `ad4a701d7d6d…` | 815.9 s (one pass, no restart) | **PASSED** — one pass, no restart, 0 errors; evidence for this blob; status stays Candidate pending review |
| 2026-09-25 | `dc86ec8` / `2b72c3f7773c` (same blob; the PR head) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-textcaps` v3; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`, `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` (CUDA 13.0) / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 868.0 s | **PASSED** — re-run at the exact PR head: 11/11 code cells ok (1 restart after install cell: the pins replaced the loaded numpy and cuda-bindings); 355 files, 1218 MB staged from the Hub into a clean cache; held-out numbers identical to the v2 row below: test CIDEr-D constant 0.052 / colour-neighbour 0.042 / frozen 0.309 / adapted 0.325, BLEU-4 0.096 → 0.099, ROUGE-L 0.329 → 0.364; by category `text` (40) 0.490 → 0.504, `no-text` (30) 0.068 → 0.086; validation CIDEr-D by epoch 0.219 / 0.237 / 0.261 / 0.264 / 0.275 (best epoch 4); adaptation 325.5 s; `adapted_beats_frozen` true; reload parity 8/8 identical captions; preserved output SHA-256: `pix2struct_textcaps_result.json` `702c829fc100…`, `pix2struct_textcaps_evaluation_report.json` `1cb9d1972fa8…`, `pix2struct_textcaps_adapter/adapter.safetensors` `1e979f50d563…` (75,522,608 bytes); run summary and executed notebook archived under `.agent/backups/kaggle-batch-2026-09-26/out/dimer-nb2-pix2struct-textcaps/v3/evidence/` in the workspace |
| 2026-09-24 | `7eb0691` / `2b72c3f7773c` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-textcaps` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 902.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 355 files, 1218 MB staged from the Hub into a clean cache; test CIDEr-D constant 0.052 / colour-neighbour 0.042 / frozen 0.309 / adapted 0.325, BLEU-4 0.096 → 0.099, ROUGE-L 0.329 → 0.364; by category `text` (40) 0.490 → 0.504, `no-text` (30) 0.068 → 0.086; validation CIDEr-D by epoch 0.219 / 0.237 / 0.261 / 0.264 / 0.275 (best epoch 4); adaptation 349.6 s; `adapted_beats_frozen` true; reload parity 8/8 identical captions; run summary and executed notebook archived under `.agent/backups/tier-b-pix2struct-2026-09-24/kaggle-out/dimer-nb2-pix2struct-textcaps/v2/evidence/` in the workspace |
| 2026-09-24 | `4d8f3dc` / `2b72c3f7773c` (same blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-sweep-pix2struct-textcaps` v1; pinned runtime as above) — **pre-flight, not the promotion evidence** | full `pytest` suite on the GPU (real-checkpoint and CUDA cases), then the notebook's own code cells with the defaults, then Sections 7–8 re-run from the frozen model at `LEARNING_RATE` 2e-5 and 5e-5 | 2992.6 s | pytest exit 0; defaults reproduced the row above exactly (0.309 → 0.325); lr 2e-5 → 0.321 (`text` 0.503, `no-text` 0.078, best epoch 2); lr 5e-5 → 0.340 (`text` 0.484, `no-text` 0.148, best epoch 4) — the higher rate trades quoted text for the corpus's style, as anticipated; peak CUDA memory 6.7–7.9 GB in adaptation |

The rows below are the earlier inference-only notebook's runs; they are history and are **not** evidence for the `E2E` blob.

### Superseded `TASK-INFERENCE` notebook — local pre-flight (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `2591c6272d77` (commit `faeb013`, generated at `6daa835`; `NOTEBOOK_SOURCE.repository_revision` = `6daa835…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (1.13 GB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font or other download in the log after staging, three `caption` calls → `A stop sign is on a pole.`, `A blue sign that says Blue Fern Bakery.`, `A blue shirt with the number 42 on it.` (9/11/13 tokens, none truncated, 2.87–2.99 s), `evaluation_report` `sample-sanity` (`text_recall` 0.75 = 1.0/0.75/0.5; `open` and `lions` missing — the recorded misses reproduced), scene digests `9432a2ba…` / `dd0853ce…` / `b1e4d658…` (Pillow 11.3.0 bundled font), 5 outputs written | 139.3 s | PASS — pre-flight only; not promotion evidence |

### Superseded `TASK-INFERENCE` notebook — manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `2b77eb7` / `f94940dd8eab` | Kaggle CPU (`kurtvalcorza/dimer-nb2-pix2struct-textcaps` v1) | Default sample path | 307.8 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 1133 MB staged |

## Current status

**Candidate.** The notebook was regenerated at `63912b4` (blob `0e4fcb004ed9`) and that exact blob passed the Colab CLI Tesla T4
run recorded above (2026-10-08 UTC, 13/13 code cells, 815.9 s, one pass, no restart, 0 errors, default path only), reproducing
the recorded held-out numbers; promotion to Release-grade is a review decision for the exact release revision. The earlier
Release-grade record identifies blob `2b72c3f7` (committed at `7eb0691`; re-run at `dc86ec8` on 2026-09-25 with identical held-out
numbers) only and remains history.
Any later change to the carried modules or the notebook yields a new blob and returns the status to Candidate.

Facts a reviewer should weigh before promotion: the fine-tuning recipe (`LEARNING_RATE = 1e-5`, four epochs, two
blocks) was carried over from the BLIP captioning row and is now measured on this checkpoint (pre-flight row above):
every recipe tried beat the frozen model, but by +0.012 to +0.031 CIDEr-D on 70 photographs with one seed, so the
notebook keeps recording `adapted_beats_frozen` instead of asserting a gain; `google/pix2struct-textcaps-base` was fine-tuned on TextCaps, whose captions quote the text in the
image, while most VizWiz references do not, so the adaptation may trade quoted text for the corpus's style (the
per-category breakdown is where that shows); each image costs a 2048-patch encoder pass (~2.8–3.4 s on the reference
CPU in the inference-only runs), which the encoder cache pays once per image but the frozen and adapted evaluations
pay again, so a GPU runtime is recommended; the 40-photograph validation split makes epoch selection noisy and the
70-photograph test split carries no dispersion estimate; the drawn scenes are three images of evidence about
behaviour outside the corpus, not a measurement.
