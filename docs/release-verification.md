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
| 2026-10-17 (17:10–17:42) | `5940841` (`5940841a0c0b095f2c7949c037437443ca6bf608`, PR #9 head) / blob `7e709222` (`7e7092226aba85eaf72127a7e8971611fc2f2137`; fetched at that commit and blob-verified before the session; `NOTEBOOK_SOURCE.repository_revision` = `ab59e42…`, the commit the generator embedded; `ab59e42..5940841` only regenerates the notebook; embedded `module_sha256` `9026563972dd…`, generator `build_notebook.py/2.5`). Evidence under `docs/execution-evidence/2026-10-17/`: `pix2struct_textcaps_colab_5940841_colab-cli-t4.ipynb` SHA-256 `5cd553c90cba749cc739b140159c6efa196ad11d562c625b8c84677ba40fdac8`, `pix2struct_textcaps_colab_5940841_colab-cli-t4_exec.log` SHA-256 `ccb65c4ec9092c4c76d9b30eb2348de5192436a5357c01fd8032aad9c1ad89a4`, `pix2struct_textcaps_colab_5940841_colab-cli-t4_run_summary.json` SHA-256 `b24c592b2ff817210d22c09566c6e569fb478b1bb1d202dc01574b540753d676` (copied byte for byte) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab **Tesla T4** (Colab account 2; session `suite-pix2struct-5940841-ae8b` started 2026-10-17T17:10:19Z and stopped after the run) via the workspace `colab-cli-serial-test-suite`; isolated uv environment from the 48-package hash lock (Python 3.12.12, 53 s), `cuda:0`; kernel Python 3.13.15 | Default path, form parameters at their defaults (the pre-registered settings; block comparison skipped), 13/13 code cells in order; forms, uploads, BYOD and the optional experiment not exercised | 1,915.2 s (CLI wall, one pass, no restart; grid 327.3 + 329.2 + 329.9 s) | **PASSED — REL13: reproduction of the pre-registered evaluation (hosted Colab T4, this blob).** 0 error outputs; the notebook printed `test_look` `reproduction`, `preregistered` `True`, `settings_differing` `[]` (the carried `PREREGISTERED_EVALUATION_RECORD` names the `e86f270` run). Every metric equals the evaluation of record: grid 1e-5 0.275 (epoch 4), 2e-5 0.276 (epoch 2), 5e-5 0.257 (epoch 4) → 1e-5 kept; fresh test CIDEr-D constant 0.049, neighbour 0.039, frozen 0.266, **adapted 0.301** (Δ **+0.035**, paired bootstrap 95 % CI **+0.0045 to +0.0683**); `text` 0.390 → 0.443, `no-text` 0.065 → 0.070 (no difference detected); exposed development test 0.309 → 0.325; reload parity 8/8. See the reproduction record below |
| 2026-10-17 (15:12–15:46) | `e86f270` (`e86f270d0353a89349b9c445f1ab2f5d9ec9b464`, an earlier PR #9 head) / blob `07064ffc` (`07064ffc7f2b51155604b8b6dd1dbe08ec25e2e3`; fetched at that commit and blob-verified before the session; `NOTEBOOK_SOURCE.repository_revision` = `ff069a5…`, the commit the generator embedded; `ff069a5..e86f270` only regenerates the notebook; embedded `module_sha256` `193349462af0…`, generator `build_notebook.py/2.5`). Evidence under `docs/execution-evidence/2026-10-17/`: `pix2struct_textcaps_colab_e86f270_colab-cli-t4.ipynb` SHA-256 `f8ce1dc8d7a04f3a57dbc8d1a7fd62bdaf634ff634f13996eebf99a1b9d5fc3b`, `pix2struct_textcaps_colab_e86f270_colab-cli-t4_exec.log` SHA-256 `a090c232c9fab70ebb3dbb38758d90ca71109afca0a88ca7cf4332588cfa1737`, `pix2struct_textcaps_colab_e86f270_colab-cli-t4_run_summary.json` SHA-256 `3b5bf21f98fbf941a8bdaac59e12b6f8fcae01db13a730927c4d0ea6de5d154d` (copied byte for byte) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab **Tesla T4** (Colab account 2; session `suite-pix2struct-e86f270-ae8b` started 2026-10-17T15:12:39Z and stopped after the run, 15:46:35Z) via the workspace `colab-cli-serial-test-suite`; isolated uv environment from the 48-package hash lock (Python 3.12.12, 67 s), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, source `local-snapshot`; kernel Python 3.13.15 | Default path, form parameters at their defaults (`RUN_GRID` on, block comparison skipped), 13/13 code cells in order (cells 3–5 are the carried modules and print nothing): stage and verify 8 snapshot files → VizWiz-Captions @ `c4a6d897` (1,550 annotations; 208 / 40 / 70 split; fresh test row group 2: 336 photographs, 324 captioned, 41 screen candidates, 0 confirmed exclusions, 12 uncaptioned not scored) → drawn scenes → baselines and frozen model → pre-registered learning-rate grid → fresh test and exposed development test → adapter export → reload parity | 2,036.4 s (CLI wall, one pass, no restart; grid 348.8 + 349.1 + 349.8 s) | **PASSED — the pre-registered evaluation of record (REL13: hosted, this blob).** 0 error outputs, no stderr; the notebook printed `test_look` `pre-registered evaluation (this runtime)`, `preregistered` `True`, `settings_differing` `[]`. Grid (validation CIDEr-D, 40 photographs, frozen epoch 0 = 0.219): 1e-5 best 0.275 (epoch 4), 2e-5 0.276 (epoch 2), 5e-5 0.257 (epoch 4) → **1e-5 kept** (best challenger 2e-5 +0.0016 < 0.02), epoch 4 kept. Fresh test (324 photographs, 62 % `text`): CIDEr-D constant 0.049, neighbour 0.039, frozen 0.266, **adapted 0.301** (Δ **+0.035**, paired bootstrap 95 % CI **+0.004 to +0.068**, 2,000 resamples, seed 0: adapted above frozen); `text` (201) 0.390 → 0.443 (+0.053, CI +0.007 to +0.105), `no-text` (123) 0.065 → 0.070 (+0.005, CI −0.016 to +0.026, no difference detected); BLEU-4 0.075 → 0.082, ROUGE-L 0.321 → 0.338. Exposed development test (not independent): CIDEr-D 0.309 → 0.325. Reload parity 8/8 identical captions. See the record below |
| 2026-10-08 (03:14–03:28) | `63912b4` / `0e4fcb004ed9` (`NOTEBOOK_SOURCE.repository_revision` = `505d93f…`, the source revision the generator embedded; `505d93f..63912b4` changes only the isolated worker's `google.colab` stubs, which now carry a module spec, plus their regression test; embedded `module_sha256` `c828b40435e4…`, generator `build_notebook.py/2.2`) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the 40-char SHA and Git-blob verified; kernel Python 3.13.15; isolated uv environment from the 48-package hash lock (Python 3.12.12, 57 s), torch 2.14.0+cu130, transformers 4.57.6, device `cuda:0`, float32, source `local-snapshot` | Default path, form parameters at their defaults, 13/13 code cells in order (cells 3–5 are the carried module definitions and print nothing). Staging: 8 manifest files (1,133,312,209 B) from `google/pix2struct-textcaps-base` @ `61bee0d7…`, `verify_snapshot` 8 files. Section 4: VizWiz-Captions @ `c4a6d897` (1,550 annotations, 336 photographs pinned), split 208 / 40 / 70 photographs (test 40 `text` + 30 `no-text`), the four refusal probes rejected. Section 5: three drawn scenes, mean text recall 0.75. Section 6 (n = 70): constant caption CIDEr-D 0.052, colour neighbour 0.042, frozen 0.309 (BLEU-4 0.096, ROUGE-L 0.329; `text` 0.490, `no-text` 0.068). Section 7: 18,879,744 of 282,285,696 parameters trainable, 889 training pairs; validation CIDEr-D 0.219 → 0.237 → 0.261 → 0.264 → 0.275, best epoch 4; 354.9 s. Section 8: adapted CIDEr-D 0.325 (+0.016), BLEU-4 0.099, ROUGE-L 0.364; `text` 0.504, `no-text` 0.086; `adapted_beats_frozen` `True` (recorded, not asserted). Section 9: drawn-scene text recall frozen 0.75 / adapted 0.583; adapter 29 tensors, 75,522,608 B, sha256 `1e979f50d563…`; reload parity 8/8 identical, 8 reloaded captions differ from the frozen ones. Not exercised: BYOD upload, the block-comparison experiment (skipped by default) and the other activities. Byte-exact evidence in `docs/execution-evidence/2026-10-08-63912b4/`: executed notebook sha256 `3fe3915b79cf…`, `exec.log` `1dc24626ad94…`, `run_summary.json` `ad4a701d7d6d…` | 815.9 s (one pass, no restart) | **PASSED** — one pass, no restart, 0 errors; evidence for this blob; status stays Candidate pending review. REL13: earlier-blob history (blob `0e4fcb00`), not evidence for blob `07064ffc`; its 70-photograph test is now the exposed development test |
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

The current blob `69e51578` (code commit `728a24c`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line (previously a differently prefixed line), with no traceback, and stops Run all at that cell; genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `7e7092226aba`, are history for it (REL14).

### Row-64 review fix (2026-10-17): pre-registered learning-rate grid and fresh test

- **What touched the old test split.** The default recipe was carried over from the BLIP captioning row, which chose it
  by comparing configurations on its test split, and that split is these same 70 photographs (test digest
  `89f298822913dc68…` in both rows). This row's own pre-flight (Kaggle, 2026-09-24, blob `2b72c3f7`) scored them at
  three learning rates. They are now the **exposed development test**, reported for continuity only.
- **Pre-registration.** `docs/learning-rate-grid-preregistration.md`, committed alone before any code: learning rate in
  {1e-5, 2e-5, 5e-5}, everything else fixed; keep 1e-5 unless beaten on validation CIDEr-D by at least 0.02; keep the
  frozen model unless beaten by at least 0.02; the grid is on by default (`RUN_GRID = True`; estimated about 35 min on a
  T4 before any run; measured on 2026-10-17: 2,036.4 s CLI wall for the whole default path, grid 348.8 + 349.1 + 349.8 s).
- **Fresh test.** Row group 2 of the pinned shard (footer read 2026-10-17: row groups 336/336/336/336/206), ids
  30303..30638: 336 photographs, 324 with a usable caption (12 have none and cannot be scored). The original screen
  (dHash <= 10 against all 336 row-group-0 photographs) flagged 53 look-alikes, not duplicates; a peer review showed
  this, and **pre-registration Amendment 1** (commit `29043d0`, 2026-10-17, before any fresh-test score) replaced it:
  dHash <= 10 against the 208 training photographs gives **41 candidates**; a recorded review of each candidate beside
  its nearest training photograph (`FRESH_TEST_SCREEN_REVIEW`, one reason per id) confirmed **0** as the same object or
  scene (different objects or blank/low-content frames; the three distance-0 hits are blank frames, not copies). The
  primary (screened) and secondary (unscreened pool) sets are therefore both the **324** photographs; id-list SHA-256
  `02e15128fc50ed47…`. The verdict reads a paired bootstrap 95 % interval of the per-image CIDEr-D
  difference (adapted minus frozen; 2,000 resamples, seed 0).
  Pins: `FRESH_IMAGE_PINS`, `FRESH_TEST_EXCLUDED`, `FRESH_TEST_SIZE`, `FRESH_TEST_IDS_SHA256` in `samples.py`.
- **Score-once rule.** The fresh test is scored only when every pre-registered setting matches; outputs carry `test_look`
  and `settings_differing`; exploratory runs write `*_exploratory_<n>.json`. The fresh test is scored only under the
  pre-registered settings; this section records the evaluation and its reproductions.
- **Recorded pre-registered evaluation (REL13):** the Colab CLI Tesla T4 run of 2026-10-17 15:12–15:46 UTC (commit `e86f270`,
  blob `07064ffc`, Colab account 2; first row of the `E2E` table and the record below). It is **the evaluation of record**;
  any later hosted run with identical pre-registered settings is a **reproduction** and is recorded as such here, whatever
  its in-notebook label says. The carried constant `PREREGISTERED_EVALUATION_RECORD` names this run (set in the requote
  regeneration of 2026-10-18, a new blob), so a later run of that blob with the pre-registered settings prints
  `test_look` `reproduction` in any runtime; the notebook's worked answers quote this run (REL13 labels).
- **ROUGE-L.** Now aggregated as `coco-caption` does (best precision and best recall over the references, taken
  separately); the tokenisation is simplified and is not the PTB tokenizer. ROUGE-L values quoted above for blobs
  `2b72c3f7` and `0e4fcb00` (0.329 / 0.364) use the earlier best-per-reference aggregation.
- REL13: CPU development look (SCORE_ONCE_RULE item 8; validation and exposed development test only, the fresh test was
  masked; code revision not recorded by the fixer; no setting was changed after it). CPU build record (2026-10-17, not clean-runtime evidence, not a hosted run): the frozen model captioned the 70 exposed development-test photographs on CPU (6.52 s per photograph): CIDEr-D 0.309, BLEU-4 0.096, ROUGE-L 0.329 under the old best-per-reference aggregation and 0.338 under the coco-caption aggregation now carried (18 of 70 photographs change). A reduced CPU smoke of the regenerated notebook (pinned checkpoint and data, splits cut to 8 / 4 / 4 after Section 4, EPOCHS = 1, so exploratory) ran all 13 code cells: the original screen reproduced its 53 pins (since amended, see above), the grid kept 1e-5 and the frozen epoch, Section 8 reported test_look 'not scored (exploratory)' and wrote only *_exploratory_1 outputs. The fresh test was masked in every CPU check (validation and the exposed development test only) and was never captioned or scored locally, so there is no development look to disclose (SCORE_ONCE_RULE item 8). In a runtime, a pre-registered scoring is labelled `pre-registered evaluation (this runtime)`; this section names the run of record. Before scoring, Section 8 checks that the in-memory model is the selected one (learning rate, kept epoch, adapter-tensor digest).

### Reproduction of the pre-registered evaluation — Colab CLI Tesla T4, 2026-10-17 (commit `5940841`, blob `7e709222`)

REL13: reproduction of the pre-registered evaluation (hosted Colab T4, this blob). It is not a new evaluation; the
`e86f270` run below remains the evaluation of record.

- **Source identity:** fetched at `5940841a0c0b095f2c7949c037437443ca6bf608`, Git blob `7e7092226aba85eaf72127a7e8971611fc2f2137`
  verified before the session; the executed notebook has the same 37 cells, ids and order as the committed blob (source
  identity OK, `inspect_run.py`); embedded `repository_revision` `ab59e42`, `module_sha256` `9026563972dd…`, generator
  `build_notebook.py/2.5`. The blob differs from `07064ffc` by the requote only (worked answers quote the record, the test
  label, and `PREREGISTERED_EVALUATION_RECORD` set); no pre-registered setting changed.
- **Run:** Colab account 2, session `suite-pix2struct-5940841-ae8b`, started 2026-10-17T17:10:19Z, a different session, VM
  and run directory from the `e86f270` run (`suite-pix2struct-e86f270-ae8b`, 15:12:39Z); 13/13 code cells in order, one
  pass, no restart, 0 error outputs; 1,915.2 s CLI wall; isolated environment built in 53 s.
- **Evaluation status (REL13):** the notebook printed `test_look` `reproduction`, `preregistered` `True` and
  `settings_differing` `[]` for the primary and secondary fresh-test blocks and the report.
- **Match with the evaluation of record:** all 120 printed output lines equal the `e86f270` run's apart from timings,
  the worker process id, the embedded revision and the `test_look` label: grid validation CIDEr-D (1e-5 0.219 → 0.237 →
  0.261 → 0.264 → 0.275; 2e-5 best 0.276 at epoch 2; 5e-5 best 0.257 at epoch 4; 1e-5 and epoch 4 kept); fresh test
  (324 photographs, 0 exclusions, 12 uncaptioned not scored) CIDEr-D constant 0.049, neighbour 0.039, frozen 0.266,
  adapted 0.301, difference +0.035001, paired bootstrap 95 % CI +0.004461 to +0.068316 (2,000 resamples, seed 0);
  `text` (201) 0.390 → 0.443, CI +0.0066 to +0.1047; `no-text` (123) 0.065 → 0.070, CI −0.0162 to +0.0263, no difference
  detected; BLEU-4 0.075 → 0.082, ROUGE-L 0.321 → 0.338; exposed development test CIDEr-D 0.309 → 0.325; drawn-scene
  captions, adapter (29 tensors, 75,522,608 B, sha256 `1e979f50d5630219…`) and reload parity 8/8 identical.
- **Non-metric differences:** environment 53 s against 67 s; grid 327.3 / 329.2 / 329.9 s against 348.8 / 349.1 / 349.8 s;
  frozen exposed-test captioning 53.0 s against 55.7 s; whole run 1,915.2 s against 2,036.4 s.
- **Evidence:** `docs/execution-evidence/2026-10-17/pix2struct_textcaps_colab_5940841_colab-cli-t4.ipynb` SHA-256 `5cd553c90cba749cc739b140159c6efa196ad11d562c625b8c84677ba40fdac8`,
  `pix2struct_textcaps_colab_5940841_colab-cli-t4_exec.log` SHA-256 `ccb65c4ec9092c4c76d9b30eb2348de5192436a5357c01fd8032aad9c1ad89a4`,
  `pix2struct_textcaps_colab_5940841_colab-cli-t4_run_summary.json` SHA-256 `b24c592b2ff817210d22c09566c6e569fb478b1bb1d202dc01574b540753d676`
  (byte-for-byte copies; `-text` applies). Executor boundary: CLI sequential execution, not a browser Run all; forms,
  uploads, BYOD and the optional experiment were not exercised. Status stays **Candidate**.

### Pre-registered evaluation of record — Colab CLI Tesla T4, 2026-10-17 (commit `e86f270`, blob `07064ffc`)

- **Source identity:** fetched at `e86f270d0353a89349b9c445f1ab2f5d9ec9b464`, Git blob `07064ffc7f2b51155604b8b6dd1dbe08ec25e2e3`
  verified before the session; the executed notebook has the same 37 cells, ids and order as the committed blob (source
  identity OK, `inspect_run.py`); embedded `repository_revision` `ff069a5`, `module_sha256` `193349462af0…`, generator
  `build_notebook.py/2.5`, notebook spec 2.5.
- **Outcome:** 13/13 code cells in order (log cells 1..13), one pass, no restart, 0 error outputs, no stderr; 2,036.4 s CLI
  wall; isolated environment built in 67 s.
- **Evaluation status (REL13):** the notebook printed `test_look` `pre-registered evaluation (this runtime)`, `preregistered`
  `True`, `settings_differing` `[]`, and `selected_model_state_matches` true for learning rate, kept epoch and tensors. It is
  the first hosted run of the pre-registered settings for this blob, so it is **the pre-registered evaluation of record**;
  later identical hosted runs are reproductions.
- **Data:** 1,550 annotations, 336 row-group-0 photographs (318 captioned) split 208 / 40 / 70 (digests `3bec1610…` /
  `f00752da…` / `89f29882…`); fresh test row group 2: 336 photographs, 324 captioned, 12 uncaptioned (not scorable),
  41 screen candidates, **0 confirmed exclusions**, primary = pool = 324 photographs (62 % `text`, 4.33 references
  each), 0 ids shared with row group 0, id-list SHA-256 `02e15128…`.
- **Grid (validation CIDEr-D, 40 photographs; epochs 1–4, epoch 0 = frozen 0.219):** 1e-5 0.237 / 0.261 / 0.264 / **0.275**
  (348.8 s); 2e-5 0.268 / **0.276** / 0.227 / 0.238 (349.1 s); 5e-5 0.232 / 0.203 / 0.250 / **0.257** (349.8 s). Selected
  **1e-5** ("default kept (no challenger beat it by the margin)"; best challenger 2e-5 +0.001579), epoch 4 kept (+0.056
  over frozen). 18,879,744 of 282,285,696 parameters trainable, 889 training pairs.
- **Fresh test (324 photographs), constant / neighbour / frozen / adapted:** CIDEr-D 0.049 / 0.039 / 0.266 / **0.301**;
  BLEU-4 0.040 / 0.023 / 0.075 / 0.082; ROUGE-L 0.321 / 0.234 / 0.321 / 0.338; unigram F1 0.365 / 0.261 / 0.349 / 0.367;
  mean words 11.0 / 11.5 / 11.2 / 13.5. Paired bootstrap of the per-image CIDEr-D difference (adapted − frozen, 2,000
  resamples, seed 0): overall **+0.035, 95 % CI +0.004 to +0.068 → adapted above frozen**; `text` (201) 0.390 → 0.443,
  +0.053 (+0.007 to +0.105); `no-text` (123) 0.065 → 0.070, +0.005 (−0.016 to +0.026, no difference detected). The
  secondary (unscreened pool) result is identical because the screen excluded nothing.
- **Exposed development test (70, not independent):** CIDEr-D 0.052 / 0.042 / 0.309 / 0.325 (+0.016); BLEU-4 0.096 → 0.099;
  ROUGE-L 0.338 → 0.372 (coco-caption aggregation); `text` 0.490 → 0.504, `no-text` 0.068 → 0.086. CIDEr-D, BLEU-4, unigram
  F1 and the 1e-5 validation curve equal the 2026-10-08 run of blob `0e4fcb00`; every ROUGE-L value differs because the
  aggregation changed.
- **Drawn scenes and export:** text recall frozen 0.75 / adapted 0.583; adapter 29 tensors, 75,522,608 B, sha256
  `1e979f50d563…` (equal to earlier runs); reload parity 8/8 identical captions, 8 differ from the frozen ones.
- **Evidence boundary:** CLI sequential execution, not a browser Run all; forms, BYOD upload and the block-comparison
  experiment were not exercised. Status stays **Candidate**.

**Candidate.** On 2026-10-17 the regenerated notebook (blob `07064ffc`, PR #9 head `e86f270`) passed on a fresh Colab Tesla T4
under the Colab CLI (13/13 code cells in order, no restart, 0 errors, 2,036.4 s). That run is the pre-registered evaluation
of record (REL13): learning rate 1e-5 and epoch 4 kept; fresh test CIDEr-D frozen 0.266 → adapted 0.301 (+0.035, paired
95 % CI +0.004 to +0.068). The requoted notebook (blob `7e709222`, PR #9 head `5940841`) then reproduced it on a fresh Colab
Tesla T4 (13/13 code cells, 0 errors, 1,915.2 s; `test_look` `reproduction`; every metric equal; REL13: reproduction of
the pre-registered evaluation). Promotion to Release-grade is a review decision for the exact release revision.

REL13: earlier-blob history. The notebook was regenerated at `63912b4` (blob `0e4fcb004ed9`) and that exact blob passed the Colab CLI Tesla T4
run recorded above (2026-10-08 UTC, 13/13 code cells, 815.9 s, one pass, no restart, 0 errors, default path only), reproducing
the recorded numbers on the 70-photograph split now labelled the exposed development test; promotion to Release-grade is a review decision for the exact release revision. The earlier
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
