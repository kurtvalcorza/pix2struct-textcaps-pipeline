# Notebook review: `tutorials/pix2struct_textcaps_colab.ipynb`

Notebook Review Framework v1 review against DIMER Notebook Specification 2.2. Review only: nothing in the
repository was changed. Finding prefix: **PST**.

## 1. Scope and evidence

### Review contract

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/pix2struct-textcaps-pipeline` |
| Notebook | `tutorials/pix2struct_textcaps_colab.ipynb` |
| Reviewed revision | `origin/main` = `4026e247af95a21c64a53fc3f531d2cd139a84bc` (notebook blob `2b72c3f7773c241988ceac978ec7757c650cfdda`) |
| Generated from | `metadata.dimer.generated_from.revision` `df6a134d6cf3…`, `tools/build_notebook.py` + `tools/notebook_template.py`. `build_notebook.py --check` reports the notebook up to date |
| Requirements baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main`). The notebook declares **2.0** |
| Profile / mode | `E2E` / `GUIDED` (metadata and opening cell) |
| Intended audience | Basic Python and PIL; what an encoder–decoder's generated tokens are; what BLEU-4, ROUGE-L and CIDEr-D measure (Prerequisites) |
| Supported runtime | Colab or Jupyter, Python 3.12. CPU float32 is stated to work; CUDA used when present and recommended for Section 7 |
| Promised outcomes | Pinned install; stage and digest-verify the 8-file snapshot; digest-pinned VizWiz-Captions sample (336 photographs, 318 captioned, split 208 / 40 / 70 by image) with four refusal probes; inference contract on three drawn scenes with `text_recall`; constant-caption and colour-neighbour baselines plus the frozen model on four metrics and per category; bounded adaptation of the last two decoder blocks with validation-CIDEr-D selection; held-out four-way comparison; scenes re-captioned; safetensors adapter export, fresh reload and 8-caption parity; optional BYOD through the same cells |

### Existing execution evidence

`docs/release-verification.md` records two **PASSED** Kaggle Tesla T4 runs of blob `2b72c3f7773c` — the blob under
review — at `7eb0691` (2026-09-24, 902.2 s) and `dc86ec8` (2026-09-25, 868.0 s). Both say "11/11 code cells ok
(**1 restart after install cell**: the pins replaced the loaded numpy and cuda-bindings)". Held-out test CIDEr-D:
constant 0.052, colour-neighbour 0.042, frozen 0.309, adapted 0.325; `text` 0.490 → 0.504, `no-text` 0.068 → 0.086;
best epoch 4; reload parity 8/8. A GPU pre-flight row also records Sections 7–8 re-run "from the frozen model" at
`LEARNING_RATE` 2e-5 (0.321, best epoch 2) and 5e-5 (0.340, best epoch 4). No Colab run, no CPU run of the E2E blob,
no BYOD run and no run of an optional experiment through the notebook's own cells are recorded. STATUS.md,
README.md and `tutorials/README.md` record **Release-grade**.

### Journeys and evidence basis

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection of all 25 cells | Barriers: PST-M3, PST-m1, PST-m3, PST-m5 |
| Clean default | Documented execution evidence (Kaggle T4, exact blob). Not run here: no local GPU by workspace rule, and the full model is 1.13 GB | Completes only after a manual restart (PST-M1). Comparison is meaningful (frozen 0.309 vs constant 0.052; no ceiling). **Colab not verified; CPU path not verified** |
| Active learning | Source inspection, plus direct execution at **reduced scale**: a random 3-decoder-block, 32-wide Pix2Struct built from the committed config (as `tests/test_adaptation_model.py`), CPU, `adapt` called twice as a rerun of Section 7 does | No rerun instructions. A rerun stacks adaptation silently, epoch 0 is still labelled `frozen model`, and the artifact mismatches the in-memory model on 14 tensors (PST-M2). The learning-rate experiment's stated outcome contradicts the repository's own recorded sweep (PST-m3). Full scale: not verified |
| Reuse and recovery | Direct execution of the BYOD branch's verbatim unzip/load/split/validate logic with a stubbed upload (no model). Artifact reload: documented evidence (8/8) | A flat 60-record zip is accepted (39 / 9 / 12). Missing `captions` and an undecodable image give actionable messages. A cancelled upload and a zip without a records file give a bare `StopIteration`; a zip with folders is rejected with a misleading path; 30- and 45-record datasets inside the stated 8..5,000 contract fail with a message naming no split (PST-m2). The documented BYOD rerun silently treats the adapted model as frozen (PST-M2). BYOD with the real model: **not verified** |

Limitations: no learner observation; no Colab or CPU execution of the full notebook; full-scale numbers come from
the repository record and were not reproduced. Repository checks run here are source checks, not execution
evidence (REL8): `build_notebook.py --check` exit 0, `validate_release_assets.py` PASS, `pytest -q -o addopts= tests`
53 passed / 7 skipped (exit 0; the skips are the real-checkpoint and CUDA cases). Probe environment: Python with
torch 2.13.0+cpu, transformers 4.57.6, Pillow 12.3.0 (not the notebook's pins), `CUDA_VISIBLE_DEVICES=-1`.

### Promise → evidence trace

| Claim | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| Run all completes in a fresh runtime (opening cell) | Cell 3 pinned in-kernel install plus stale-module guard | Kaggle T4: 1 restart after the install cell, both runs | Contradicted (PST-M1) |
| Pinned, digest-verified snapshot | Cell 11 | Documented: 8 entries staged and verified | Delivered |
| Digest-pinned VizWiz sample, split by image, 4 refusals | Cell 13 | Documented: 208 / 40 / 70, disjoint; refusals raise | Delivered |
| Inference contract on drawn scenes, no score | Cell 15 | Documented: `sample-sanity`, `text_recall` reported | Delivered; well explained |
| Baselines and frozen model, four metrics, per category | Cell 17 | Documented: 0.052 / 0.042 / 0.309; `text` vs `no-text` | Delivered |
| Bounded adaptation, validation selection | Cell 19 | Documented: best epoch 4, 18,879,744 trainable | Delivered on the default path; breaks on rerun (PST-M2) |
| Held-out comparison, gain recorded not asserted | Cell 21 | Documented: +0.016 CIDEr-D, `adapted_beats_frozen` true | Delivered; small delta with no interval (PST-S3) |
| Scenes re-captioned, adapter export, fresh reload, parity | Cell 23 | Documented: 8/8 identical captions | Delivered; parity's discriminating power not shown (PST-S6) |
| "Optional experiments" (learning rate, 1 block, epochs) | Interpretation cell prose only | No code; rerun stacks (PST-M2); LR prediction contradicts record (PST-m3) | Not delivered as an activity |
| BYOD after the default completes, "re-run from that cell" | Cell 13 `USE_BYOD` branch | Flat zip reaches the split (probe); then Sections 5–7 score the adapted model as frozen | Partly delivered (PST-M2, PST-m2) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Install, stage and verify identities | Printed dicts, Sections 1 and 3 | Shown; no question asks the learner to read them (PST-M3) |
| Read captions/photographs, validate, split without leakage | Section 4 manifests and refusal probes | Shown; no activity (PST-M3) |
| Read `caption`, `new_tokens`, `truncated`, `text_recall` correctly | Section 5 prints and prose | Explained well; no check-your-reading prompt |
| Score frozen model beside baselines, read per category | Section 6 | Exercised on the default path; explained |
| Run bounded fine-tuning with explicit hyperparameters | Section 7 form fields | Exercised once; changing a value and rerunning gives invalid results (PST-M2) |
| Evaluate on an image-disjoint test split | Section 8 | Exercised; reading order explained |
| Export and reload with verified parity | Section 9 | Exercised (documented 8/8) |

## 2. Separate judgments

- **Technical correctness:** the default path is carefully engineered: immutable revisions for model and corpus,
  per-file and per-photograph digests, a column-only range read of one parquet shard, `trust_remote_code=False`, a
  non-VQA processor (no font fetch), a transactional `adapt`, and manifest, digest and exact-tensor-set checks before
  an adapter loads. Two defects: the documented install restart (a `MUST` failure), and `adapt` training the shared
  `pipe` in place from its current weights, so every documented rerun (BYOD, optional experiments) silently corrupts
  the frozen-vs-adapted comparison and can make the exported artifact misdescribe the in-memory model.
- **Promise fulfilment:** identity, corpus, split, inference contract, baselines, adaptation, comparison, export and
  reload are met by documented evidence on the default path. "Run all completes" is not met. Unlike the sibling
  `pix2struct-docvqa` row, the corpus leaves headroom (frozen 0.309, constant 0.052), so the central demonstration is
  valid. The optional experiments and the BYOD rerun are not completable as written.
- **Learner experience:** the prose is unusually clear about what a caption is not, what the metrics mean, how to read
  the comparison and what the result does not establish. But the GUIDED learner meets about 105,000 characters of
  unlabelled carried code before any model step; there is no How-to-use section, roadmap, glossary, troubleshooting,
  prediction, checkpoint, coded activity or conclusion scaffold; template braces leak into two markdown cells; and
  the one predicted experiment outcome contradicts the repository's own measurements.
- **Spec conformance:** fails the `MUST`s RUN1, RUN10 and ENV6 (restart) and DAT13 on the documented BYOD rerun
  (same semantics: the "frozen" baseline is not frozen); DAT19 for the cancelled-upload, missing-records-file,
  nested-path and small-split BYOD cases; SRC3 for the knowingly wrong id pattern in the data contract. Declares spec
  2.0, not 2.2. GDL1–GDL14, EXE2 and EXE5 `SHOULD` deviations are not recorded as deviations. Other applicable
  `MUST`s checked by source inspection appear met on the default path: ST, MOD, DAT1–DAT9, VAL, SPL, FT, EVAL1–EVAL7,
  ART and VER1–VER3.

## 3. Findings

### Major

**PST-M1 — Section 1 install cell: the default `Run all` requires a manual restart.**
Cell 3 `pip install`s exact pins (`torch==2.14.0`, `numpy==2.5.3`, `transformers==4.57.6`, …) into the running
kernel and raises "Restart the runtime, then rerun from the top" if a pinned distribution was already imported. Both
recorded clean runs of this blob hit it ("1 restart after install cell: the pins replaced the loaded numpy and
cuda-bindings"). `docs/release-verification.md` step 4 calls the restart "expected", and the repository records
Release-grade.
- *Consequence:* a learner choosing Run all on a fresh hosted runtime stops at cell 3; the Release-grade label
  overstates the evidence for a one-pass Run all.
- *Evidence:* documented execution evidence (release-verification.md rows 2026-09-24 and 2026-09-25; STATUS.md);
  source inspection of cell 3 and `tools/build_notebook.py:61–69`. Colab: not verified.
- *Recommended correction:* adopt the fleet's uv isolated-environment pattern: the setup cell bootstraps uv, creates
  an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked
  `requirements.txt` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that
  environment, so the kernel's preloaded NumPy/torch are never replaced. Reference:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (also
  `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`). Implement in
  `tools/build_notebook.py`, regenerate, and correct release-verification.md step 4 so a restart is not part of a pass.
- *Acceptance check:* a fresh Colab (or Kaggle) runtime executes every code cell in order in one kernel with no
  restart and no error output, recorded against the new blob; release-verification.md no longer accepts a restart.
- *Spec:* RUN1, RUN10, ENV6 (MUST); REL11.

**PST-M2 — Sections 3–9: no rerun instructions; the documented BYOD rerun and every optional experiment silently
stack adaptation onto the already-adapted model.**
`pipe` is built only in cell 11 (Section 3) and never rebuilt or deleted (static probe: `pipe_built_only_in_cell:
[11]`, `del_pipe_anywhere: false`). `pipe.adapt` (carried `pipeline.py`, `adapt`) trains in place from the pipeline's
*current* weights and labels epoch 0 `"frozen model"`.
- The BYOD text says "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and **re-run from that
  cell**". Doing so runs Section 5 ("frozen" scene captions), Section 6 (`frozen_test`, the "frozen model" baselines
  and per-category row) and Section 7 (epoch 0 "frozen model") on the VizWiz-adapted model, then trains it further.
  Section 8's `delta_vs_frozen` and `adapted_beats_frozen` then compare two adapted models. Nothing errors.
- "Optional experiments": "raise `LEARNING_RATE` …; set `TRAINABLE_DECODER_LAYERS = 1` and compare …; raise `EPOCHS`"
  have no code and no rerun route. Editing Section 7 and rerunning it continues from the adapted weights; with one
  block, the first run's block stays modified in memory but is not exported, so the artifact no longer reproduces
  the in-memory model.
- *Consequence:* the learner's frozen-vs-adapted, learning-rate and 1-vs-2-block conclusions are wrong without any
  signal, and the exported adapter misdescribes its lineage. Framework dimensions 3, 7 and 8. This matches the
  sibling `pix2struct-ai2d` finding (PSA-M2); here the BYOD route is silent (no `del pipe`), unlike
  `pix2struct-docvqa` (PSD-M3, `NameError`).
- *Evidence:* static probe as above plus `byod_rerun_text`. Direct execution at **reduced scale**
  (`stacked_adaptation`): after `adapt(layers=2)` then `adapt(layers=1)`, run 2 did not start from base
  (`run2_started_from_base: false`), reported epoch 0 as `"frozen model"`, all 14 `decoder.layer.1.*` tensors (the
  first-run-only block) differ from base, and the artifact reloaded onto a fresh base mismatches the in-memory model on
  14 tensors. Rebuilding the pipeline, as Section 3 does, restores base (`rebuild_pipeline_restores_base: true`).
- *Recommended correction:* make every experiment start from the verified base and say exactly what to rerun. Either
  rebuild `pipe` with `Pix2StructTextCapsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of Section 5
  (no download; snapshot already verified), or snapshot the adaptable tensors after Section 3 and restore them before
  Sections 5 and 7 (refusing to run Section 6 when `pipe.adapter is not None`). Change the BYOD text
  (`tools/notebook_template.py:41–46`) and the optional-experiments text (`:539–542`) to "set the value, then run
  from Section N to the end".
- *Acceptance check:* after a completed default run, following the written BYOD instruction gives a Section 6
  `frozen_test` with `adapted: False` and Section 7 epoch-0 validation equal to the frozen score; following the
  written 1-block instruction leaves every non-exported decoder tensor equal to base. A reduced-scale probe like
  `stacked_adaptation` reports 0 mismatched tensors.
- *Spec:* DAT13 (MUST), RUN9, UX7, GDL10, VER3–VER5.

**PST-M3 — Whole notebook: the declared `GUIDED` layer is largely missing, and about 105,000 characters of carried
code are unlabelled.**
Static probe: no "How to use this notebook", roadmap, glossary, troubleshooting, collapsible answers, "what to
notice" guidance or conclusion scaffold (`predict_prompt: true` is a false positive: the only match is the code
function `def predict` in cell 17). 0 `cellView: form` cells, no "Infrastructure" label. The three carried module
cells hold 41,395, 12,207 and 51,630 characters and sit between Section 1 and the first model step. No code cell is a
learner activity; the "Optional experiments" are one prose paragraph.
- *Consequence:* the stated learner can follow the default path but is never asked to apply the objectives (predict
  the frozen-vs-constant gap, read the per-category split, explain a caption error). The infrastructure dominates the
  scroll before any learning content.
- *Evidence:* source inspection; static probe `guided_markers`, `embedded_module_cell_chars`,
  `optional_experiments_have_code: false`.
- *Recommended correction:* in `tools/notebook_template.py`, add a How-to-use cell and roadmap after the opening,
  a short glossary (patch, teacher forcing, CIDEr-D document frequency, medoid), label the three module cells
  Infrastructure with `cellView: form` and a one-line "what to look for", add one Predict → Change → Run → Observe →
  Explain activity (e.g. predict which category gains from adaptation before Section 8) with a collapsible answer,
  checkpoints after Sections 4, 6 and 8, a troubleshooting table (restart, CUDA absent, upload cancelled), and a
  conclusion scaffold.
- *Acceptance check:* a static probe finds How-to-use, roadmap, glossary, troubleshooting, at least one prediction
  prompt with a collapsible answer, Infrastructure labels on every carried module cell, and a conclusion scaffold.
- *Spec:* GDL1–GDL14, UX5, UX8 (SHOULD); spec 2.2 notes existing notebooks are not nonconformant solely for lacking
  them, so this is a learner-impact Major, not a `MUST` failure.

### Minor

**PST-m1 — Opening cell and Prerequisites: template brace escaping leaks `{{…}}` into learner markdown, including a
wrong id pattern.** The rendered text shows `{{id, image, captions}}` twice and the id pattern
`[A-Za-z0-9_.:-]{{1,64}}`; the code's pattern is `{1,64}`. The markdown strings in `tools/notebook_template.py:42`
and `:129` are passed through `.format` with doubled braces that are not substituted.
- *Consequence:* a BYOD user copying the pattern gets a regex that matches `{1,64}` literally; the record shape looks
  like a templating error.
- *Evidence:* static probe `double_brace_in_markdown` (cells 0 and 1, 3 occurrences).
- *Recommended correction:* single braces in those markdown strings (or format them), then regenerate.
- *Acceptance check:* no `{{` or `}}` in any rendered markdown cell.
- *Spec:* SRC3, UX2.

**PST-m2 — Section 4 BYOD: four failure cases are not actionable.**
Probe replay of the verbatim branch: a cancelled upload and a zip with no `records.jsonl`/`records.json` raise a bare
`StopIteration`; a zip whose records reference `images/img0.png` (the natural layout) is flattened by basename and
then rejected with "image file not found: …/byod/images/img0.png", although the file was in the zip; datasets of 30
and 45 records — inside the stated "8..5,000 records" — fail with "6 records; 8..5000 are required" / "7 records; …"
because each split is re-validated with `min_records=8`, and the message names no split. The effective minimum is
about 50 single-caption-per-image records (probe). There is no location field (upload dialog only), and the Section 6
`assert frozen_test['cider_d'] > baseline_constant['cider_d']` gives a bare `AssertionError` if a BYOD corpus defeats
it. Missing `captions` and an undecodable image are reported clearly.
- *Consequence:* a learner with valid-looking data gets an unexplained error and no corrective action.
- *Evidence:* direct execution (`byod_replay`), source inspection of cell 13 and `samples.py`
  `split_dataset`/`validate_dataset`.
- *Recommended correction:* in the template's BYOD branch, check the upload and the records file with messages naming
  the expected file; preserve relative paths (or resolve by basename consistently); validate splits with the split
  name and state the effective minimum in the Prerequisites; add a `BYOD_ZIP_PATH` form field; give the Section 6
  assert a message.
- *Acceptance check:* each probe case ends in a `ValueError` naming the failed condition and the fix; a 60-record
  nested-folder zip is accepted.
- *Spec:* DAT19 (MUST), UX10, EXE1, EXE2.

**PST-m3 — Interpretation and "Optional experiments": the predicted learning-rate outcome contradicts the
repository's own recorded sweep.** The notebook says "raise `LEARNING_RATE` and watch the training loss fall while
the validation CIDEr-D drops and the selector keeps an early epoch", and that a too-high rate "overfits this little
data within a few epochs". The pre-flight row in `docs/release-verification.md` records lr 2e-5 → 0.321 (best epoch
2) and lr 5e-5 → **0.340, best epoch 4** — higher than the default 0.325 with the last epoch selected.
- *Consequence:* a learner trying the natural values sees the opposite of the prediction and may conclude the
  notebook is broken or that higher rates are always better; the notebook gives no range where overfitting appears.
- *Evidence:* documented execution evidence (pre-flight row) against the cell 24 text
  (`tools/notebook_template.py:520`, `:539`).
- *Recommended correction:* state the recorded outcome (5e-5 improved CIDEr-D, mostly on `no-text`, at some cost to
  `text`) and name a rate at which the selector does keep an early epoch, measured, or drop the prediction.
- *Acceptance check:* the stated expected outcome for each optional experiment matches a recorded run.
- *Spec:* GDL10, UX2.

**PST-m4 — Metadata and opening: declares NOTEBOOK_SPEC 2.0 rather than the current 2.2.**
- *Evidence:* `metadata.dimer.notebook_spec: "2.0"`, opening cell, `NOTEBOOK_SOURCE`.
- *Recommended correction:* re-baseline against 2.2 in `tools/build_notebook.py` and record any `SHOULD` deviations.
- *Acceptance check:* metadata, opening cell and `tutorials/README.md` declare 2.2.
- *Spec:* §3.4, §32.

**PST-m5 — Opening and Prerequisites: no runtime estimate, and the CPU path is called workable without one.**
The notebook says "the CPU path works but is slow" and points to `docs/release-verification.md` for timings; no
markdown cell states a duration (static probe). Inferred CPU cost: about 540 greedy caption calls (3 + 140 + 200
validation scorings + 180 + 19) at the recorded 2.8–3.4 s per image, plus 208 encoder passes and four epochs of
~900 decoder steps — on the order of an hour, not verified.
- *Consequence:* a learner on a CPU runtime cannot plan, and may abandon a run that is working.
- *Evidence:* source inspection; static probe `runtime_estimate_in_notebook_markdown: false`; README "~15 min on a
  Kaggle Tesla T4".
- *Recommended correction:* state the recorded T4 wall time (labelled with its environment) and a labelled CPU
  estimate, or recommend GPU as required for Section 7.
- *Acceptance check:* the opening cell states a labelled runtime estimate per supported runtime.
- *Spec:* RUN12, UX12, GDL1.

**PST-m6 — Section 1: `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but undocumented.** Setting it skips the pinned install.
- *Evidence:* static probe (`env_var_…_in_code: true`, `env_var_documented_in_markdown: false`).
- *Recommended correction:* one sentence in the Section 1 markdown.
- *Acceptance check:* the variable is named and explained in markdown.
- *Spec:* EXE5.

### Suggestions

- **PST-S1** — Print the `adapted` flag that `pipe.evaluate` already returns in Sections 6 and 8, so a stale-state run
  (PST-M2) is visible. Spec: none.
- **PST-S2** — Sections 6 and 8 caption the 70 test photographs twice each (`pipe.evaluate`, then `model_caption` for
  the per-category table; static probe `test_photos_captioned_twice`): 140 redundant generate calls, roughly 7 minutes
  on CPU. Return per-record predictions from `evaluate` and reuse them. Spec: RUN12.
- **PST-S3** — The held-out gain is +0.016 CIDEr-D on 70 photographs with one seed. A paired bootstrap interval over
  the test photographs (overall and per category) would give "recorded, not asserted" a number the learner can read.
  Spec: EVAL15 (related).
- **PST-S4** — Provide the three optional experiments as optional code cells (after the PST-M2 fix), each following
  Predict → Change → Run → Observe → Explain. Spec: GDL10, UX5, UX9.
- **PST-S5** — Add DOI/APA references for Lee et al. 2023, Sidorov et al. 2020, Gurari et al. 2020 and Vedantam et
  al. 2015 beside the arXiv links. Spec: UX2, SRC10.
- **PST-S6** — In the Section 9 parity check, also report how many of the 8 reloaded captions differ from the frozen
  captions (`frozen_predictions` already holds them), so parity demonstrably shows the adapter applied rather than the
  base reproduced. Not verified whether the current 8 differ. Spec: VER4, VER5.

## 4. Readiness

**Needs revision.** Three Majors are open (PST-M1 restart, PST-M2 silent stacked adaptation on the documented
reruns, PST-M3 missing GUIDED layer), and the `MUST`s RUN1/RUN10/ENV6, DAT13 and DAT19 are unresolved. Execution
evidence for the default path exists (Kaggle T4, exact blob) but records the restart. Remaining gates after fixes: a
one-pass hosted Run all on the new blob (Colab preferred), a recorded BYOD run, and a rerun of one optional
experiment through the written instructions.

Sibling consistency: PST-M1 = PSA-M1 / PSD-M1; PST-M2 ≈ PSA-M2 / PSD-M3 (silent here); PST-M3 = PSA-M3 / PSD-M4;
PST-m1 = PSA-m1; PST-m4–m6 = PSA-m3–m5. The docvqa ceiling finding (PSD-M2) does **not** recur: frozen test CIDEr-D
0.309 sits well above the 0.052 constant baseline and below any ceiling. New in this row: PST-m3 (experiment
prediction contradicted by the record) and PST-S2 (double captioning).

## 5. Verified versus inferred

- **Verified by direct execution (CPU, reduced scale or no model):** BYOD branch outcomes (8 cases); stacked
  adaptation semantics on a 3-block random Pix2Struct (14 mismatched tensors); static notebook properties; generator
  parity, release validator and the offline test suite (53 passed / 7 skipped).
- **From documented evidence only:** every full-scale number, the restart, reload parity 8/8.
- **Inferred:** the CPU runtime estimate (PST-m5); the claim that the BYOD rerun reproduces the stacking at full
  scale (same code path as the probe, not run with the real checkpoint).
- **Most likely to be wrong:** PST-M3's severity — the guided layer is `SHOULD`, and spec 2.2 says existing notebooks
  are not nonconformant solely for lacking it; given how well the prose explains the results, it could fairly be
  Minor. It is kept Major for consistency with the sibling rows.

Probe bundle: `pix2struct_textcaps_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
