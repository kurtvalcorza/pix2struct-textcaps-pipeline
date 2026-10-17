# Pre-registration: learning-rate grid and fresh test (pix2struct_textcaps_colab)

Recorded 2026-10-17, before any run of the grid and before any image of the fresh test has been captioned or scored.
This file is committed on its own, ahead of the code that implements it, so the rules below cannot have been
chosen after seeing a result. The notebook computes `preregistered` against every setting listed here.

## Why

The default recipe (`LEARNING_RATE = 1e-5`, `EPOCHS = 4`, `BATCH_SIZE = 8`, two trainable decoder blocks) was carried
over from the BLIP captioning row, which chose it by comparing configurations on its test split. That test split is
the same 70 VizWiz photographs this notebook calls its test split (test digest `89f298822913dc68…` in both rows).
This row's own pre-flight (Kaggle, 2026-09-24, blob `2b72c3f7`) also scored that test split at three learning rates
(1e-5: 0.325, 2e-5: 0.321, 5e-5: 0.340 CIDEr-D). Those 70 photographs are therefore an **exposed development
test**, not a held-out test. The recipe is re-selected here on validation only, and a fresh test is scored under
the settings below.

## Grid (validation only)

- Configurations: `LEARNING_RATE` in {**1e-5** (current default), 2e-5, 5e-5}. Nothing else varies.
- Fixed settings: `EPOCHS = 4` with per-epoch validation scoring, `BATCH_SIZE = 8`, `TRAINABLE_DECODER_LAYERS = 2`,
  adaptation seed 0, `SPLIT_SEED = 42`, the training and validation photographs of row group 0 (208 / 40),
  `CAPTION_MAX_TOKENS = 30`, greedy decoding, AdamW (weight decay 0.01), gradient clipping 1.0, no scheduler.
  Every configuration starts from the pinned frozen weights.
- Score of a configuration: its best validation CIDEr-D over epochs 1..4.
- Selection metric: validation CIDEr-D (the notebook's own CIDEr-D, document frequencies from the validation
  references).
- **Tolerance (keep the default):** 1e-5 is replaced only if another configuration's score beats the 1e-5 score by
  **at least 0.02**. If both 2e-5 and 5e-5 clear the margin, the higher score wins; an exact tie goes to the smaller
  learning rate.
- **Keep frozen unless beaten:** within the chosen configuration, an adapted epoch is kept only if its validation
  CIDEr-D beats epoch 0 (the frozen model) by **at least 0.02**; otherwise epoch 0 is kept, the frozen model is the
  system, and the adapter is still exported and labelled as reproducing the frozen captions.
- Within a configuration, the epoch with the highest validation CIDEr-D is kept (an exact tie keeps the earlier
  epoch), subject to the keep-frozen rule.
- The grid runs on the default path (`RUN_GRID = True`). Estimated T4 cost, from the 2026-10-08 hosted run of blob
  `0e4fcb00` (815.9 s wall, 354.9 s for one adaptation, about 0.83 s per photograph captioned): two more adaptations
  (about 710 s), two model reloads, and the fresh test captioned once by the frozen and once by the adapted model
  (about 540 s) bring the default path to roughly **35 min**, under the 55 min limit. This is an estimate, not a
  measurement.

## Fresh test

- Source: **row group 2** of the shard the notebook already pins, `mm-eval/VizWiz-Captions` at revision
  `c4a6d897836e7885d0095134f92d392e4e770539`, `data/val-00004-of-00005.parquet` (392,245,504 bytes, SHA-256
  `4492465a…`). The parquet footer, read 2026-10-17, holds five row groups of 336, 336, 336, 336 and 206 rows. Row
  group 2 holds 336 photographs, ids 30303..30638, of which **324** carry at least one reference caption. No fleet
  row has fetched this row group (the captioning rows read group 0, the image-text matching row groups 0-1).
- Pinning: the row-group index, each photograph's SHA-256 and size, and the SHA-256 of the kept id list. The captions
  are already covered by the shard's text digest `9799ebb1…`.
- Disjointness: image ids are disjoint from the 336 photographs of row group 0 by construction (asserted).
- **Near-duplicate screen**, applied before the first scoring: a 64-bit difference hash (dHash: greyscale, resized
  to 9 x 8, one bit per horizontally adjacent pair) of every fresh-pool photograph is compared with the dHash of
  every photograph in row group 0 (training, validation and exposed test). A fresh photograph within Hamming
  distance **10** of any row-group-0 photograph is excluded. The excluded ids are pinned in the carried module in
  the same commit as the code, before any fresh photograph is captioned; the notebook recomputes the screen and
  refuses to continue if it disagrees with the pin. The same screen is reported (not acted on) between the exposed
  test and the training and validation photographs.
- The 70 photographs of the current test split are reported as the **exposed development test**.

## Score-once rule

- The fresh test is scored only when every setting above matches (`preregistered` over data source, BYOD off, split
  seed, split sizes, caption budget, epochs, batch size, trainable blocks, adaptation seed, the grid and its default,
  both margins, `RUN_GRID`) and only with the configuration the grid selects.
- Any other run is exploratory: it compares on validation (it may show the exposed development test), never scores
  the fresh test, and writes `*_exploratory_<n>.json` instead of the canonical report.
- A selection-only run would record validation only. The default path both selects and scores, so the fresh test is
  scored after the selection, with the selected configuration.
- The first recorded hosted run with these settings is the pre-registered evaluation; a later identical run is a
  reproduction of it. Machine-readable outputs carry `test_look` and `settings_differing`.
