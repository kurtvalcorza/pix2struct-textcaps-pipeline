"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.5 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-aware captioning workflow: the pinned google/pix2struct-textcaps-base
snapshot is digest-verified and loaded, the captions of a digest-pinned VizWiz-Captions shard are read
column-only and the 336 photographs of its first row group are read the same way, validated and split by
image, three drawn scenes with known text are captioned through the inference contract, the frozen model is
scored on the held-out photographs beside two non-neural baselines, a bounded fine-tuning of the caption
decoder's last blocks runs in the kernel, the held-out split is scored again per category, the adapted model
re-captions the drawn scenes, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_textcaps_pipeline",
    "repo_name": "pix2struct-textcaps-pipeline",
    "stem": "pix2struct_textcaps",
    "notebook_name": "pix2struct_textcaps_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "notebook_kind": "TUTORIAL",
    # DOC17-DOC21: canonical NOTEBOOK_SPEC 2.5 use notice (BYOD sentence kept: this notebook has a BYOD path).
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
        "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
        "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
        "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
        "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
        "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
        "> its use."
    ),
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated hash-locked environment from the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the "
        "pinned `google/pix2struct-textcaps-base` snapshot (a 1.13 GB `model.safetensors`), reads the four text columns of one "
        "digest-pinned VizWiz-Captions shard from the Hugging Face Hub (about 0.5 MB over HTTP range requests, no credential) "
        "and the image column of its first row group (336 photographs, about 84 MB, each refused on any size or SHA-256 "
        "mismatch), cuts the 318 captioned ones by image into 208 / 40 / 70 training, validation and exposed-development-test photographs, "
        "reads the image column of row group 2 the same way (336 more photographs, about 84 MB) as the **fresh test** — its 324 captioned "
        "photographs, minus any near-copy of a training photograph confirmed by a pre-registered, recorded review (none was confirmed), "
        "captions three drawn scenes with known text through the inference contract with an input manifest and a rejection "
        "probe, scores the frozen model on the test photographs with BLEU-4, ROUGE-L, CIDEr-D and unigram F1 beside the "
        "constant-caption and colour-nearest-neighbour baselines, runs a bounded fine-tuning of the caption decoder's last "
        "blocks on the training photographs at three learning rates and keeps one by pre-registered validation-CIDEr-D rules, "
        "scores the fresh test only under the pre-registered settings, with a paired bootstrap interval (and the exposed development test for continuity), re-captions the drawn scenes with the adapted model, exports the adapter as safetensors with a manifest, "
        "and reloads that artifact into a fresh pipeline to verify caption parity. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.5 "
        "§5). A CUDA runtime is used automatically when present; the CPU path works but is slow (Pix2Struct encodes every "
        "image at up to 2,048 patches). In the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`), the default path took 2,036.4 s wall on a T4, "
        "348.8 / 349.1 / 349.8 s of it the three grid fine-tunings; the CPU path is estimated at several hours (an estimate, not a measurement), so choose a GPU runtime."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` and `BYOD_PATH` (a zip already in the runtime; on Colab an empty path opens an upload dialog) in Section 4, then re-run Section 4 and every code cell of Sections 5–9 in order (Sections 6 and 7 reload the frozen pipeline from the verified snapshot when `pipe` was adapted by the VizWiz pass, so the frozen numbers are the base model's on your photographs) to supply one zip "
        "holding a `records.jsonl` (or `records.json`) of `{id, image, captions}` objects — `image` a file name inside the "
        "zip, `captions` one or more reference captions, optional `category` — beside the image files (folders inside the zip are kept, so `image` may be `images/x.jpg`; the records file may sit at the root or inside one folder, and image paths are relative to it). They pass through the "
        "same validation, seeded image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and "
        "reload-parity cells as the VizWiz sample. The expected schema and the ceilings (at least 50 records at one record per image so every split keeps 8, at most 5,000 records, 5,002 zip members and 2 GiB extracted) are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path; a BYOD run is "
        "exploratory, so it has no fresh test and its reports go to `*_exploratory_<n>.json`. A refused zip prints one message naming the zip and "
        "the rule, keeps earlier output files but marks them as not results for your data (`outputs/pix2struct_textcaps_run_invalidated.json`), and stops the "
        "later sections until Section 4 succeeds."
    ),
    "pipeline_class": "Pix2StructTextCapsPipeline",
    "weights_key": "pix2struct-textcaps-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct TextCaps-base — DIMER E2E text-aware captioning fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/tutorials/pix2struct_textcaps_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fpix2struct--textcaps--base-ffcc4d?style=flat",
            "https://huggingface.co/google/pix2struct-textcaps-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/pix2struct",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2210.03347-b31b1b.svg", "https://arxiv.org/abs/2210.03347"),
    ],
    "capability": "text-aware image captioning and bounded supervised fine-tuning of the caption decoder's last blocks on a photograph/reference-captions dataset, using the pinned `google/pix2struct-textcaps-base` weights",
    "intro": (
        "`google/pix2struct-textcaps-base` is the Pix2Struct model [1] — a ViT-style image encoder over "
        "variable-resolution 16×16 patches (up to 2,048 per image) and a 12-layer text decoder that cross-attends to them; "
        "282,285,696 parameters, pretrained by parsing masked web screenshots into simplified HTML and fine-tuned on TextCaps [2], "
        "whose captions must mention the text visible in the image — published under the **Apache-2.0** licence. At "
        "inference it reads any text in the image from pixels and generates a sentence token by token with greedy decoding "
        "under a caller-owned `max_new_tokens` budget. **No score exists**: the caption is generated text with no probability "
        "and no correctness signal, and a fluent caption is **not evidence that it describes the image** or that the text it "
        "quotes is really there.\n\n"
        "What this notebook adds to inference is **adaptation with reference captions**. The dataset is real and out of the "
        "model's distribution: VizWiz-Captions [3] (**CC BY 4.0**) — photographs taken by blind people, "
        "each with up to five crowd-written captions in a style TextCaps does not have (what is being held, what a label says, "
        "that the picture is blurry or badly framed). About half the photographs contain text, which is exactly what this "
        "checkpoint was trained to quote, so the per-category breakdown (`text` / `no-text`) is part of the reading. The "
        "notebook reads only the four text columns of one pinned Hub shard (about 0.5 MB over HTTPS range requests) and the "
        "image column of its **first row group** (336 photographs in one 84 MB range read, each pinned by size and SHA-256 in "
        "the carried module) — the same pinned sample the BLIP captioning pipeline uses, so the two captioners can be read "
        "side by side. **Data overlap (DAT9):** these are photographs from VizWiz, which is not in the checkpoint's TextCaps "
        "fine-tuning data (TextCaps images are Open Images photographs); the web-screenshot pretraining corpus is not public, so "
        "overlap with it cannot be fully ruled out, though rendered web pages make it unlikely. **What touched the test:** the "
        "default recipe was carried over from the BLIP captioning row, which chose it by comparing configurations on these same 70 test "
        "photographs, and this row's own pre-flight (blob `2b72c3f7`, 2026-09-24) also scored them at three learning rates. They are "
        "therefore reported as the **exposed development test**. The learning rate is re-selected here on validation only, by rules "
        "pre-registered in `docs/learning-rate-grid-preregistration.md` before any run, and the headline comparison is made on a **fresh "
        "test** — row group 2 of the same pinned shard, never fetched or scored by any earlier run — scored only under those settings. The honest question is a narrow one: does a bounded adaptation of the caption decoder's last blocks "
        "on 208 photographs move the consensus metric on a fresh, image-disjoint test, on which category, and which of the "
        "four metrics move with it? Three captioning metrics are implemented in pure Python in the carried modules "
        "(**BLEU-4** [4], **ROUGE-L** [5] and **CIDEr-D** [6]: BLEU-4 and CIDEr-D follow the `coco-caption` formulas and ROUGE-L uses its "
        "aggregation — the best precision and the best recall over the references taken separately — with CIDEr-D's document "
        "frequencies taken from the evaluated set; the tokenisation is simplified (lower-case, non-word characters replaced by spaces), "
        "**not the PTB tokenizer** `coco-caption` runs, so absolute values differ from `pycocoevalcap` on the same captions) beside the plumbing check `unigram_f1`, and two **non-neural baselines** — "
        "the corpus-medoid constant caption and a colour nearest neighbour — show where a system with no model sits. Nothing "
        "here is a quality claim about your photographs: it is one seeded split of one corpus.\n\n"
        "**Weight-format note:** the pinned revision ships the model as SafeTensors (`model.safetensors`, digest-pinned in the "
        "manifest); the processor is the non-VQA variant, so no header text is rendered into the image and no font is "
        "downloaded. Section 3 stages and digest-verifies the snapshot before the processor or the model is constructed."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met image captioning, and wants to see how an OCR-free captioner reads the text in a photograph, how captions are scored honestly (BLEU-4, ROUGE-L, CIDEr-D) against baselines that barely look at the image, and how a bounded fine-tuning is measured and exported. The audience is students and practitioners preparing their own captioning data; no prior experience with Pix2Struct or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. A T4 GPU runtime is recommended (CPU works but the fine-tuning is slow).\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | photographs with human reference captions: a digest-pinned VizWiz-Captions sample split by photograph (208 training, 40 validation, 70 exposed development-test photographs) plus a fresh test of 324 captioned photographs from another row group, or your own zip |\n| Model | Pix2Struct TextCaps-base: variable-resolution patches into an image encoder and a 12-layer decoder that generates the caption with greedy decoding; only the last two decoder blocks are trained |\n| Output | a caption per photograph, fresh-test BLEU-4 / ROUGE-L / CIDEr-D overall and per `text` / `no-text` category beside a constant-caption and a colour-neighbour baseline (scored only under the pre-registered settings), the same on the exposed development test, a validation-selected learning rate, and a safetensors adapter that reloads with identical captions |\n\n**How to use this notebook.** Choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`) completed in one pass). Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults are the **pre-registered settings**: changing any of them makes the run exploratory, and an exploratory run never scores the fresh test (it says so when it happens). Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** whose worked answer quotes the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`) — the validation grid, the 70 exposed development-test photographs and the fresh test; `docs/release-verification.md` records it and any reproductions. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the VizWiz sample and a photograph-level split *(evaluation practice: leakage)* → 5 the inference contract on a drawn scene with text → 6 two baselines and the frozen model *(core concept: CIDEr-D and the text / no-text split)* → 7 a validation-only learning-rate grid and bounded fine-tuning of two decoder blocks *(core concepts: what is trained; selecting without the test)* → 8 the fresh test, scored only under the pre-registered settings, and the exposed development test → 9 captions again, export and reload *(engineering)* → conclude."
    )]},
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; read the captions of a digest-pinned corpus without downloading its "
        "shard and its photographs from one pinned row group with per-file digests, validate them and split by image without "
        "leakage; caption through the public API over drawn scenes with known text and read `caption`, `new_tokens`, "
        "`truncated` and `text_recall` correctly (generated text, no score); score the frozen model against several reference "
        "captions per photograph with BLEU-4, ROUGE-L and CIDEr-D beside two non-neural baselines and read the `text` / "
        "`no-text` breakdown; run a bounded fine-tuning with explicit hyperparameters, choose its learning rate and epoch on "
        "validation under pre-registered rules, and say what a test split that was used for selection can and cannot show; "
        "score a fresh test only under the pre-registered settings and read a paired bootstrap interval rather than a point difference; re-caption drawings from a different image family with the adapted model; "
        "and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "OCR as a product (the model quotes text inside a caption; it returns no transcription, boxes or confidences), visual "
        "question answering (a separate checkpoint), dense or region captioning (one sentence per image, no localisation), "
        "captions in languages other than English, batch throughput, sampling, beam search or repetition penalties (the "
        "notebook decodes greedily for reproducibility), SPICE (needs a scene-graph parser), evaluation on the TextCaps or "
        "VizWiz benchmarks proper (one seeded sample from row group 0 and a 324-photograph fresh test from row group 2 are scored here), fine-tuning of the image encoder, "
        "the embeddings or the output projection, training on images that are not the pinned sample or your own uploads, and "
        "any claim that a VizWiz split stands in for your photographs. The repository exposes none of these."
    ),
    "prerequisites": [
        '- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with Pix2Struct or fine-tuning. The metrics, baselines, image-level splits, validation selection and adapters are explained where they are first used and again in the Glossary.',
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU (float32) and uses CUDA automatically when available; a GPU runtime is recommended for the fine-tuning section. Every image is encoded at up to 2,048 patches, so captioning and validation scoring cost seconds per photograph on CPU. The pinned `torch==2.14.0` install and the 1.13 GB checkpoint are the large downloads of the run; the row group of photographs adds about 84 MB.",
        "- **Duration:** in the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`), the default path took **2,036.4 s** wall on a T4 (Colab CLI wall, including the 67 s isolated-environment build), 348.8 / 349.1 / 349.8 s of it the three grid fine-tunings at 1e-5 / 2e-5 / 5e-5. **CPU has not been timed** for the whole path: the recorded CPU inference runs took 2.8–3.4 s per image for the 2,048-patch encoder pass, and the default path now captions about 1,300 images and trains three times four epochs of about 900 decoder steps, so expect **several hours** — an estimate, not a measurement. Choose **Runtime → Change runtime type → T4 GPU** before Section 1.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what BLEU-4, ROUGE-L and CIDEr-D measure (n-gram precision with a brevity penalty, longest-common-subsequence F-measure, TF-IDF-weighted n-gram consensus) and why none is a human judgement; why a confident caption is not a correct one, and why a quoted string is not proof the text is in the image.",
        "- **Data contract:** records are `{id, image, captions}` — an image file (a path relative to the records file; folders inside a BYOD zip are kept) decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px and one or more non-empty reference captions of at most `MAX_CAPTION_CHARS` (500) characters (`MIN_CAPTIONS` = 1; VizWiz supplies up to five); optional `image_id` (defaults to the id) groups records on the same image and optional `category` labels the breakdown (`text` / `no-text` in the sample, from the corpus's text-detected flag). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; every record on the same image lands in the same split so a test image is never trained on; every (image, reference caption) pair is one training sample. BYOD accepts one zip of images plus a `records.jsonl` / `records.json` in that shape. Because each split is validated with the same 8-record floor, the effective minimum is **50 records at one record per image** (the 0.15 / 0.20 validation / test fractions leave 8 and 10); at most 5,002 zip members and 2 GiB extracted; a zip is refused, with the file and the rule named, when it has no records file, more than one, a member path outside the zip's folder, a split under 8 records, or exceeds those ceilings.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every caption checked, but nothing checks that a reference caption is right — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people, documents or homes are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads two parts of one object in the Hub dataset repository `mm-eval/VizWiz-Captions` at the immutable revision `c4a6d897…` (`data/val-00004-of-00005.parquet`, 392,245,504 bytes, SHA-256 `4492465a…`): the declared size and SHA-256 are checked against the pins before any byte is read; the four text columns of all 1,550 rows are fetched over HTTPS range requests through `pyarrow` (only the parquet footer and those column chunks) and refused unless their decoded SHA-256 matches; then the image columns of row groups 0 and 2 only (336 JPEG files each, about 84 MB each) are read the same way, each photograph pinned by size and SHA-256 in the carried module and refused on any mismatch. The corpus is CC BY 4.0 [3].",
    ],
    "cells": [
        {
            "md": (
                "## 4. VizWiz photographs, captions and split\n\n"
                "`fetch_annotations` reads the pinned shard's four text columns (or the cache under `weights/vizwiz-captions/`): "
                "it first checks the byte size and SHA-256 the Hub declares for the file against the pins, then reads only the "
                "parquet footer and those column chunks through `pyarrow` over HTTPS range requests, and refuses the decoded "
                "columns unless their SHA-256 matches. `fetch_images` stages the 336 pinned photographs of row group 0 — each "
                "cached file is re-hashed; anything missing is read from the shard's image column in one range read and refused "
                "on any size or SHA-256 mismatch. `build_sample_dataset` keeps the 318 photographs that carry at least one "
                "reference caption (the Hub re-conversion dropped rejected and pre-canned captions), shuffles them with "
                "`SPLIT_SEED` and cuts them **by image** into 208 / 40 / 70 training, validation and **exposed development test** records "
                "(the 70 were used to choose earlier recipes, see the introduction), each labelled `text` or `no-text` from the corpus's text-detected flag. `validate_dataset` then opens and decodes "
                "every image and checks every record against the contract, `check_split_disjoint` asserts no image is shared, "
                "and the training split is written to `outputs/{stem}_train.jsonl` in the shape BYOD expects. The **fresh test** is staged "
                "the same way from row group 2 (`FRESH_IMAGE_PINS`): `build_fresh_test` takes its 324 photographs with a usable caption "
                "(12 more have none — every caption was rejected or pre-canned upstream — so unrecognisable photographs cannot be scored), "
                "re-runs the pre-registered **candidate screen** (a 64-bit difference hash of each fresh photograph against the 208 "
                "training photographs; within 10 bits makes a candidate, 41 of them) and refuses to continue unless it reproduces the "
                "pinned candidates. A candidate is excluded only if a recorded review of the two photographs confirmed the same object or "
                "scene (`FRESH_TEST_SCREEN_REVIEW`, one reason per id); none was, because a whole-image hash also matches different "
                "objects and blank frames. The primary fresh set (screened) and the secondary set (the unscreened pool) are both pinned "
                "by an id-list SHA-256."
                "Look for: 1,550 annotation rows, 336 photographs, three digests, the fresh test's 324 photographs, 41 candidates and 0 confirmed exclusions, the category mix per split (a little over "
                "half the photographs contain text), captions per image between 1 and 5, and four refusal probes — a duplicate "
                "id, a missing image file, an empty caption list and a dataset too small to split — each rejected before "
                "`torch` does anything."
                '\n\n**Predict before running:** why split by photograph rather than by caption? About how many of the test photographs will have readable text?'
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import shutil\n"
                "import time\n"
                "import zipfile\n"
                "\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "\n"
                "\n"
                "class RunInvalidated(RuntimeError):\n"
                "    \"\"\"This run was stopped on purpose (a refused BYOD input); the message says what to fix. Shown without a traceback.\"\"\"\n"
                "\n"
                "    dimer_stop = True\n"
                "\n"
                "    def _render_traceback_(self):\n"
                "        return [str(self)]\n"
                "\n"
                "\n"
                "def require_valid_run():\n"
                "    \"\"\"Every later section starts here: after a refused BYOD input nothing is scored until Section 4 succeeds.\"\"\"\n"
                "    state = globals().get('RUN_STATE') or {{'valid': False, 'reason': 'Section 4 has not run yet'}}\n"
                "    if not state['valid']:\n"
                "        raise RunInvalidated(f\"This run is invalidated ({{state['reason']}}). Fix the input named above and run Section 4 again; this section did not run.\")\n"
                "\n"
                "\n"
                "def byod_file(path, kind, suffixes=()):\n"
                "    \"\"\"BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog.\"\"\"\n"
                "    if str(path).strip():\n"
                "        source = Path(str(path).strip()).expanduser()\n"
                "        if not source.is_file():\n"
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise FileNotFoundError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                "        name, payload = next(iter(uploaded.items()))\n"
                "        source = Path('work') / Path(name).name\n"
                "        source.parent.mkdir(parents=True, exist_ok=True)\n"
                "        source.write_bytes(payload)\n"
                "    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n"
                "        raise ValueError(f'{{source.name}}: expected a {{kind}} ending in {{\" or \".join(suffixes)}}.')\n"
                "    return source\n"
                "\n"
                "\n"
                "def load_byod_splits():\n"
                "    \"\"\"Acquire, extract, parse and split the BYOD zip; every refusal names the zip (or the upload step) and the rule.\"\"\"\n"
                "    byod_zip = byod_file(BYOD_PATH, 'zip of images plus records.jsonl', ('.zip',))\n"
                "    file_name, payload = byod_zip.name, byod_zip.read_bytes()\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    shutil.rmtree(byod_dir, ignore_errors=True)\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    BYOD_MAX_MEMBERS, BYOD_MAX_EXPANDED_BYTES = MAX_RECORDS + 2, 2 * 1024 ** 3  # one image per record plus the records file; 2 GiB extracted\n"
                "    try:\n"
                "        archive = zipfile.ZipFile(io.BytesIO(payload))\n"
                "    except zipfile.BadZipFile:\n"
                "        raise ValueError(f'{{file_name}}: not a readable zip archive; re-create it with a zip tool.') from None\n"
                "    with archive:\n"
                "        members = [m for m in archive.infolist() if not m.is_dir() and Path(m.filename).name and not Path(m.filename).name.startswith('.') and '__MACOSX' not in m.filename]\n"
                "        expanded = sum(m.file_size for m in members)\n"
                "        if len(members) > BYOD_MAX_MEMBERS or expanded > BYOD_MAX_EXPANDED_BYTES:\n"
                "            raise ValueError(f'{{file_name}}: {{len(members)}} files and {{expanded:,}} bytes when extracted exceed the BYOD ceiling of {{BYOD_MAX_MEMBERS}} files / {{BYOD_MAX_EXPANDED_BYTES:,}} bytes (a dataset holds at most MAX_RECORDS = {{MAX_RECORDS}} records); pack fewer or smaller images.')\n"
                "        for member in members:  # folders inside the zip are kept, so records may name images as images/x.jpg\n"
                "            relative = Path(member.filename.replace(chr(92), '/'))\n"
                "            if relative.is_absolute() or '..' in relative.parts:\n"
                "                raise ValueError(f'{{file_name}}: member {{member.filename!r}} points outside the zip (absolute path or ..); repack it with relative paths.')\n"
                "            target = byod_dir / relative\n"
                "            target.parent.mkdir(parents=True, exist_ok=True)\n"
                "            target.write_bytes(archive.read(member))\n"
                "    records_files = sorted(p for p in byod_dir.rglob('*') if p.is_file() and p.name in ('records.jsonl', 'records.json'))\n"
                "    if len(records_files) != 1:\n"
                "        found = [p.relative_to(byod_dir).as_posix() for p in records_files]\n"
                "        raise ValueError(f'{{file_name}}: the zip must hold exactly one records.jsonl (or records.json) — at its root or inside one folder — next to the images; found {{found or \"none\"}}. Add one JSON object per line.')\n"
                "    records_file = records_files[0]\n"
                "    try:\n"
                "        records = load_byod_dataset(records_file)\n"
                "        splits = split_dataset(records, seed=SPLIT_SEED, base_dir=records_file.parent)  # image paths are relative to the records file\n"
                "    except (TypeError, ValueError, json.JSONDecodeError) as exc:\n"
                "        raise ValueError(f'{{file_name}}: {{exc}}') from None\n"
                "    small = {{name: len(rows) for name, rows in splits.items() if len(rows) < MIN_RECORDS}}\n"
                "    if small:\n"
                "        raise ValueError(f'{{file_name}}: split(s) {{small}} hold fewer than MIN_RECORDS = {{MIN_RECORDS}} records (rows per split {{ {{name: len(rows) for name, rows in splits.items()}} }}, {{len(records)}} records in total). Every split is validated with the same floor; with the 0.15 / 0.20 validation / test fractions the smallest dataset that passes is 50 records at one record per image — add records.')\n"
                "    print({{'byod_zip': file_name, 'members': len(members), 'expanded_bytes': expanded, 'records_file': records_file.relative_to(byod_dir).as_posix(), 'records': len(records)}})\n"
                "    return file_name, records, splits\n"
                "\n"
                "\n"
                "invalid_marker = Path('outputs/{stem}_run_invalidated.json')\n"
                "fresh_records, fresh_pool_records, fresh_report = None, None, None\n"
                "if USE_BYOD:\n"
                "    try:\n"
                "        file_name, records, splits = load_byod_splits()\n"
                "        dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "    except (FileNotFoundError, ValueError, TypeError) as exc:  # a refused input; other errors keep their traceback\n"
                "        refusal = str(exc)\n"
                "        for stale_name in ('splits', 'train_records', 'val_records', 'test_records', 'dataset_manifests', 'disjoint', 'categories', 'fresh_records', 'fresh_report', 'frozen_test', 'adapt_result', 'adapted_test', 'comparison'):\n"
                "            globals().pop(stale_name, None)\n"
                "        kept = sorted(p.name for p in Path('outputs').glob('{stem}_*') if p.name != invalid_marker.name)\n"
                "        RUN_STATE = {{'valid': False, 'reason': 'BYOD input refused'}}\n"
                "        invalid_marker.write_text(json.dumps({{'run_valid': False, 'refusal': refusal, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'earlier_outputs_kept_not_results_for_this_input': kept}}, indent=2), encoding='utf-8')\n"
                "        note = f' Earlier files in outputs/ were kept but are not results for this input (listed in {{invalid_marker}}).' if kept else ''\n"
                "        raise RunInvalidated(f'BYOD refused: {{refusal}}{{note}} Nothing was scored; fix the named file or rule and run this cell again.') from None\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    annotations = fetch_annotations(cache_dir='weights/vizwiz-captions')\n"
                "    image_paths = fetch_images(sorted(IMAGE_PINS), cache_dir='weights/vizwiz-captions')\n"
                "    raw_rows = {{'annotations': len(annotations), 'photographs': len(image_paths), 'captioned': sum(1 for r in annotations if r['id'] in IMAGE_PINS and r['captions'])}}\n"
                "    splits = build_sample_dataset(annotations, seed=SPLIT_SEED, image_paths=image_paths)\n"
                "    data_source = f'{{CORPUS_NAME}} {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    # The fresh test: row group 2 of the same shard. dHash hits against the training photographs are candidates only;\n"
                "    # only near-copies confirmed by the pinned manual review are excluded (pre-registration Amendment 1).\n"
                "    fresh_paths = fetch_images(sorted(FRESH_IMAGE_PINS), cache_dir='weights/vizwiz-captions', fresh=True)\n"
                "    screen_train = build_sample_dataset(annotations, seed=SAMPLE_SEED, image_paths=image_paths)['train']  # the pre-registered training photographs\n"
                "    fresh_records, fresh_pool_records, fresh_report = build_fresh_test(annotations, fresh_paths, {{r['image_id']: image_paths[r['image_id']] for r in screen_train}})\n"
                "    fresh_pool_records = validate_dataset(fresh_pool_records)['records']\n"
                "    fresh_records = [r for r in fresh_pool_records if r['image_id'] not in set(FRESH_TEST_EXCLUDED)]\n"
                "    dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint({{**splits, **({{'fresh_test_pool': fresh_pool_records}} if fresh_pool_records else {{}})}})\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "categories = {{name: manifest['categories'] for name, manifest in dataset_manifests.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "RUN_STATE = {{'valid': True, 'reason': None}}\n"
                "if invalid_marker.exists():\n"
                "    invalid_marker.unlink()\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'text_sha256': CORPUS_FILE['text_sha256'][:16] + '...', 'pinned_photographs': len(IMAGE_PINS)}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    label = 'exposed development test' if name == 'test' and not USE_BYOD else name\n"
                "    print({{label: {{'n': manifest['n_records'], 'unique_images': manifest['unique_images'], 'categories': manifest['categories'], 'captions_per_image': manifest['captions_per_image'], 'caption_words': manifest['caption_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "if fresh_report is not None:\n"
                "    print({{'fresh_test': {{k: fresh_report[k] for k in ('row_group', 'photographs', 'pool', 'candidates', 'kept', 'shared_ids_with_row_group_0')}}, 'uncaptioned_not_scorable': len(fresh_report['uncaptioned']), 'confirmed_near_copies_excluded': fresh_report['excluded'], 'screen': f\"dHash candidates at distance <= {{fresh_report['screen']['max_distance']}} against {{fresh_report['screen']['references']}} training photographs, confirmed by a recorded review\", 'ids_sha256': fresh_report['ids_sha256'][:16] + '...'}})\n"
                "    print({{'profiles (n, text share, mean references)': fresh_report['profiles']}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'category': example['category'], 'captions': example['captions']}}}})\n"
                "\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.jpg'}}, *splits['train'][1:8]],\n"
                "    'empty caption list': [{{**splits['train'][0], 'captions': []}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>The five captions of one photograph describe the same picture, so a caption-level split would put the test photograph in training; `check_split_disjoint` confirms no photograph is shared. In the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`), the 70 exposed development-test photographs split into 40 `text` and 30 `no-text` photographs, and the 324 fresh-test photographs into 201 `text` and 123 `no-text` (62 % `text`).</details>"
            ),
        },
        {
            "md": (
                "## 5. Caption through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: three flat cartoon scenes with "
                "**known text rendered into them** using Pillow's bundled font — a red octagonal `STOP` sign on a pole; a "
                "storefront whose fascia reads `Blue Fern Bakery` with a small `OPEN` card in the window; a green jersey reading "
                "`LIONS` above the number `42` — a different image family from the photographs, and scenes the model will be "
                "asked to caption again after adaptation. `validate_inputs` applies exactly the checks `caption` applies (image "
                "sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, an optional prefix up to `MAX_PREFIX_CHARS`, `max_new_tokens` in "
                "`[1, MAX_NEW_TOKENS]`) and returns an input manifest; a 4-pixel image is validated too and its rejection recorded "
                "as a finding. `caption` returns the decoded text, the checked prefix, `image_size`, `new_tokens`, a `truncated` "
                "flag and the model identity. **No score exists**: the caption is generated text with no probability and no "
                "correctness signal, and greedy decoding is a reproducibility property, not a quality one. As recorded in the "
                "model card, the inference-only smoke captioned these drawings `A stop sign is on a pole.`, `A blue sign that "
                "says Blue Fern Bakery.` and `A blue shirt with the number 42 on it.` (missing `OPEN` and `LIONS`). "
                "`evaluation_report` scores the fraction of the drawn words each caption quotes (`text_recall`) — plumbing "
                "evidence for the OCR-free reading path on text you drew yourself, verdict `sample-sanity`, not a metric; whether "
                "captions are *right* is what Sections 6 and 8 measure on photographs with several human references each. The "
                "image digests depend on the Pillow build's bundled font rendering."
            ),
            "code": (
                "import math\n"
                "import time\n\n"
                "require_valid_run()\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "CAPTION_MAX_TOKENS = 30  # @param {{type:\"integer\"}}\n\n\n"
                "def synthetic_scenes():\n"
                "    \"\"\"Three flat cartoon scenes with text rendered in Pillow's bundled font; returns [(name, image, drawn strings)].\"\"\"\n"
                "    sign = Image.new('RGB', (640, 480), (135, 206, 235))  # sky\n"
                "    d = ImageDraw.Draw(sign)\n"
                "    d.rectangle([0, 340, 640, 480], fill=(60, 179, 75))  # grass\n"
                "    d.rectangle([312, 250, 328, 400], fill=(110, 110, 110))  # pole\n"
                "    cx, cy, r = 320, 160, 110\n"
                "    octagon = [(cx + r * math.cos(math.radians(22.5 + 45 * k)), cy + r * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]\n"
                "    d.polygon(octagon, fill=(200, 30, 30), outline='white', width=6)\n"
                "    d.text((cx, cy), 'STOP', fill='white', font=ImageFont.load_default(size=64), anchor='mm')\n"
                "    shop = Image.new('RGB', (640, 480), (230, 230, 220))\n"
                "    d = ImageDraw.Draw(shop)\n"
                "    d.rectangle([40, 120, 600, 460], fill=(190, 150, 110))  # facade\n"
                "    d.rectangle([40, 120, 600, 200], fill=(40, 70, 140))  # fascia\n"
                "    d.text((320, 160), 'Blue Fern Bakery', fill='white', font=ImageFont.load_default(size=40), anchor='mm')\n"
                "    d.rectangle([260, 260, 380, 460], fill=(90, 60, 30))  # door\n"
                "    d.rectangle([80, 240, 220, 400], fill=(200, 230, 250))  # left window\n"
                "    d.rectangle([420, 240, 560, 400], fill=(200, 230, 250))  # right window\n"
                "    d.rectangle([120, 300, 180, 340], fill='white')  # card\n"
                "    d.text((150, 320), 'OPEN', fill=(200, 30, 30), font=ImageFont.load_default(size=22), anchor='mm')\n"
                "    jersey = Image.new('RGB', (480, 560), (245, 245, 245))\n"
                "    d = ImageDraw.Draw(jersey)\n"
                "    d.polygon([(90, 80), (180, 40), (300, 40), (390, 80), (420, 180), (360, 200), (360, 520), (120, 520), (120, 200), (60, 180)], fill=(30, 120, 60))\n"
                "    d.text((240, 250), 'LIONS', fill='white', font=ImageFont.load_default(size=52), anchor='mm')\n"
                "    d.text((240, 380), '42', fill=(255, 215, 0), font=ImageFont.load_default(size=120), anchor='mm')\n"
                "    return [\n"
                "        ('synthetic_stop_sign_640x480.png', sign, ['STOP']),\n"
                "        ('synthetic_bakery_640x480.png', shop, ['Blue Fern Bakery', 'OPEN']),\n"
                "        ('synthetic_jersey_480x560.png', jersey, ['LIONS', '42']),\n"
                "    ]\n\n\n"
                "scenes = synthetic_scenes()\n"
                "scene_names = [name for name, _, _ in scenes]\n"
                "scene_images = [image for _, image, _ in scenes]\n"
                "drawn_texts = [texts for _, _, texts in scenes]\n"
                "scene_digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in zip(scene_names, scene_images)}}\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_PREFIX_CHARS': MAX_PREFIX_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CAPTIONS': MIN_CAPTIONS, 'MAX_CAPTION_CHARS': MAX_CAPTION_CHARS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(scene_images, max_new_tokens=CAPTION_MAX_TOKENS, names=scene_names)\n"
                "try:\n"
                "    validate_inputs([Image.new('RGB', (4, 4))])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'tiny-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scenes': scene_names, 'rgb_sha256': {{k: v[:16] + '...' for k, v in scene_digests.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "results = []\n"
                "for name, image, texts in scenes:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.caption(image, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    result['image'] = name\n"
                "    results.append({{'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print(f\"{{name}}\\n   caption: {{result['caption']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\\n   drawn: {{texts}}\")\n"
                "checks = {{\n"
                "    'one_result_per_image': len(results) == len(scenes),\n"
                "    'captions_are_text': all(isinstance(r['caption'], str) and r['caption'] for r in results),\n"
                "    'budget_respected': all(r['new_tokens'] <= CAPTION_MAX_TOKENS for r in results),\n"
                "    'setting_echoed': all(r['generation']['max_new_tokens'] == CAPTION_MAX_TOKENS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'caption output failed a sanity check: {{checks}}')\n"
                "frozen_scene = evaluation_report(results, drawn_texts, sample_kind='synthetic')\n"
                "for entry in frozen_scene['per_image']:\n"
                "    print(f\"  text_recall {{entry['text_recall']:.2f}}  {{entry['image']}} (found: {{entry['found']}}, missing: {{entry['missing']}})\")\n"
                "print({{'checks': checks, 'frozen_scene_verdict': frozen_scene['verdict'], 'mean_text_recall': round(frozen_scene['metrics'][0]['value'], 3), 'any_truncated': any(r['truncated'] for r in results)}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the exposed development test\n\n"
                "Three systems frame the adaptation, each read four ways. The **constant-caption baseline** answers every "
                "photograph with the one training caption that scores highest against all other training references — the "
                "corpus medoid, a sentence that is safe everywhere and right nowhere. The **colour-nearest-neighbour "
                "baseline** answers with the first reference caption of the training photograph whose 3×3 mean-colour grid is "
                "closest — a lookup that knows the image through 27 numbers. The **frozen model** captions the 70 exposed "
                "development-test photographs (Section 8 repeats all three systems on the fresh test when the run is pre-registered) with the budget from Section 5 and is scored with the same metrics: **BLEU-4** (corpus-level "
                "clipped n-gram precision with a brevity penalty), **ROUGE-L** (longest-common-subsequence F-measure from the "
                "best precision and the best recall over the references), **CIDEr-D** (TF-IDF-weighted n-gram consensus over all references, the metric captioning "
                "leaderboards are ranked by) and the plumbing check **unigram F1**, all after a simplified tokenisation "
                "(lower-casing, non-word characters replaced by spaces; not the PTB tokenizer). Expect the frozen model above both baselines — it is a trained captioner — and read the per-category "
                "breakdown: a checkpoint trained to quote text should do comparatively better on the `text` photographs "
                "(labels, screens, packaging) than on the `no-text` ones. Whether the frozen model beats the constant caption is recorded as "
                "`frozen_beats_constant` (a verdict, not an assert, so a BYOD run still exports). If `pipe` was adapted by an earlier run of Section 7, "
                "the cell first reloads the frozen pipeline from the verified snapshot (Section 7 does the same)."
                '\n\n**Predict before running:** the checkpoint was trained on TextCaps, not on VizWiz. Will it beat the constant caption by a lot or a little? On which category will it do better?'
            ),
            "code": (
                "require_valid_run()\n"
                "# SWP-F: adapt() trains the last decoder blocks of `pipe` in place. If this pipeline was already adapted (a re-run\n"
                "# after Section 7), start again from the pinned, digest-verified snapshot, so the frozen scores and every new\n"
                "# adaptation begin from the frozen weights they are labelled with.\n"
                "if pipe.adapter is not None:\n"
                "    pipe = Pix2StructTextCapsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    print({{'reloaded_frozen_pipeline': True, 'reason': 'the previous pipeline had been adapted in place'}})\n"
                "\n"
                "TEST_LABEL = 'exposed development test' if not USE_BYOD else 'test'\n"
                "baseline_constant = constant_caption_baseline(train_records, test_records)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, test_records)\n"
                "METRICS = ('bleu4', 'rouge_l', 'cider_d', 'unigram_f1')\n"
                "print({{'split': TEST_LABEL, 'constant_caption_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'n': baseline_constant['n'], 'caption': baseline_constant['baseline']}})\n"
                "print({{'split': TEST_LABEL, 'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "print({{'split': TEST_LABEL, 'frozen_model': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'mean_words': round(frozen_test['mean_words'], 1), 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "if frozen_test['adapted']:\n"
                "    raise RuntimeError('the pipeline scored here carries an adapter; these are not frozen-model numbers')\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "\n"
                "\n"
                "def by_category(predictions, records):\n"
                "    \"\"\"CIDEr-D per category for captions already generated (`predictions`: record id -> caption), with document\n"
                "    frequencies from the whole evaluated set (as in `evaluate`).\"\"\"\n"
                "    scores = cider_d([predictions[r['id']] for r in records], [reference_captions(r) for r in records])\n"
                "    groups = collections.defaultdict(list)\n"
                "    for record, score in zip(records, scores, strict=True):\n"
                "        groups[record['category']].append(score)\n"
                "    return {{category: {{'n': len(values), 'cider_d': round(sum(values) / len(values), 3)}} for category, values in sorted(groups.items())}}\n"
                "\n"
                "\n"
                "def model_caption(pipeline):\n"
                "    def predict(record):\n"
                "        with Image.open(record['image']) as photo:\n"
                "            photo.load()\n"
                "            return pipeline.caption(photo, max_new_tokens=CAPTION_MAX_TOKENS)['caption']\n"
                "    return predict\n"
                "\n"
                "\n"
                "medoid = medoid_caption(train_records)\n"
                "constant_fields = by_category({{r['id']: medoid for r in test_records}}, test_records)\n"
                "frozen_predictions = frozen_test['predictions']\n"
                "frozen_fields = by_category(frozen_predictions, test_records)\n"
                "print({{'split': TEST_LABEL, 'by_category': {{'constant': constant_fields, 'frozen': frozen_fields}}}})\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'frozen': frozen_predictions[record['id']], 'references': reference_captions(record)[:2]}})\n"
                "frozen_beats_constant = frozen_test['cider_d'] > baseline_constant['cider_d']\n"
                "print({{'frozen_beats_constant_caption': frozen_beats_constant, 'split': TEST_LABEL}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>By a lot overall but very unevenly. In the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`), on the 70 exposed development-test photographs, the constant caption scored CIDEr-D 0.052, the colour neighbour 0.042 and the frozen model 0.309 (BLEU-4 0.096; ROUGE-L 0.338); per category the frozen model scored 0.490 on `text` photographs and only 0.068 on `no-text` ones — a captioner trained to quote text does well where there is text to quote.</details>'
            ),
        },
        {
            "md": (
                "## 7. A validation-only learning-rate grid and bounded fine-tuning of the caption decoder's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` blocks of the caption decoder plus the decoder's "
                "final layer norm — two blocks by default, 18,879,744 of 282,285,696 parameters; the image encoder, every "
                "embedding and the untied output projection (a 50,244 × 768 matrix) stay frozen. The frozen encoder's output is "
                "computed **once** per training photograph and reused across epochs, so an epoch costs a decoder pass per "
                "(image, caption) pair — every reference caption is a training sample, about 900 pairs per epoch here. The "
                "target is the tokenised caption with its end-of-sequence token, decoded with teacher forcing from the decoder "
                "start token and scored with the model's own cross-entropy (padding ignored); AdamW at a fixed learning rate, "
                "gradient clipping at 1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen model's validation "
                "metrics; every epoch is scored on the 40 validation photographs. **The grid** (`RUN_GRID`, on by default) trains this "
                "recipe three times from the frozen weights, at learning rates 1e-5 (the default), 2e-5 and 5e-5, and nothing else changes. "
                "The rules were pre-registered before any run (`docs/learning-rate-grid-preregistration.md`): a configuration's score is its "
                "best validation CIDEr-D over epochs 1–4; 1e-5 is replaced only if another rate beats it by **at least 0.02** (otherwise the "
                "smaller default is kept); and within the chosen rate an adapted epoch is kept only if it beats epoch 0, the frozen model, by "
                "at least 0.02 — otherwise the frozen model is the system and the exported adapter reproduces the frozen captions, which is "
                "reported, not hidden. The test photographs play no part in any of this. Forty validation photographs make the choice noisy; "
                "the margins are a practical tolerance — a tie-break toward the default rate and toward the frozen model — not a guard against that noise. The rate is chosen first and the keep-frozen rule applied second, so a challenger that beats the frozen model clearly but the default by less than 0.02 loses to the default (conservative by design). The cell ends by comparing every setting with the pre-registration: any difference (a form "
                "field, BYOD, `RUN_GRID` off, a smaller `BATCH_SIZE` for memory) makes the run **exploratory**, and Section 8 then does not "
                "score the fresh test."
                '\n\n**Predict before running:** will validation CIDEr-D improve every epoch, which epoch will be kept, and will a higher learning rate clear the 0.02 margin?'
            ),
            "code": (
                "import gc\n"
                "\n"
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "RUN_GRID = True  # @param {{type:\"boolean\"}}\n"
                "ADAPT_SEED = 0\n"
                "\n"
                "require_valid_run()\n"
                "\n"
                "\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "        row['val_mean_words'] = round(entry['val']['mean_words'], 1)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n"
                "\n"
                "\n"
                "# Score-once rule: the fresh test is scored only when EVERY pre-registered setting matches. The comparison reads the\n"
                "# module's own constants (optimiser, decoding, model revision, fresh-test digests), so an edited module is exploratory.\n"
                "run_settings = {{\n"
                "    'data_source': PREREGISTERED_DATA_SOURCE if not USE_BYOD else data_source,\n"
                "    'byod': USE_BYOD,\n"
                "    'split_seed': SPLIT_SEED,\n"
                "    'split_sizes': [len(train_records), len(val_records), len(test_records)],\n"
                "    'caption_max_tokens': CAPTION_MAX_TOKENS,\n"
                "    'validation_max_new_tokens': CAPTION_MAX_TOKENS,\n"
                "    'epochs': EPOCHS,\n"
                "    'batch_size': BATCH_SIZE,\n"
                "    'trainable_decoder_layers': TRAINABLE_DECODER_LAYERS,\n"
                "    'adapt_seed': ADAPT_SEED,\n"
                "    'learning_rate': LEARNING_RATE,\n"
                "    'run_grid': RUN_GRID,\n"
                "    'lr_grid': list(LR_GRID),\n"
                "    'lr_margin': LR_MARGIN,\n"
                "    'frozen_margin': FROZEN_MARGIN,\n"
                "    'weight_decay': ADAPT_WEIGHT_DECAY,\n"
                "    'grad_clip': ADAPT_GRAD_CLIP,\n"
                "    'scheduler': ADAPT_SCHEDULER,\n"
                "    'decoding': DECODING,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'fresh_test_ids_sha256': FRESH_TEST_IDS_SHA256,\n"
                "    'fresh_pool_ids_sha256': FRESH_POOL_IDS_SHA256,\n"
                "}}\n"
                "prereg = check_preregistration(run_settings)\n"
                "grid_look = plan_test_look(prereg, 'outputs', '{stem}')\n"
                "\n"
                "# Every configuration starts from the pinned frozen weights: the in-memory `pipe` is released and each grid member\n"
                "# loads a fresh frozen pipeline from the verified snapshot (SWP-F), so nothing stacks on an earlier adaptation.\n"
                "globals().pop(\"pipe\", None)\n"
                "gc.collect()\n"
                "grid_lrs = sorted({{*LR_GRID, LEARNING_RATE}}) if RUN_GRID else [LEARNING_RATE]\n"
                "grid_dir = Path(f\"outputs/{stem}_grid{{grid_look['suffix']}}\")  # an exploratory or repeated pass never deletes the pre-registered grid\n"
                "shutil.rmtree(grid_dir, ignore_errors=True)\n"
                "grid_runs = {{}}\n"
                "for lr in grid_lrs:\n"
                "    print({{'grid_configuration': {{'learning_rate': lr, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'trainable_decoder_layers': TRAINABLE_DECODER_LAYERS}}}})\n"
                "    candidate = Pix2StructTextCapsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    t0 = time.perf_counter()\n"
                "    result = candidate.adapt(train_records, val_records, epochs=EPOCHS, lr=lr, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, seed=ADAPT_SEED, min_gain=FROZEN_MARGIN, val_max_new_tokens=CAPTION_MAX_TOKENS, progress=report)\n"
                "    seconds = round(time.perf_counter() - t0, 1)\n"
                "    member_dir = grid_dir / f'lr_{{lr:g}}'\n"
                "    candidate.save_artifact(member_dir, metadata={{'tutorial': '{stem}', 'grid_learning_rate': lr, 'data_source': data_source}})\n"
                "    (member_dir / 'history.json').write_text(json.dumps(result['history'], indent=1), encoding='utf-8')  # saved as each member completes\n"
                "    grid_runs[lr] = {{'best_adapted_val_cider_d': best_adapted_validation(result['history']), 'kept_epoch': result['best_epoch'], 'kept_frozen_by_margin': result['kept_frozen_by_margin'], 'seconds': seconds, 'state_sha256': candidate.adapter_state_sha256(), 'result': result}}\n"
                "    del candidate\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "if RUN_GRID:\n"
                "    selection = select_learning_rate({{lr: run['best_adapted_val_cider_d'] for lr, run in grid_runs.items()}}, default=LEARNING_RATE)\n"
                "else:\n"
                "    selection = {{'chosen_lr': LEARNING_RATE, 'default_lr': LEARNING_RATE, 'reason': 'RUN_GRID is off: one configuration, no selection (exploratory)'}}\n"
                "chosen_lr = selection['chosen_lr']\n"
                "print({{'grid_validation_cider_d': {{f'{{lr:g}}': {{'best_adapted_epoch_score': round(run['best_adapted_val_cider_d'], 3), 'frozen_epoch_0': round(run['result']['history'][0]['val']['cider_d'], 3), 'kept_epoch': run['kept_epoch'], 'seconds': run['seconds']}} for lr, run in grid_runs.items()}}}})\n"
                "print({{'selection': {{k: v for k, v in selection.items() if k != 'scores'}}}})\n"
                "# Load the chosen configuration's kept epoch from its exported tensors onto a fresh frozen base.\n"
                "pipe = Pix2StructTextCapsPipeline.from_artifact(grid_dir / f'lr_{{chosen_lr:g}}', weights_dir=WEIGHTS_DIR)\n"
                "adapt_result = grid_runs[chosen_lr]['result']\n"
                "adapt_seconds = grid_runs[chosen_lr]['seconds']\n"
                "system = 'frozen model (no adapted epoch beat epoch 0 by FROZEN_MARGIN)' if adapt_result['best_epoch'] == 0 else 'adapted model'\n"
                "print({{'chosen_learning_rate': chosen_lr, 'kept_epoch': adapt_result['best_epoch'], 'system': system, 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_pairs': adapt_result['n_pairs'], 'seconds': adapt_seconds}})\n"
                "if prereg['preregistered']:\n"
                "    print({{'preregistered': True, 'next': 'Section 8 scores the fresh test with the selected configuration'}})\n"
                "else:\n"
                "    print({{'preregistered': False, 'settings_differing': prereg['settings_differing'], 'next': 'exploratory run: Section 8 compares on validation and the exposed development test only; the fresh test is not scored'}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`), validation CIDEr-D for the default rate 1e-5 went 0.219 (epoch 0, frozen) → 0.237 → 0.261 → 0.264 → **0.275**, rising every epoch, so epoch 4 was kept (348.8 s); its gain over the frozen model (+0.056) clears the 0.02 keep-frozen margin. 2e-5 went 0.219 → 0.268 → **0.276** → 0.227 → 0.238 (epoch 2, 349.1 s) and 5e-5 went 0.219 → 0.232 → 0.203 → 0.250 → **0.257** (epoch 4, 349.8 s); neither beat the default by the 0.02 margin (best challenger 2e-5, +0.0016), so 1e-5 was kept. Each run of this cell restarts from the frozen weights.</details>'
            ),
        },
        {
            "md": (
                "## 8. The fresh test, scored only under the pre-registered settings, and the exposed development test\n\n"
                "No test photograph appears in the training or validation splits, and neither test played any part in the grid "
                "or the epoch choice. But the 70 photographs of the exposed development test were used to choose recipes before "
                "(the BLIP captioning row chose this recipe on them, and this row's 2026-09-24 pre-flight scored them at three "
                "learning rates), so their numbers are reported for continuity, not as a held-out estimate. The **fresh test** — 324 "
                "captioned photographs of row group 2 — is scored only when every setting matches the pre-registration and the in-memory model is the selected one, with the configuration "
                "the grid selected: the frozen model (reloaded from the snapshot), the selected system and both baselines, side by side "
                "on all four metrics and per category. `test_look` in every output says which kind of look this was: the first recorded "
                "hosted run with these settings is the **pre-registered evaluation** (the hosted run of 2026-10-17 is that evaluation: `PREREGISTERED_EVALUATION_RECORD` names it and `docs/release-verification.md` records it), so a later identical run, in any runtime, is labelled a **reproduction** of the pre-registered evaluation, and "
                "before scoring, the cell checks that the in-memory model is the selected one (learning rate, kept epoch and a digest of the adapter tensors), "
                "anything else is **not scored (exploratory)** and writes `outputs/{stem}_evaluation_report_exploratory_<n>.json` "
                "instead of the canonical report. The fresh test is scored only under the pre-registered settings; see "
                "`docs/release-verification.md` for the recorded evaluation and its reproductions. Read it in "
                "this order: **CIDEr-D** first (the consensus metric the epoch was selected on, and the one that rewards the "
                "corpus's own vocabulary), then BLEU-4 and ROUGE-L, which can move the other way when the adapted captions get "
                "longer, then the `text` / `no-text` split — a gain that sits only on `no-text` photographs means the decoder "
                "learned the corpus's descriptive style, not to read more labels. `adapted_beats_frozen` records whether the "
                "fresh-test CIDEr-D rose as a point estimate only; the verdict comes from a **paired bootstrap 95 % interval** of the per-image "
                "CIDEr-D difference (adapted minus frozen; 2,000 resamples, seed 0), on the screened set (primary) and on the unscreened pool "
                "(secondary), overall and per category. **Compare models within one set, never absolute CIDEr-D across sets:** each set "
                "computes its own document frequencies and differs in size, `text` share and references per image (printed beside its scores). One seed still gives "
                "no estimate of training variance; "
                "the deltas are evidence that the adaptation contract works, not a benchmark, and a gain on VizWiz says nothing about "
                "your photographs until you measure it there."
                '\n\n**Predict before running:** will the frozen-to-adapted delta on the fresh test have the same sign as on the exposed development test, will its interval exclude 0, and on which category?'
            ),
            "code": (
                "require_valid_run()\n"
                "look = plan_test_look(prereg, 'outputs', '{stem}')\n"
                "print({{'test_look': look['test_look'], 'preregistered': look['preregistered'], 'settings_differing': look['settings_differing']}})\n"
                "\n"
                "# 1. The exposed development test (the 70 photographs earlier recipes were chosen on): reported for continuity.\n"
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "print({{'scored_weights': {{'frozen_test_adapted': frozen_test['adapted'], 'adapted_test_adapted': adapted_test['adapted']}}, 'best_epoch': adapt_result['best_epoch'], 'learning_rate': chosen_lr}})\n"
                "adapted_predictions = adapted_test['predictions']\n"
                "adapted_fields = by_category(adapted_predictions, test_records)\n"
                "\n"
                "\n"
                "def compare(constant, neighbour, frozen, adapted, constant_by, frozen_by, adapted_by):\n"
                "    table = {{metric: {{'constant': round(constant[metric], 3), 'neighbour': round(neighbour[metric], 3), 'frozen': round(frozen[metric], 3), 'adapted': round(adapted[metric], 3)}} for metric in METRICS}}\n"
                "    table['mean_words'] = {{'constant': round(constant['mean_words'], 1), 'neighbour': round(neighbour['mean_words'], 1), 'frozen': round(frozen['mean_words'], 1), 'adapted': round(adapted['mean_words'], 1)}}\n"
                "    table['delta_vs_frozen'] = {{metric: round(adapted[metric] - frozen[metric], 3) for metric in METRICS}}\n"
                "    table['by_category'] = {{category: {{'n': frozen_by[category]['n'], 'constant': constant_by[category]['cider_d'], 'frozen': frozen_by[category]['cider_d'], 'adapted': adapted_by[category]['cider_d']}} for category in frozen_by}}\n"
                "    return table\n"
                "\n"
                "\n"
                "comparison = compare(baseline_constant, baseline_neighbour, frozen_test, adapted_test, constant_fields, frozen_fields, adapted_fields)\n"
                "print({{'split': TEST_LABEL, **set_profile(test_records)}})\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'frozen': frozen_predictions[record['id']], 'adapted': adapted_predictions[record['id']], 'references': reference_captions(record)[:2]}})\n"
                "adapted_beats_frozen_exposed = adapted_test['cider_d'] > frozen_test['cider_d']\n"
                "\n"
                "# 2. The fresh test: scored only under the pre-registered settings, with the configuration the grid selected, and only\n"
                "# when the in-memory model is the selected adapted model (not only the settings).\n"
                "selected_state = {{'learning_rate': pipe.adapter.get('lr') == chosen_lr, 'kept_epoch': pipe.adapter.get('best_epoch') == adapt_result['best_epoch'], 'tensors': pipe.adapter_state_sha256() == grid_runs[chosen_lr]['state_sha256']}}\n"
                "print({{'selected_model_state_matches': selected_state}})\n"
                "if look['fresh_test_scored'] and fresh_records and not all(selected_state.values()):\n"
                "    raise RuntimeError(f'the in-memory model is not the selected configuration ({{selected_state}}); run Section 7 again, then Section 8. The fresh test was not scored.')\n"
                "\n"
                "\n"
                "def fresh_summary(records, frozen_predictions_all, adapted_predictions_all):\n"
                "    \"\"\"Scores, baselines, per-category CIDEr-D and the paired bootstrap interval on one fresh set, from captions that\n"
                "    were generated once for the whole pool (document frequencies from this set's references).\"\"\"\n"
                "    refs = [reference_captions(r) for r in records]\n"
                "    frozen = caption_metrics([frozen_predictions_all[r['id']] for r in records], refs)\n"
                "    adapted = caption_metrics([adapted_predictions_all[r['id']] for r in records], refs)\n"
                "    constant, neighbour = constant_caption_baseline(train_records, records), colour_neighbour_baseline(train_records, records)\n"
                "    frozen_by, adapted_by = by_category(frozen_predictions_all, records), by_category(adapted_predictions_all, records)\n"
                "    table = compare(constant, neighbour, frozen, adapted, by_category({{r['id']: medoid for r in records}}, records), frozen_by, adapted_by)\n"
                "    per_frozen = cider_d([frozen_predictions_all[r['id']] for r in records], refs)\n"
                "    per_adapted = cider_d([adapted_predictions_all[r['id']] for r in records], refs)\n"
                "    diffs = [a - f for a, f in zip(per_adapted, per_frozen, strict=True)]\n"
                "    interval = {{'overall': paired_bootstrap_ci(diffs)}}\n"
                "    for category in sorted({{r['category'] for r in records}}):\n"
                "        interval[category] = paired_bootstrap_ci([d for d, r in zip(diffs, records, strict=True) if r['category'] == category])\n"
                "    return {{'profile': set_profile(records), 'comparison': table, 'cider_d_difference_interval': interval, 'adapted_beats_frozen_point_estimate': adapted['cider_d'] > frozen['cider_d'],\n"
                "            'per_image_cider_d': {{'frozen': dict(zip([r['id'] for r in records], per_frozen, strict=True)), 'adapted': dict(zip([r['id'] for r in records], per_adapted, strict=True))}}}}\n"
                "\n"
                "\n"
                "fresh = None\n"
                "if look['fresh_test_scored'] and fresh_records:\n"
                "    frozen_pipe = Pix2StructTextCapsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)  # the frozen weights, for the frozen row\n"
                "    frozen_pool = frozen_pipe.evaluate(fresh_pool_records, max_new_tokens=CAPTION_MAX_TOKENS)  # one caption pass serves both sets\n"
                "    del frozen_pipe\n"
                "    adapted_pool = pipe.evaluate(fresh_pool_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    fresh = {{\n"
                "        'primary (screened)': fresh_summary(fresh_records, frozen_pool['predictions'], adapted_pool['predictions']),\n"
                "        'secondary (unscreened captioned pool)': fresh_summary(fresh_pool_records, frozen_pool['predictions'], adapted_pool['predictions']),\n"
                "        'excluded': fresh_report['excluded'],\n"
                "        'uncaptioned_not_scored': fresh_report['uncaptioned'],\n"
                "    }}\n"
                "    for name in ('primary (screened)', 'secondary (unscreened captioned pool)'):\n"
                "        print({{'split': 'fresh test, ' + name, 'test_look': look['test_look'], **fresh[name]['profile']}})\n"
                "        for key, row in fresh[name]['comparison'].items():\n"
                "            print({{key: row}})\n"
                "        print({{'cider_d_difference_adapted_minus_frozen_95pct_bootstrap': fresh[name]['cider_d_difference_interval']}})\n"
                "    verdict = fresh['primary (screened)']['cider_d_difference_interval']['overall']['verdict']\n"
                "    print({{'verdict (primary interval)': verdict, 'system': system, 'excluded': len(fresh['excluded']), 'uncaptioned_not_scored': len(fresh['uncaptioned_not_scored'])}})\n"
                "else:\n"
                "    verdict = None\n"
                "    print({{'fresh_test': 'not scored', 'why': 'BYOD run (no fresh test)' if USE_BYOD else f\"exploratory run: {{look['settings_differing']}} differ from the pre-registration\"}})\n"
                "adapted_beats_frozen = fresh['primary (screened)']['adapted_beats_frozen_point_estimate'] if fresh else None\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'test_look': look['test_look'],\n"
                "    'preregistered': look['preregistered'],\n"
                "    'settings_differing': look['settings_differing'],\n"
                "    'selected_model_state_matches': selected_state,\n"
                "    'run_settings': prereg['settings'],\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'categories': categories,\n"
                "    'max_new_tokens': CAPTION_MAX_TOKENS,\n"
                "    'grid': {{'runs': {{f'{{lr:g}}': {{k: v for k, v in run.items() if k != 'result'}} | {{'history': run['result']['history']}} for lr, run in grid_runs.items()}}, 'selection': selection, 'dir': str(grid_dir)}},\n"
                "    'system': system,\n"
                "    'fresh_test': None if fresh is None else {{'definition': fresh_report, 'verdict': verdict, **fresh}},\n"
                "    'exposed_development_test': {{'note': 'the 70 photographs earlier recipes were chosen on (BLIP captioning row, and this row\\'s 2026-09-24 pre-flight at three learning rates); reported for continuity, not a held-out estimate' if not USE_BYOD else 'BYOD test split', 'profile': set_profile(test_records), 'baselines': {{'constant_caption': baseline_constant, 'colour_neighbour': baseline_neighbour}}, 'frozen': frozen_test, 'adapted': adapted_test, 'comparison': comparison, 'adapted_beats_frozen': adapted_beats_frozen_exposed}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "    'adapted_beats_frozen_note': 'point estimate only; the verdict is read from the paired bootstrap interval',\n"
                "    'frozen_beats_constant': frozen_beats_constant,\n"
                "}}\n"
                "report_path = Path(f\"outputs/{stem}_evaluation_report{{look['suffix']}}.json\")\n"
                "report_path.write_text(json.dumps(evaluation_report_payload, indent=2, ensure_ascii=False), encoding='utf-8')\n"
                "print({{'report': str(report_path), 'test_look': look['test_look']}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>On the exposed development test, in the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`) (learning rate 1e-5): small — CIDEr-D 0.309 → 0.325 (+0.016), BLEU-4 0.096 → 0.099, ROUGE-L 0.338 → 0.372; `text` 0.490 → 0.504 and `no-text` 0.068 → 0.086. The 2026-09-24 pre-flight (blob `2b72c3f7`) saw all three rates beat the frozen model on these 70 photographs by only +0.012 to +0.031 — those looks are why this split is no longer a held-out test. On the fresh test of the same run (324 photographs, 0 confirmed near-copy exclusions, 12 uncaptioned photographs not scored), CIDEr-D went frozen 0.266 → adapted 0.301 (+0.035, paired bootstrap 95 % CI [0.004, 0.068]: adapted above frozen); `text` (201) 0.390 → 0.443 and `no-text` (123) 0.065 → 0.070, no difference detected. Compare frozen and adapted within one set, and read the sign of the delta, not the absolute value, against the exposed development test.</details>'
            ),
        },
        {
            "md": (
                "## 9. Re-caption the drawn scenes, export the adapter and reload it\n\n"
                "The three scenes from Section 5 are captioned again by the adapted model and their `text_recall` is repeated — "
                "drawings with known text, a different image family from the photographs it was tuned on, so this is a small "
                "look at whether the adaptation kept or lost the checkpoint's habit of quoting text *outside* its corpus (three "
                "drawings of evidence, not a measurement; a different caption here is a finding to record, not a failure). Both "
                "caption sets are written as CSV.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the caption decoder's last two blocks and its final layer norm, "
                "about 76 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id "
                "and revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `Pix2StructTextCapsPipeline.from_artifact` re-verifies the "
                "base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, "
                "refuses any tensor that is not a caption-decoder tensor, and overlays the tensors onto a freshly loaded base — a "
                "new object from files, not the in-memory model (VER2). The cell asserts identical captions on eight test "
                "exposed-development-test photographs (VER4; a parity check, not a score) and also counts how many of those eight reloaded captions differ from the frozen model's, so parity demonstrably shows the adapter applied rather than the base reproduced (VER5; zero is expected only when the selector kept epoch 0)."
                '\n\n**Predict before running:** will the reloaded adapter caption the eight test photographs exactly as the in-memory model did?'
            ),
            "code": (
                "import csv\n"
                "\n"
                "require_valid_run()\n"
                "adapted_results = []\n"
                "for name, image, texts in scenes:\n"
                "    result = pipe.caption(image, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    result['image'] = name\n"
                "    adapted_results.append(result)\n"
                "adapted_scene = evaluation_report(adapted_results, drawn_texts, sample_kind='synthetic')\n"
                "for before, after, entry in zip(results, adapted_results, adapted_scene['per_image'], strict=True):\n"
                "    print({{'image': before['image'], 'frozen': before['caption'], 'adapted': after['caption'], 'adapted_text_recall': round(entry['text_recall'], 2)}})\n"
                "print({{'mean_text_recall': {{'frozen': round(frozen_scene['metrics'][0]['value'], 3), 'adapted': round(adapted_scene['metrics'][0]['value'], 3)}}}})\n"
                "with open(f\"outputs/{stem}_captions{{look['suffix']}}.csv\", 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'frozen_caption', 'adapted_caption', 'adapted_new_tokens', 'drawn_text'])\n"
                "    for before, after, texts in zip(results, adapted_results, drawn_texts, strict=True):\n"
                "        writer.writerow([before['image'], before['caption'], after['caption'], after['new_tokens'], ' | '.join(texts)])\n"
                "\n"
                "artifact_dir = Path(f\"outputs/{stem}_adapter{{look['suffix']}}\")  # an exploratory pass never overwrites the pre-registered artifact\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source, 'learning_rate': chosen_lr, 'test_look': look['test_look']}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'system': system}})\n"
                "\n"
                "reloaded = Pix2StructTextCapsPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [model_caption(pipe)(r) for r in test_records[:8]]\n"
                "after = [model_caption(reloaded)(r) for r in test_records[:8]]\n"
                "parity = {{'identical_captions': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before), 'reloaded_captions_differing_from_frozen': sum(frozen_predictions[r['id']] != b for r, b in zip(test_records[:8], after, strict=True))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch'], 'note': 'reloaded_captions_differing_from_frozen says whether parity shows the adapter applied (> 0) or only the base reproduced (0, expected when the selector kept epoch 0)'}})\n"
                "assert parity['identical_captions'] == parity['of']\n"
                "\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'SafeTensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'text_columns': list(CORPUS_TEXT_COLUMNS), 'file': CORPUS_FILE, 'pinned_images': len(IMAGE_PINS), 'fresh_test_row_group': FRESH_TEST_ROW_GROUP, 'fresh_test_size': FRESH_TEST_SIZE, 'fresh_test_ids_sha256': FRESH_TEST_IDS_SHA256, 'fresh_pool_size': FRESH_POOL_SIZE, 'fresh_pool_ids_sha256': FRESH_POOL_IDS_SHA256}},\n"
                "    'test_look': look['test_look'],\n"
                "    'preregistered': look['preregistered'],\n"
                "    'settings_differing': look['settings_differing'],\n"
                "    'evaluation_report': str(report_path),\n"
                "    'selection': selection,\n"
                "    'system': system,\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'scenes': {{'names': scene_names, 'sizes': [list(image.size) for image in scene_images], 'rgb_sha256': scene_digests, 'drawn_texts': drawn_texts}}, 'items': [{{k: r[k] for k in ('image', 'caption', 'new_tokens', 'truncated', 'seconds')}} for r in results], 'frozen_report': frozen_scene, 'adapted_items': [{{k: r[k] for k in ('image', 'caption', 'new_tokens', 'truncated')}} for r in adapted_results], 'adapted_report': adapted_scene}},\n"
                "    'fresh_test_comparison': None if fresh is None else {{name: fresh[name]['comparison'] for name in ('primary (screened)', 'secondary (unscreened captioned pool)')}},\n"
                "    'fresh_test_verdict': verdict,\n"
                "    'exposed_development_test_comparison': comparison,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "result_path = Path('outputs/{stem}_result.json') if look['suffix'] == '' else Path(f\"outputs/{stem}_result{{look['suffix']}}.json\")\n"
                "result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding='utf-8')\n"
                "print({{'result': str(result_path)}})\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Yes: the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`) reported 8 of 8 identical captions, with a 75,522,608-byte `adapter.safetensors`, and all 8 reloaded captions differed from the frozen ones, so parity showed the adapter applied. If the grid keeps the frozen model this time, expect 0 differing captions.</details>'
            ),
        },
        {
            "md": (
                "### Optional experiment: one trainable decoder block instead of two (off by default)\n\n"
                "**Predict → change one thing → run → observe → explain.** Set `RUN_BLOCK_COMPARISON = True` and run this cell: it loads a "
                "**fresh frozen pipeline** from the verified snapshot (so nothing stacks on the adapter you just exported), adapts only the "
                "last decoder block with the Section 7 settings and the selected learning rate, scores the 70 exposed development-test photographs overall and per category (it is exploratory, so it never scores the fresh test and writes its own `*_exploratory_<n>.json`), saves "
                "the one-block adapter to its own folder and prints both runs side by side. Predict first: will one block reach the same "
                "held-out CIDEr-D as two, which category will move, and how much smaller will the artifact be (count the tensors)? The "
                "default artifact and `result.json` are untouched; the experiment costs one more adaptation (348.8 s on a T4 for two blocks at 1e-5 "
                "in the pre-registered evaluation of record (REL13: the Colab CLI T4 run of 2026-10-17 of blob `07064ffc`, commit `e86f270`)) plus one evaluation."
            ),
            "code": (
                "RUN_BLOCK_COMPARISON = False  # @param {{type:\"boolean\"}}\n"
                "\n"
                "if not RUN_BLOCK_COMPARISON:\n"
                "    print({{'block_comparison_experiment': 'skipped (set RUN_BLOCK_COMPARISON = True to run it); nothing was trained or written'}})\n"
                "else:\n"
                "    require_valid_run()\n"
                "    # Exploratory by construction (trainable_decoder_layers differs from the pre-registration): validation and the\n"
                "    # exposed development test only; the fresh test is never scored here.\n"
                "    experiment_check = check_preregistration({{**run_settings, 'trainable_decoder_layers': 1}})\n"
                "    experiment_look = plan_test_look(experiment_check, 'outputs', '{stem}')\n"
                "    experiment_pipe = Pix2StructTextCapsPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)  # fresh frozen weights: the experiment never touches `pipe`, `reloaded` or the exported artifact\n"
                "    one_block = experiment_pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=chosen_lr, batch_size=BATCH_SIZE, trainable_decoder_layers=1, seed=ADAPT_SEED, min_gain=FROZEN_MARGIN, progress=report)\n"
                "    one_block_test = experiment_pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    one_block_fields = by_category(one_block_test['predictions'], test_records)\n"
                "    experiment_dir = Path('outputs/{stem}_adapter_experiment_1block')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_pipe.save_artifact(experiment_dir, metadata={{'tutorial': '{stem}', 'experiment': 'one trainable decoder block', 'data_source': data_source}})\n"
                "    experiment_manifest = json.loads((experiment_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "    summary = {{\n"
                "        'test_look': experiment_look['test_look'],\n"
                "        'settings_differing': experiment_look['settings_differing'],\n"
                "        'split': TEST_LABEL,\n"
                "        'one_block': {{'best_epoch': one_block['best_epoch'], 'best_adapted_val_cider_d': round(best_adapted_validation(one_block['history']), 3), 'n_trainable': one_block['n_trainable'], 'test': {{k: round(one_block_test[k], 3) for k in METRICS}}, 'by_category': one_block_fields, 'artifact_tensors': len(experiment_manifest['tensors']), 'artifact_bytes': experiment_manifest['files'][0]['bytes']}},\n"
                "        'two_blocks': {{'best_epoch': adapt_result['best_epoch'], 'best_adapted_val_cider_d': round(best_adapted_validation(adapt_result['history']), 3), 'n_trainable': adapt_result['n_trainable'], 'test': {{k: round(adapted_test[k], 3) for k in METRICS}}, 'by_category': adapted_fields, 'artifact_tensors': len(artifact_manifest['tensors']), 'artifact_bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "        'default_artifact_unchanged': True,\n"
                "    }}\n"
                "    experiment_path = Path(f\"outputs/{stem}_evaluation_report{{experiment_look['suffix']}}.json\")\n"
                "    experiment_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')\n"
                "    print({{'block_comparison_experiment': summary, 'report': str(experiment_path)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>This experiment has not been run on a hosted runtime, so there is no reference number. What to look for: the one-block artifact holds about half the tensors and bytes of the two-block one (the last block plus the final layer norm); with 70 test photographs and one seed a CIDEr-D difference of a few hundredths between the two runs is within what one split shows by chance, so read the per-category rows and the kept epoch rather than the headline number.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is a trained captioner scored on photographs from VizWiz, which is not in the checkpoint's TextCaps "
        "fine-tuning data (its web-screenshot pretraining corpus is not public, so overlap with it cannot be fully ruled out), "
        "beside two non-neural baselines, and a bounded fine-tuning of the caption decoder's last two blocks on 208 VizWiz "
        "photographs, with its learning rate chosen on validation, is then scored on a fresh, image-disjoint "
        "test — overall, and separately on the photographs with and without text — with an adapter "
        "that reloads to identical captions. That is the claim: the adaptation contract works end to end on a real "
        "out-of-distribution captioning corpus, and the numbers it produces are read on four metrics and per category "
        "against two non-neural baselines and the frozen model rather than in isolation. Whether the held-out CIDEr-D rose is "
        "recorded as `adapted_beats_frozen`, not assumed.\n\n"
        "The fresh test is the 324 captioned photographs of one row group (none excluded by the screen; the 12 unrecognisable, uncaptioned photographs are not measured, so a fluent caption of an unusable frame is never scored) of one shard of one corpus, the validation split that picks the "
        "learning rate and the epoch is 40, the metrics are four reference-based scores (pure-Python BLEU-4 and CIDEr-D following "
        "the `coco-caption` formulas and ROUGE-L with its aggregation, but with a simplified tokenisation rather than the PTB "
        "tokenizer, and CIDEr-D's document frequencies from the evaluated set — so absolute values are not comparable to "
        "leaderboard numbers — and none a human judgement). The 70-photograph exposed development test was used to choose recipes "
        "before (the BLIP captioning row's recipe search and this row's 2026-09-24 pre-flight, blob `2b72c3f7`, which kept epoch 2 "
        "of 4 at `LEARNING_RATE` 2e-5 and the last epoch at 1e-5 and 5e-5), so its deltas are not a held-out estimate. So a gain here says the contract works, not that the adapted model is better on "
        "your photographs, that it reads the labels it now mentions, or that its captions are faithful — it still generates "
        "a sentence for every image, quotes text that may not be there, and can be wrong fluently. Fine-tuning on a narrow "
        "corpus can also erode the model elsewhere; the drawn scenes re-captioned in Section 9 are three images of evidence "
        "about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the constant-caption and colour-neighbour baselines and "
        "the frozen model's score on *your* references are the numbers to read before any adapted one, per category and on "
        "CIDEr-D. **Leakage:** keep every record on an image in one split (the contract does this) and split by photographer "
        "or session when your images come from few sources, never at random over near-duplicate frames. **References:** "
        "every reference caption is a training target; a corpus with one caption per image trains the model on one "
        "annotator's phrasing, and CIDEr-D with a single reference rewards that phrasing rather than a consensus.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify the captions and "
        "photographs of a real captioning corpus, validate the demonstrated dataset contract without leakage, execute the "
        "inference contract and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model on an "
        "image-disjoint split, and emit the shown machine-readable artifacts — without the repository being reachable. It "
        "does **not** establish benchmark superiority, caption quality on any other population or camera, or production "
        "fitness.\n\n"
        "**Optional experiments (they do not affect the default path or its exported artifact until you re-run Section 9):** each "
        "re-run of the Section 7 cell first reloads the frozen pipeline from the verified snapshot when `pipe` already carries an "
        "adapter, so every experiment starts from the base weights and its epoch 0 equals the Section 6 frozen validation score. "
        "**Learning rate** — the grid already compares three rates on validation. To try another, set `RUN_GRID = False` and "
        "`LEARNING_RATE` in Section 7, then re-run the Section 7 and Section 8 cells: the run is exploratory, compares on "
        "validation and the exposed development test, and does not score the fresh test. The 2026-09-24 pre-flight (blob "
        "`2b72c3f7`) scored the exposed development test at three rates — 1e-5 gave CIDEr-D 0.325 with epoch 4 kept; 2e-5 gave 0.321 "
        "with **epoch 2** kept (`text` 0.503, `no-text` 0.078); 5e-5 gave 0.340 with epoch 4 kept (`text` 0.484, `no-text` 0.148) "
        "— and those test looks are exactly why the rate is now chosen on validation instead. Treat a loss that keeps falling while "
        "validation CIDEr-D stalls as the thing to watch for. "
        "**Blocks** — run the one-block comparison cell above Section 9's checkpoint, or set `TRAINABLE_DECODER_LAYERS = 1` and "
        "re-run the Section 7, 8 and 9 cells — exploratory, so no fresh-test score — (the artifact always reproduces the in-memory model because no tensor outside the "
        "exported set was ever trained). **Epochs** — raise `EPOCHS`, re-run Sections 7 and 8, and watch which epoch validation "
        "CIDEr-D picks (exploratory; no recorded run beyond four epochs). **Your data** — BYOD (Section 4 cell, then the Sections 5–9 cells in "
        "order) and read the two baselines before the adapted number.\n\n"
        '## Troubleshooting\n\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** or **CUDA out of memory** — restart the session and choose **Run all** on a GPU runtime; lower `BATCH_SIZE` if it repeats (numbers will differ slightly, and the run becomes exploratory, so the fresh test is not scored).\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again.\n'
        '- **A `sha256` or size error naming a VizWiz parquet or photograph in Section 4** — a cached file under `weights/vizwiz-captions/` is incomplete; delete it and run Section 4 again.\n'
        '- **"the candidate screen found …, pinned …" in Section 4** — the fresh-test photographs, the training split or the Pillow build differ from the pre-registration; run in the isolated environment Section 1 builds (pinned `pillow==11.3.0`). The fresh test is not scored until this passes.\n'
        '- **"This run is invalidated (BYOD input refused)"** — a later section after a refused BYOD zip: fix the zip named in the Section 4 message and run Section 4 again.\n'
        '- **Section 6 says the frozen model is not above the constant caption, or `adapted_beats_frozen` is `False`** — a finding worth recording on the default path; on your own data read the baselines and caption lengths first.\n'
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one …"** — set `BYOD_PATH` to the zip in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **BYOD refused: "the zip must hold exactly one records.jsonl", "exceed the BYOD ceiling", "points outside the zip", "hold fewer than MIN_RECORDS" or a `validate_dataset` refusal** — one records file at the root or in one folder, image paths relative to it (folders are kept), at most 5,002 members and 2 GiB extracted, at least 50 records at one per image; refusals name the zip, the record index and the rule.\n'
        '- **A re-run scores "frozen" numbers that differ from the first pass** — Sections 6 and 7 reload the frozen pipeline whenever `pipe` carries an adapter (they print `reloaded_frozen_pipeline`), so run the Section 6 cell again; `frozen_test["adapted"]` must be `False`.\n\n'
        '## Glossary\n\n'
        '- **OCR-free captioning** — generating a caption that quotes visible text, read from pixels without a separate OCR step.\n'
        '- **Greedy decoding / `max_new_tokens`** — always taking the most likely next token, up to a caller-owned budget; a caption that hits the budget is truncated.\n'
        '- **BLEU-4 / ROUGE-L / CIDEr-D / unigram F1** — n-gram precision with a brevity penalty; longest-common-subsequence F-measure; TF-IDF-weighted n-gram consensus over all references (the headline metric); word overlap as a plumbing check.\n'
        '- **Constant-caption / colour-neighbour baseline** — one training caption (the corpus medoid) for every photograph; the caption of the training photograph with the closest 3×3 mean-colour grid.\n'
        '- **`text` / `no-text`** — photographs whose annotators flagged readable text, and those without.\n'
        '- **Photograph-level split** — every caption of one photograph stays in one split.\n'
        '- **Epoch / validation selection** — one pass over the training records; keeping the epoch with the best validation score (epoch 0, the frozen model, kept unless an adapted epoch beats it by 0.02).\n'
        '- **Fresh test** — photographs never fetched, scored or used for any choice before the pre-registration; scored only under the pre-registered settings.\n'
        '- **Exposed development test** — the 70 photographs earlier recipes were chosen on; still image-disjoint from training, but no longer a held-out estimate.\n'
        '- **Pre-registration / `test_look`** — settings and selection rules fixed before any run; `test_look` says whether a run was the pre-registered evaluation, a reproduction of it, or exploratory.\n'
        '- **Candidate screen / confirmed near-copy** — a 64-bit difference hash flags fresh photographs that look like a training photograph (candidates); only a recorded review confirming the same object or scene excludes one.\n'
        '- **Adapter / reload parity** — the trained decoder tensors only, overlaid on the pinned base; the reloaded model gives identical outputs.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: your images and labels through the same cells.\n\n'
        '## Conclusion (your notes)\n\nWrite your conclusion from **your** run (optional); the recorded run is only a reference:\n\n'
        '- The `test_look` of my run was ___; the grid chose learning rate ___ (validation CIDEr-D ___ / ___ / ___).\n'
        '- Constant ___, colour neighbour ___, frozen ___, adapted ___ (fresh-test CIDEr-D); the `text` / `no-text` scores were ___ / ___.\n'
        '- The kept epoch was ___; the fresh-test delta over frozen was ___ CIDEr-D (95 % interval ___ to ___), its sign on the exposed development test was ___, which means ___.\n'
        '- One caption pair that shows the change: ___ → ___.\n'
        '- One reason not to trust this gain on my own photographs yet: ___.\n'
        '\n'
        "## References\n\n"
        "[1] K. Lee, M. Joshi, I. Turc, H. Hu, F. Liu, J. Eisenschlos, U. Khandelwal, P. Shaw, M.-W. Chang, and K. Toutanova, \"Pix2Struct: Screenshot parsing as pretraining for visual language understanding,\" in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 202, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2210.03347\n\n"
        "[2] O. Sidorov, R. Hu, M. Rohrbach, and A. Singh, \"TextCaps: A dataset for image captioning with reading comprehension,\" in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020. [Online]. Available: https://doi.org/10.48550/arXiv.2003.12462\n\n"
        "[3] D. Gurari, Y. Zhao, M. Zhang, and N. Bhattacharya, \"Captioning images taken by people who are blind,\" in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020 (VizWiz-Captions, CC BY 4.0). [Online]. Available: https://doi.org/10.48550/arXiv.2002.08565 — data: https://vizwiz.org/tasks-and-datasets/image-captioning/\n\n"
        "[4] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, \"BLEU: A method for automatic evaluation of machine translation,\" in *Proc. 40th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2002. [Online]. Available: https://aclanthology.org/P02-1040/\n\n"
        "[5] C.-Y. Lin, \"ROUGE: A package for automatic evaluation of summaries,\" in *Text Summarization Branches Out (ACL Workshop)*, 2004. [Online]. Available: https://aclanthology.org/W04-1013/\n\n"
        "[6] R. Vedantam, C. L. Zitnick, and D. Parikh, \"CIDEr: Consensus-based image description evaluation,\" in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2015. [Online]. Available: https://doi.org/10.48550/arXiv.1411.5726\n\n"
        "Repository README: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/README.md · model card: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/MODEL_CARD.md · weight provenance: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/docs/WEIGHTS.md · pre-registration: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/docs/learning-rate-grid-preregistration.md · upstream model (Google, Apache-2.0): https://huggingface.co/{MODEL_ID} · upstream code: https://github.com/google-research/pix2struct · DIMER Notebook Specification 2.5 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
