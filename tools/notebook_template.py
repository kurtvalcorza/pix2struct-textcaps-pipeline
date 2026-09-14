"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_textcaps_pipeline",
    "repo_name": "pix2struct-textcaps-pipeline",
    "stem": "pix2struct_textcaps",
    "notebook_name": "pix2struct_textcaps_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "Pix2StructTextCapsPipeline",
    "weights_key": "pix2struct-textcaps-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct TextCaps-base — DIMER text-aware image captioning tutorial (standalone)",
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
    "capability": "Text-aware image captioning — one image (optionally with a caption prefix to continue) → one sentence that quotes the text visible in the image — using the pinned `google/pix2struct-textcaps-base` weights",
    "intro": (
        "At inference the Pix2Struct image-encoder/text-decoder (a ViT-style encoder over variable-resolution 16×16 patches "
        "and a 12-layer text decoder, 282M parameters, pretrained by parsing masked web screenshots into HTML and fine-tuned "
        "on TextCaps, whose captions must mention the text in the image) scales the image to fill at most 2048 patches, "
        "reads any text in it from pixels, and generates the caption token by token, either from scratch (unconditional) or "
        "continuing a prefix you supply such as `A picture of` (conditional, the upstream README's example). Decoding is "
        "greedy (`do_sample=False`, one beam) under a caller-owned `max_new_tokens` budget. **No adaptation occurs:** no "
        "training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the upstream "
        "checkpoint supplies the weights, processor and tokenizer, and the carried module adds snapshot verification, the "
        "input contract (image side ceilings, an optional prefix up to 128 characters, the token budget), a fixed output "
        "contract, and the `text_recall`, `keyword_hits`, `unigram_f1`, `validate_inputs` and `evaluation_report` helpers. "
        "The default sample is three flat cartoon scenes drawn in code with **known text rendered into them**, so `text_recall` "
        "— the fraction of the drawn words the caption quotes — is demonstration (plumbing) evidence for the OCR-free reading "
        "path, not a TextCaps benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw three synthetic scenes with known text (or upload your own photographs of "
        "signs and labels) and validate them into an input manifest, choose a token budget and an optional prefix, run the "
        "supported task, read the captions correctly (generated text, no score, a `truncated` flag), exercise an optional "
        "BYOD path, produce an evaluation report that is `sample-sanity` with `text_recall` when the drawn text is known "
        "and `not-measurable` otherwise, and export the captions, an annotated contact sheet and provenance."
    ),
    "exclusions": (
        "General OCR (the model quotes text it deems caption-worthy, not every string in the image, and returns no "
        "transcript or location), visual question answering and document QA (separate checkpoints), captions in "
        "languages other than English, batch throughput, sampling, beam search or repetition penalties (greedy decoding for "
        "reproducibility), evaluation on the TextCaps benchmark (not bundled; CIDEr and BLEU-4 need several references per "
        "image and are not computed here), and any training. The model was fine-tuned on photographs of signs, products, "
        "screens and labels; flat drawings, documents and diagrams are outside what this notebook measures, and a fluent "
        "wrong caption carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.6 s to load and 2.8–3.4 s per caption on the drawn scenes in the Windows venv (Intel Core Ultra 9 275HX) — the 2048-patch encoder pass dominates. The pinned `torch==2.14.0` install and the 1.13 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what reference-based caption metrics (CIDEr, BLEU) need and why token recall of drawn text is not one of them; that a confident caption is not a correct one.",
        "- **Data:** the default sample is three deterministic cartoon scenes drawn in code with Pillow's bundled font (a red octagonal sign reading `STOP`; a storefront reading `Blue Fern Bakery` with an `OPEN` card; a green jersey reading `LIONS` and `42`), each with the list of strings drawn into it, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one or more images decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px; the drawn-text list is unknown for uploads, so their report is `not-measurable`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scenes or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: three flat cartoon scenes with text "
                "rendered into them using Pillow's bundled font — a red octagonal `STOP` sign on a pole; a storefront whose "
                "fascia reads `Blue Fern Bakery` with a small `OPEN` card in the window; a green jersey reading `LIONS` above "
                "the number `42` — the same drawings the repository's smoke run used. The strings drawn into each scene are "
                "the references for the `text_recall` sanity check later. They are not a labelled dataset, so nothing here is "
                "a TextCaps measurement; two of the smoke run's misses are known in advance (the model did not quote `OPEN` "
                "or `LIONS`, and called the green jersey blue). The image digests are printed for the record; they depend on "
                "the Pillow build's bundled font rendering. BYOD is optional and disabled by default; when enabled, upload one "
                "or more images — the drawn text is unknown for them, so the evaluation report will be `not-measurable`.\n\n"
                "Two **caller-owned request parameters** are exposed: `max_new_tokens` bounds the caption "
                "(`DEFAULT_MAX_NEW_TOKENS = 30` fits any TextCaps-style sentence; `MAX_NEW_TOKENS = 64` is the ceiling), and "
                "`caption_prefix` (empty for unconditional captioning; `A picture of` is the upstream README's example — the "
                "model continues whatever you start, sometimes mid-word: the smoke run produced `A redoptical sign that says "
                "STOP.` from the prefix `A red`, so the default here is unconditional). Nothing is validated in this cell — the "
                "next section hands the images to the pipeline's own validation stage, which is the only checker. Look for one "
                "dictionary per image naming the sample kind, size, digest and drawn text, plus the budget and prefix."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import math\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "max_new_tokens = 30  # @param {{type:\"integer\"}}\n"
                "caption_prefix = ''  # @param {{type:\"string\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    samples = []\n"
                "    for name, data in uploaded.items():\n"
                "        image = Image.open(io.BytesIO(data))\n"
                "        image.load()\n"
                "        samples.append((name, image, None))\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawings: no randomness, so no seed is needed; the digests depend on the Pillow build's bundled font.\n"
                "    samples = synthetic_scenes()\n"
                "    sample_kind = 'synthetic'\n\n"
                "names = [name for name, _, _ in samples]\n"
                "images = [image for _, image, _ in samples]\n"
                "drawn_texts = [texts for _, _, texts in samples] if sample_kind == 'synthetic' else None\n"
                "prefix = caption_prefix.strip() or None\n"
                "digests = {{name: hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() for name, image in zip(names, images)}}\n"
                "for index, (name, image) in enumerate(zip(names, images)):\n"
                "    print({{'sample_kind': sample_kind, 'name': name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': digests[name], 'drawn_text': drawn_texts[index] if drawn_texts else None}})\n"
                "print({{'max_new_tokens': max_new_tokens, 'prefix': prefix, 'n_images': len(images)}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `caption` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, an optional prefix that is a non-empty string of "
                "at most `MAX_PREFIX_CHARS` characters (whitespace collapsed), and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — "
                "and returns an **input manifest** naming the schema (including the patch-budget preprocessing and the decoding "
                "rule), each input's observed mode and size, the checked prefix, the budget and the verdict. The manifest is "
                "written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "4-pixel image and records the pipeline's own error message as a finding. Inside the pipeline each image is "
                "converted to RGB and scaled to fill at most `MAX_PATCHES` 16×16 patches (aspect ratio preserved); nothing else "
                "is dropped or altered. The pipeline cannot tell whether an image contains text worth quoting: that contract is "
                "the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_PREFIX_CHARS': MAX_PREFIX_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "input_manifest = validate_inputs(images, prefix=prefix, max_new_tokens=max_new_tokens, names=names)\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs([Image.new('RGB', (4, 4))])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'tiny-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Caption the images and read the output correctly\n\n"
                "`caption` returns, per image, a dict with `caption` (the decoded text, stripped; the prefix is echoed at the "
                "start when one was given), the checked `prefix`, `image_size`, `new_tokens` (tokens generated after the "
                "prefix, end-of-sequence included), a `truncated` flag that is true when the budget was exhausted, the "
                "generation settings and the model identity. **No score exists**: the caption is generated text with no "
                "probability and no correctness signal, and a fluent caption is not evidence that it describes the image or "
                "that the quoted text is really there. Greedy decoding is deterministic on a fixed device and dtype; CUDA kernel "
                "selection can change a token and therefore the rest of the sentence, so GPU and CPU outputs need not match. "
                "Each call re-encodes the image at up to 2048 patches, so cost is per image (about 2.8–3.4 s on the reference "
                "CPU). As recorded in the model card, the repository's CPU smoke captioned these same drawings `A stop sign is "
                "on a pole.`, `A blue sign that says Blue Fern Bakery.` and `A blue shirt with the number 42 on it.` (missing "
                "`OPEN` and `LIONS`, and miscolouring the jersey) — and captioned a blank white image `A man is holding a "
                "bottle of alcohol and the bottle says 'the man's'.`: the model always produces a caption with quoted text, "
                "whether or not there is any."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for name, image in zip(names, images):\n"
                "    t0 = time.time()\n"
                "    result = pipe.caption(image, prefix=prefix, max_new_tokens=max_new_tokens)\n"
                "    result['image'] = name\n"
                "    results.append(result)\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds_per_image': seconds, 'any_truncated': any(r['truncated'] for r in results)}})\n"
                "for result in results:\n"
                "    print(f\"{{result['image']}}\\n   caption: {{result['caption']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\")\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('A budget was exhausted: that caption is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No caption quality "
                "is reported by default: TextCaps-style metrics (CIDEr, BLEU-4) need several human-written reference captions "
                "per image and corpus-level statistics, and this repository ships none (TextCaps is not bundled). The "
                "repository's helper `text_recall` — the fraction of the distinct normalised tokens of the strings drawn into "
                "an image that appear in its caption, with the found and missing tokens listed — is what the report uses when "
                "the drawn text is known: on the synthetic path those strings are text **you drew yourself**, so a high recall "
                "proves only that the input contract, patch extraction, forward pass and decoding round-trip and that the model "
                "reads rendered text, and the verdict is `sample-sanity`; the two known misses show what a partial recall looks "
                "like. On BYOD no drawn text is known, the verdict is `not-measurable`, and the report states what would make "
                "the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, drawn_texts, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_image')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:12}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for entry in report.get('per_image', []):\n"
                "    print(f\"  text_recall {{entry['text_recall']:.2f}}  {{entry['image']}} -> {{entry['prediction']!r}} (found: {{entry['found']}}, missing: {{entry['missing']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('The text in these images is not known to the notebook, so nothing is scored; read the captions against the images yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (image, caption, prefix, `new_tokens`, `truncated`, the budget), "
                "the evaluation report, the input manifest, the sample identities, digests and drawn strings, the notebook's "
                "source (repository, revision, embedded module digest, generator), the model identifier, the immutable model "
                "revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The captions "
                "are also written as CSV with explicit `image`, `prefix`, `caption`, `new_tokens`, `truncated` columns, and an "
                "annotated PNG contact sheet shows each image with its caption printed beneath it for visual inspection (the "
                "model returns no location, so nothing is drawn on the images themselves) — a supplement to, not a replacement "
                "for, the machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "thumb_w, thumb_h, panel_h = 320, 280, 44\n"
                "sheet = Image.new('RGB', (thumb_w * len(images), thumb_h + panel_h), 'white')\n"
                "draw = ImageDraw.Draw(sheet)\n"
                "panel_font = ImageFont.load_default(size=13)\n"
                "for index, (image, result) in enumerate(zip(images, results)):\n"
                "    thumb = image.convert('RGB').copy()\n"
                "    thumb.thumbnail((thumb_w, thumb_h))\n"
                "    sheet.paste(thumb, (index * thumb_w + (thumb_w - thumb.width) // 2, (thumb_h - thumb.height) // 2))\n"
                "    draw.text((index * thumb_w + 6, thumb_h + 6), result['caption'][:60], fill=(40, 90, 220), font=panel_font)\n"
                "    if len(result['caption']) > 60:\n"
                "        draw.text((index * thumb_w + 6, thumb_h + 24), result['caption'][60:120], fill=(40, 90, 220), font=panel_font)\n"
                "sheet.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'names': names, 'sizes': [list(image.size) for image in images], 'rgb_sha256': digests, 'drawn_texts': drawn_texts}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_captions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'prefix', 'caption', 'new_tokens', 'truncated'])\n"
                "    for result in results:\n"
                "        writer.writerow([result['image'], result['prefix'] or '', result['caption'], result['new_tokens'], result['truncated']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The captions are the text the model generates for an image, quoting whatever text it reads from the pixels; nothing "
        "in the output scores that text, the model returns no transcript or location, and it captions every image — "
        "including a blank one, for which it invented a bottle with writing on it — with equal fluency. On the drawn scenes the "
        "`text_recall` values in the evaluation report compare the captions with strings you drew yourself and the verdict is "
        "`sample-sanity`, which proves only that the input contract, patch extraction, forward pass and decoding work and that "
        "rendered text is read (the repository's smoke run recalled `STOP` fully, `Blue Fern Bakery` but not `OPEN`, and `42` "
        "but not `LIONS`); they say nothing about photographs, curved or stylised type, small print, non-Latin scripts, "
        "attributes the model hallucinates (it called the green jersey blue), or captions longer than a sentence, and a BYOD "
        "result is a per-image observation with the verdict `not-measurable`. **The model captions any image** and stops only "
        "at end-of-sequence or the token budget: check `truncated`, and treat a plausible caption with plausible quoted text on "
        "an image that has none as the expected failure mode, not an exception. A prefix steers the caption and can be "
        "continued mid-word (`A redoptical sign that says STOP.`), so a conditional caption is partly your sentence. The "
        "pipeline provides no OCR transcript, no VQA, no localisation, no benchmark evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** set `caption_prefix` to `A picture of` and compare (the smoke run got `A picture of is shown with a "
        "blue sign that says Blue Fern Bakery.` for the storefront and an immediate end-of-sequence for the stop sign); change "
        "the drawn strings in `synthetic_scenes` and watch `text_recall` follow; lower `max_new_tokens` to 3 and watch "
        "`truncated` turn true on `A stop sign`; enable `USE_BYOD` with photographs of signs you know, then pass the strings you "
        "can read yourself as `drawn_texts` to `evaluation_report` to see the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/pix2struct-textcaps-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., 2022): https://arxiv.org/abs/2210.03347\n"
        "- TextCaps: a Dataset for Image Captioning with Reading Comprehension (Sidorov et al., 2020): https://arxiv.org/abs/2003.12462\n"
        "- CIDEr: Consensus-based Image Description Evaluation (Vedantam, Zitnick, Parikh, 2015): https://arxiv.org/abs/1411.5726"
    ),
}
