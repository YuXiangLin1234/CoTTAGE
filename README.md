<div align="center">

# CoTTAGE: An Explainable Judge for Text-to-Audio Generation

[![Hugging Face Model](https://img.shields.io/badge/Model-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/AutoAudioReasoningMOS/CoTTAGE)
[![Hugging Face Datasets](https://img.shields.io/badge/Datasets-9C27B0?logo=huggingface&logoColor=fff)](https://huggingface.co/datasets?author=AutoAudioReasoningMOS)
[![Demo](https://img.shields.io/badge/Demo-4285F4)](https://yuxianglin1234.github.io/CoTTAGE/)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2611.17644)

<!-- TODO: Add the arXiv URL to the badge above. -->

<p>
CoTTAGE is an audio judge based on Qwen2.5-Omni-7B. It explains its judgment,
then predicts a MOS from 1.00 to 5.00 for text relevance (REL) or audio quality (OVL).
</p>

<p>
In addition to the existing REL annotations, we collected human OVL annotations for the RELATE and XACLE training sets, covering 10,362 audio clips.
</p>

<p align="center">
<img src="./CoTTAGE%20Teaser.png" alt="CoTTAGE compares audio with a text prompt and explains a reversed event order, alongside CLAPScore and Qwen2.5-Omni predictions.">
</p>

</div>

- **REL**: semantic alignment between the clip and its text instruction
- **OVL**: overall audio quality of the clip

## Contents

- [Installation](#installation)
- [Models](#models)
- [Quick start](#quick-start)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)
- [Repository layout](#repository-layout)

## Installation

```bash
conda create -n cottage python=3.10 && conda activate cottage
pip install -r requirements.txt
```

Tested with:

| Package | Version |
|---|---|
| ms-swift | 3.12 |
| transformers | 4.57 |
| torch | 2.8 |
| vLLM | 0.15.1|

## Models

| Checkpoint | Description |
|---|---|
| `AutoAudioReasoningMOS/CoTTAGE` | Final CoTTAGE model for inference and evaluation |

Every script that takes a model (`--model` / `MODEL=`) accepts either:

- a merged full model, or
- the merged SFT model plus a LoRA adapter (`--adapter path/to/checkpoint-N`,
  or `ADAPTER=` for `scripts/evaluate.sh`).

## Quick start

Score a single clip on REL (needs the text instruction) or OVL:

```bash
export MODEL=AutoAudioReasoningMOS/CoTTAGE

python cottage/judge.py --model "$MODEL" --audio clip.wav \
    --text "A loud motor begins to pick up speed then cuts out and slowly stops."

python cottage/judge.py --model "$MODEL" --audio clip.wav --dimension ovl
```

The model answers in a fixed format, so the score can be extracted with a regex:

```
<think>
The text instruction describes a sequence of events: a loud motor beginning to pick up speed, then cutting out ... the audio remains at a constant, steady pitch and intensity from the beginning to the end ... 
</think>
<score>1.80</score>
```

## Data

### Evaluation benchmarks

| Benchmark | Clips | Dimensions | Source |
|---|---|---|---|
| XACLE | 3,000 | REL | [XACLE Challenge](https://xacle.org/) |
| RELATE | 1,311 | REL | [RELATE](https://github.com/sarulab-speech/RELATE) |
| PAM-Audio | 500 | REL, OVL | [PAM](https://github.com/soham97/PAM) human evaluation set |
| Clotho | 500 | REL, OVL | [AudioBERTScore](https://github.com/lourson1091/audiobertscore) |

### Training data

- **Audio**: the XACLE training split (7,500 clips) and the RELATE training
  split (2,862 clips), fetched by the same download script.
- **MOS labels and rationales**: You can pull from the Hugging Face Hub, `AutoAudioReasoningMOS/{xacle_rel,xacle_ovl,relate_rel,relate_ovl}`. No manual download is needed.

### Download

```bash
bash scripts/download_data.sh
```

The script fetches XACLE and RELATE from their official servers, PAM-Audio from Zenodo, and the Clotho MOS data from the AudioBERTScore repository.
Downloads are resumable, and dataset directories that already exist are skipped.

Everything lands under `data/`:

```
data/
├── xacle/wav/train/           XACLE training clips
├── xacle/test_16k/            XACLE test clips
├── relate/wav/                RELATE clips (train + test)
├── pam/human_eval/audio/      PAM clips, one sub-directory per TTA system,
│                              plus scores.csv
└── clotho/                    clotho_ovl_rel_test_set.csv + wave_all_16k/
```

If your copies live elsewhere, override the paths defined in `config.sh`, e.g. `XACLE_TEST_DIR=/data/xacle_test_16k bash scripts/evaluate.sh`.

## Evaluation

### All benchmarks

```bash
MODEL=$MODEL bash scripts/evaluate.sh
MODEL=ckpts/sft-merged ADAPTER=ckpts/rl/checkpoint-N bash scripts/evaluate.sh
```

This runs all four benchmarks and prints the correlation table.

### A single benchmark

```bash
python cottage/run_inference.py --model $MODEL --benchmark pam --dimension all
```

| Option | Values |
|---|---|
| `--benchmark` | `xacle`, `relate`, `pam`, `clotho` |
| `--dimension` | `rel`, `ovl`, `all` |
| `--num_shards`, `--shard_index` | split a benchmark across GPUs |
| `--limit N` | only the first N rows (smoke test) |

Output goes to `results/<benchmark>/<run_name>_<REL|OVL>.json`, one row per clip with the human score and the raw response.

### Correlations

```bash
python cottage/compute_correlation.py results/*/cottage_*.json
```

## Training

### Pipeline

```bash
bash scripts/download_data.sh
bash scripts/sft.sh                                  # -> ckpts/sft
bash scripts/merge_lora.sh ckpts/sft/checkpoint-N    # -> ckpts/sft-merged
bash scripts/rl.sh                                   # -> ckpts/rl
```

RL starts from the **merged** SFT model, so the merge step is required.

The resulting RL adapter can be evaluated directly on top of the merged SFT model (see [Evaluation](#evaluation)) or merged in turn with `scripts/merge_lora.sh <adapter> <output_dir> ckpts/sft-merged`.
