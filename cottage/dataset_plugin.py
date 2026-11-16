import json
import os
import random
from collections import namedtuple

import filelock
from datasets import Audio, concatenate_datasets, load_dataset
from swift.llm import DatasetMeta, register_dataset

from audio_utils import duration_seconds, ensure_16k
from prompts import OVL_EVAL_PROMPT, REL_EVAL_PROMPT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO_ROOT, "data"))

AudioDirs = namedtuple("AudioDirs", "relate xacle cache")
AUDIO_DIRS = AudioDirs(
    relate=os.environ.get("RELATE_AUDIO_DIR", os.path.join(DATA_ROOT, "relate", "wav")),
    xacle=os.environ.get("XACLE_AUDIO_DIR", os.path.join(DATA_ROOT, "xacle", "wav", "train")),
    cache=os.environ.get("AUDIO_CACHE_DIR", os.path.join(DATA_ROOT, "cache")),
)

TRAIN_DATASETS = {
    ("relate", "rel"): "AutoAudioReasoningMOS/relate_rel",
    ("relate", "ovl"): "AutoAudioReasoningMOS/relate_ovl",
    ("xacle", "rel"): "AutoAudioReasoningMOS/xacle_rel",
    ("xacle", "ovl"): "AutoAudioReasoningMOS/xacle_ovl",
}

MIN_DURATION = 0.5
NUM_PROC = int(os.environ.get("DATASET_NUM_PROC", "4"))
SEED = int(os.environ.get("DATASET_SEED", "531"))
PAIR_CHUNK_ROWS = int(os.environ.get("PAIR_CHUNK_ROWS", "16"))


def _locate(wave_filename, source, dirs):
    from_xacle = source == "xacle"
    if from_xacle:
        return os.path.join(dirs.xacle, wave_filename), True
    return os.path.join(dirs.relate, wave_filename), False


def _audio_path(wave_filename, source, dirs):
    path, already_16k = _locate(wave_filename, source, dirs)
    if already_16k:
        return path
    return ensure_16k(path, os.path.join(dirs.cache, wave_filename))


def _has_usable_audio(row, source, dirs):
    path, _ = _locate(row["wave_filename"], source, dirs)
    return duration_seconds(path) > MIN_DURATION


def _user_turn(row, source, dimension, dirs):
    prompt = REL_EVAL_PROMPT.format(instruction=row["text"]) if dimension == "rel" else OVL_EVAL_PROMPT
    return {
        "role": "user",
        "content": [
            {"type": "audio", "audio": _audio_path(row["wave_filename"], source, dirs)},
            {"type": "text", "text": prompt},
        ],
    }


def _sft_row(row, source, dimension, dirs):
    return {
        "messages": [
            _user_turn(row, source, dimension, dirs),
            {"role": "assistant", "content": [{"type": "text", "text": row["reasoning"]}]},
        ]
    }


def _rl_row(row, source, dimension, dirs):
    return {
        "messages": [_user_turn(row, source, dimension, dirs)],
        "solution": json.dumps({
            "gt_score": float(row["score"]),
            "dimension": dimension,
        }),
    }


def _download(hf_id):
    os.makedirs(DATA_ROOT, exist_ok=True)
    with filelock.FileLock(os.path.join(DATA_ROOT, f".{hf_id.replace('/', '__')}.lock")):
        dataset = load_dataset(hf_id, split="train")
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def _build(hf_id, source, dimension, row_fn, dirs=AUDIO_DIRS):
    dataset = _download(hf_id)
    kwargs = {"source": source, "dirs": dirs}
    dataset = dataset.filter(_has_usable_audio, fn_kwargs=kwargs, num_proc=NUM_PROC)
    return dataset.map(
        row_fn,
        fn_kwargs={**kwargs, "dimension": dimension},
        remove_columns=dataset.column_names,
        num_proc=NUM_PROC,
    )


def _sources(row_fn):
    return [
        _build(hf_id, source, dimension, row_fn)
        for (source, dimension), hf_id in TRAIN_DATASETS.items()
    ]


def load_cottage_sft(dataset_id, *args, **kwargs):
    parts = _sources(_sft_row)
    dataset = concatenate_datasets(parts).shuffle(seed=SEED)
    print(f"[cottage_sft] {len(dataset)} rows = " + " + ".join(str(len(p)) for p in parts))
    return dataset


def _pair_rows(dataset, tag, seed):
    dataset = dataset.shuffle(seed=seed)
    if len(dataset) % 2:
        dataset = dataset.select(range(len(dataset) - 1))

    def label(row, index):
        solution = json.loads(row["solution"])
        solution["pair_id"] = f"{tag}-{index // 2}"
        solution["position"] = "a" if index % 2 == 0 else "b"
        return {"solution": json.dumps(solution)}

    return dataset.map(label, with_indices=True)


def _interleave_chunks(parts, chunk_rows, seed):
    """Shuffle the parts together in contiguous single-source chunks of one
    optimizer step, so REL and OVL stay spread out but never share a batch."""
    chunks, offset = [], 0
    for part in parts:
        kept = (len(part) // chunk_rows) * chunk_rows
        chunks += [range(offset + start, offset + start + chunk_rows) for start in range(0, kept, chunk_rows)]
        if kept < len(part):
            print(f"[cottage_rl] dropped {len(part) - kept} rows that did not fill a chunk")
        offset += len(part)

    random.Random(seed).shuffle(chunks)
    return concatenate_datasets(parts).select([index for chunk in chunks for index in chunk])


def load_cottage_rl(dataset_id, *args, **kwargs):
    relate_rel, relate_ovl, xacle_rel, xacle_ovl = _sources(_rl_row)
    rel_pairs = _pair_rows(concatenate_datasets([relate_rel, xacle_rel]), "rel", SEED)
    ovl_pairs = _pair_rows(concatenate_datasets([relate_ovl, xacle_ovl]), "ovl", SEED)
    dataset = _interleave_chunks([rel_pairs, ovl_pairs], PAIR_CHUNK_ROWS, SEED)
    print(f"[cottage_rl] {len(dataset)} rows = {len(dataset) // 2} pairs "
          f"(rel {len(rel_pairs) // 2} + ovl {len(ovl_pairs) // 2}), chunk={PAIR_CHUNK_ROWS}")
    return dataset


register_dataset(DatasetMeta(ms_dataset_id="cottage_sft", load_function=load_cottage_sft))
register_dataset(DatasetMeta(ms_dataset_id="cottage_rl", load_function=load_cottage_rl))
