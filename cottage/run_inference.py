import argparse
import csv
import json
import os
from collections import namedtuple

from tqdm import tqdm

from audio_utils import ensure_16k
from prompts import OVL_EVAL_PROMPT, REL_EVAL_PROMPT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO_ROOT, "data"))

Benchmark = namedtuple("Benchmark", "kind source audio_root dimensions")

def data_path(variable, *parts):
    return os.environ.get(variable, os.path.join(DATA_ROOT, *parts))


BENCHMARKS = {
    "xacle": Benchmark(
        "xacle_csv",
        data_path("XACLE_TEST_CSV", "xacle", "test.csv"),
        data_path("XACLE_TEST_DIR", "xacle", "wav", "test"),
        ("rel",),
    ),
    "relate": Benchmark(
        "relate_csv",
        data_path("RELATE_CSV", "relate", "scores", "REL.csv"),
        data_path("RELATE_AUDIO_DIR", "relate", "wav"),
        ("rel",),
    ),
    "pam": Benchmark(
        "csv",
        data_path("PAM_CSV", "pam", "human_eval", "audio", "scores.csv"),
        data_path("PAM_AUDIO_ROOT", "pam", "human_eval", "audio"),
        ("rel", "ovl"),
    ),
    "clotho": Benchmark(
        "csv",
        data_path("CLOTHO_CSV", "clotho", "scores.csv"),
        data_path("CLOTHO_AUDIO_ROOT", "clotho", "wav"),
        ("rel", "ovl"),
    ),
}


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _column(row, *names):
    lowered = {key.lower().strip(): value for key, value in row.items()}
    return next((lowered[name] for name in names if name in lowered), None)


def _load_xacle_rows(benchmark):
    rows = []
    with open(benchmark.source, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            wave_filename = _column(row, "wavname", "wave_filename", "wav_file_name", "file name", "filename")
            if not wave_filename:
                continue
            rows.append({
                "clip_id": wave_filename,
                "instruction": _column(row, "text", "caption", "instruction") or "",
                "audio": os.path.join(benchmark.audio_root, wave_filename.lstrip("/")),
                "rel": _as_float(_column(row, "score", "mos", "average", "average_score", "rel")),
                "ovl": None,
            })
    return rows


def _load_relate_rows(benchmark):
    grouped = {}
    with open(benchmark.source, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row.get("in RELATE dataset") or "").lower() != "test":
                continue
            key = (row["wavname"], row["text"])
            grouped.setdefault(key, []).append(float(row["score"]))
    return [{
        "clip_id": wavname.lstrip("/"),
        "instruction": instruction,
        "audio": os.path.join(benchmark.audio_root, wavname.lstrip("/")),
        "rel": sum(scores) / len(scores),
        "ovl": None,
    } for (wavname, instruction), scores in grouped.items()]


def _load_csv_rows(benchmark):
    rows = []
    with open(benchmark.source, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model, file_name = row["Model"], row["File Name"]
            wav_name = file_name if file_name.endswith(".wav") else f"{file_name}.wav"
            rows.append({
                "clip_id": f"{model}/{file_name}",
                "instruction": row["Text"],
                "audio": os.path.join(benchmark.audio_root, model, wav_name),
                "rel": _as_float(row.get("REL")),
                "ovl": _as_float(row.get("OVL")),
            })
    return rows


def load_rows(name, benchmark, shard_index, num_shards, limit):
    loaders = {"xacle_csv": _load_xacle_rows, "relate_csv": _load_relate_rows, "csv": _load_csv_rows}
    rows = loaders[benchmark.kind](benchmark)
    rows = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
    if limit:
        rows = rows[:limit]

    cache_root = os.path.join(data_path("AUDIO_CACHE_DIR", "cache"), "eval", name)
    resolved, missing = [], 0
    for row in tqdm(rows, desc=f"preparing {name}"):
        if not os.path.exists(row["audio"]):
            missing += 1
            continue
        cache_name = row["clip_id"].lstrip("/")
        if not cache_name.endswith(".wav"):
            cache_name += ".wav"
        row["audio"] = ensure_16k(row["audio"], os.path.join(cache_root, cache_name))
        resolved.append(row)
    if missing:
        print(f"[{name}] skipped {missing} rows with no audio file")
    return resolved


def build_engine(args):
    from swift.llm import VllmEngine

    kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.adapter:
        kwargs.update(adapters=[args.adapter], max_lora_rank=args.max_lora_rank)
    return VllmEngine(args.model, **kwargs)


def run_dimension(engine, request_config, rows, dimension, output_path, batch_size):
    from swift.llm import InferRequest

    results = []
    for start in tqdm(range(0, len(rows), batch_size), desc=f"infer {dimension}"):
        batch = rows[start:start + batch_size]
        requests = [
            InferRequest(messages=[{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": row["audio"]},
                    {"type": "text", "text": REL_EVAL_PROMPT.format(instruction=row["instruction"])
                     if dimension == "rel" else OVL_EVAL_PROMPT},
                ],
            }])
            for row in batch
        ]
        for row, response in zip(batch, engine.infer(requests, request_config)):
            results.append({
                "clip_id": row["clip_id"],
                "instruction": row["instruction"] if dimension == "rel" else "",
                "dimension": dimension,
                "gt": row[dimension],
                "response": response.choices[0].message.content,
            })
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"[done] {len(results)} rows -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--model", required=True, help="base model path or HuggingFace id")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter directory")
    parser.add_argument("--dimension", default="all", choices=["rel", "ovl", "all"])
    parser.add_argument("--run_name", default="cottage")
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N rows (smoke test)")
    return parser.parse_args()


def main():
    from swift.llm import RequestConfig

    args = parse_args()
    benchmark = BENCHMARKS[args.benchmark]
    dimensions = benchmark.dimensions if args.dimension == "all" else (args.dimension,)
    unsupported = set(dimensions) - set(benchmark.dimensions)
    if unsupported:
        raise SystemExit(f"{args.benchmark} has no ground truth for {sorted(unsupported)}")

    rows = load_rows(args.benchmark, benchmark, args.shard_index, args.num_shards, args.limit)
    print(f"[{args.benchmark}] {len(rows)} clips, dimensions={dimensions}")

    output_dir = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(output_dir, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_index}"

    engine = build_engine(args)
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
    for dimension in dimensions:
        output_path = os.path.join(output_dir, f"{args.run_name}_{dimension}{suffix}.json")
        run_dimension(engine, request_config, rows, dimension, output_path, args.batch_size)


if __name__ == "__main__":
    main()
