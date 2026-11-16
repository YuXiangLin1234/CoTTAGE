"""Score a single audio clip with CoTTAGE.

    python cottage/judge.py --model <model> --audio clip.wav --text "a dog barking"
    python cottage/judge.py --model <model> --audio clip.wav --dimension ovl
"""
import argparse
import os
import tempfile

from audio_utils import ensure_16k
from prompts import OVL_EVAL_PROMPT, REL_EVAL_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged model path or HuggingFace id")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter directory")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", default=None, help="the instruction the audio was generated from (rel only)")
    parser.add_argument("--dimension", default="rel", choices=["rel", "ovl"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    args = parser.parse_args()
    if args.dimension == "rel" and not args.text:
        parser.error("--text is required for --dimension rel")
    return args


def main():
    from swift.llm import InferRequest, RequestConfig, VllmEngine

    args = parse_args()
    prompt = REL_EVAL_PROMPT.format(instruction=args.text) if args.dimension == "rel" else OVL_EVAL_PROMPT
    engine_kwargs = {"adapters": [args.adapter], "max_lora_rank": args.max_lora_rank} if args.adapter else {}
    engine = VllmEngine(args.model, **engine_kwargs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        request = InferRequest(messages=[{
            "role": "user",
            "content": [
                {"type": "audio", "audio": ensure_16k(args.audio, os.path.join(tmp_dir, "clip.wav"))},
                {"type": "text", "text": prompt},
            ],
        }])
        response = engine.infer([request], RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0))

    print(response[0].choices[0].message.content)


if __name__ == "__main__":
    main()
