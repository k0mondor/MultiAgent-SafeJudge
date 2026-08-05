"""Load a local Qwen2.5-Omni checkpoint and print one text response."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", default="请用一句话回答: 你是谁?")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = args.model.resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"invalid model checkpoint: {model_path}")
    if args.max_new_tokens < 1:
        raise SystemExit("max-new-tokens must be at least 1")

    import torch
    from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    device = (
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device == "cuda:0":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    load_started = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - load_started
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_started = time.perf_counter()
    generated = model.generate(
        **inputs,
        generation_mode="text",
        thinker_max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    if device == "cuda:0":
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    generated_tokens = generated[:, inputs.input_ids.shape[1] :]
    answer = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result: dict[str, object] = {
        "model": str(model_path),
        "device": str(model.device),
        "prompt": args.prompt,
        "answer": answer,
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "generated_tokens": generated_tokens.shape[1],
        "tokens_per_second": round(generated_tokens.shape[1] / generation_seconds, 3),
    }
    if device == "cuda:0":
        result["cuda_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024**2,
            1,
        )
        result["cuda_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024**2,
            1,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
