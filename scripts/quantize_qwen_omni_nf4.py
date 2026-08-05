"""Quantize a local Qwen2.5-Omni checkpoint to a text-output NF4 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-memory", default="6000MiB")
    parser.add_argument("--cpu-memory", default="14000MiB")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--prompt", default="请用一句话回答: 你是谁?")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise SystemExit(f"invalid source checkpoint: {source}")
    if source == output:
        raise SystemExit("output must not overwrite the source checkpoint")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if args.max_new_tokens < 1:
        raise SystemExit("max-new-tokens must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        Qwen2_5OmniConfig,
        Qwen2_5OmniForConditionalGeneration,
    )

    cuda_available = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if cuda_available else torch.float32
    config = Qwen2_5OmniConfig.from_pretrained(source, local_files_only=True)
    config.enable_audio_output = False
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    device_map: str | dict[str, str] = "auto" if cuda_available else {"": "cpu"}
    max_memory: dict[int | str, str] | None = None
    if cuda_available:
        max_memory = {0: args.gpu_memory, "cpu": args.cpu_memory}

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.",
        dir=output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        offload = temporary / "offload"
        offload.mkdir()
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            source,
            config=config,
            quantization_config=quantization,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)

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
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)
        generated = model.generate(
            **inputs,
            generation_mode="text",
            thinker_max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        generated_tokens = generated[:, inputs.input_ids.shape[1] :]
        answer = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        checkpoint = temporary / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(checkpoint)
        # The speaker dictionary is loaded separately by Qwen's model loader and
        # is not included by save_pretrained(). Keep it even though this artifact
        # disables generated audio so the checkpoint remains self-contained.
        speaker_dictionary = source / "spk_dict.pt"
        if speaker_dictionary.is_file():
            shutil.copy2(speaker_dictionary, checkpoint / speaker_dictionary.name)
        # Preserve preprocessing metadata for future multimodal-input support.
        # The text smoke test deliberately uses only the tokenizer, so Pillow and
        # torchvision are not required just to prove the quantized model answers.
        preprocessor_config = source / "preprocessor_config.json"
        if preprocessor_config.is_file():
            shutil.copy2(preprocessor_config, checkpoint / preprocessor_config.name)
        verification = {
            "answer": answer,
            "compute_dtype": str(compute_dtype).removeprefix("torch."),
            "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
            "quantization": "bitsandbytes-nf4-double-quant",
            "source": str(source),
            "text_output_only": True,
        }
        (checkpoint / "safejudge-quantization.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(checkpoint, output)

    print(json.dumps({"output": str(output), **verification}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
