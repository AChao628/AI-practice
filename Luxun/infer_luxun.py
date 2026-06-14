"""
加载鲁迅文风 LoRA 并对话推理（底座：本地 CodeLlama-13B）。

用法:
  python Luxun/infer_luxun.py --lora ./Luxun/lora_luxun
  python Luxun/infer_luxun.py --lora ./Luxun/lora_luxun --question "星期一不想去上班，怎么办？"
"""
import argparse
import os
import re
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "codellama/CodeLlama-13b-hf"
DEFAULT_LORA = "./Luxun/lora_luxun"
_MIN_LOAD_VRAM_GB = 8.0

_SOCKS_PROXY_KEYS = (
    "ALL_PROXY", "all_proxy",
    "SOCKS_PROXY", "socks_proxy",
    "SOCKS5_PROXY", "socks5_proxy",
)


def setup_hf_env(allow_download: bool) -> bool:
    for key in _SOCKS_PROXY_KEYS:
        val = os.environ.get(key, "")
        if val.lower().startswith("socks"):
            os.environ.pop(key, None)
    if allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    return True


def _gpu_process_hint() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return f"当前占用 GPU 的进程:\n{out}\n可用 kill <PID> 释放显存。\n"
    except (OSError, subprocess.CalledProcessError):
        pass
    return ""


def build_bnb_config(cpu_offload: bool) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )


def build_model_load_kwargs(cpu_offload: bool, local_files_only: bool) -> dict:
    kwargs = {"local_files_only": local_files_only}
    if not torch.cuda.is_available():
        return kwargs

    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"GPU 可用显存: {free_gb:.1f} / {total_gb:.1f} GB")

    if cpu_offload:
        print("已启用 CPU offload（推理会明显变慢）")
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{max(1, int(free_gb))}GiB", "cpu": "48GiB"}
        return kwargs

    if free_gb < _MIN_LOAD_VRAM_GB:
        print(
            f"\n错误: 显存不足（仅剩 {free_gb:.1f} GB，加载 13B 4bit 约需 {_MIN_LOAD_VRAM_GB:.0f} GB）。\n"
            f"{_gpu_process_hint()}"
            "释放显存后重试:\n"
            "  nvidia-smi\n"
            "  kill <PID>\n"
            "或勉强使用 CPU offload:\n"
            "  python Luxun/infer_luxun.py --cpu_offload\n",
            file=sys.stderr,
        )
        sys.exit(1)

    kwargs["device_map"] = {"": 0}
    return kwargs


def build_prompt(tokenizer, question: str) -> tuple[dict, int]:
    """无 system 提示词；CodeLlama 走 ### 指令 / ### 回复 格式。"""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {"input_ids": prompt_ids}
        if tokenizer.pad_token_id is not None:
            inputs["attention_mask"] = (prompt_ids != tokenizer.pad_token_id).long()
        else:
            inputs["attention_mask"] = torch.ones_like(prompt_ids)
        return inputs, prompt_ids.shape[-1]

    text = f"### 指令: {question}\n### 回复:"
    inputs = tokenizer(text, return_tensors="pt")
    return inputs, inputs["input_ids"].shape[-1]


def clean_answer(text: str) -> str:
    """去掉解码失败的替换字符；去掉末尾未完结的半句碎片。"""
    text = text.replace("\ufffd", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = text.strip()
    # 若以「终究」「大抵」等悬在半空结尾，且无句号，视为不完整，保留原文供用户判断
    if text and text[-1] not in "。！？；…":
        for end in ("。", "！", "？", "；", "…"):
            idx = text.rfind(end)
            if idx != -1:
                text = text[: idx + 1]
                break
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--lora", type=str, default=DEFAULT_LORA)
    parser.add_argument("--question", type=str, default="你今天心情怎么样？")
    parser.add_argument("--max_new_tokens", type=int, default=220,
                        help="最大生成长度；训练集 assistant 约 139~197 token")
    parser.add_argument("--min_new_tokens", type=int, default=120,
                        help="最少生成 token，防止过早输出 eos 导致半截话")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do_sample", action="store_true", help="开启采样；默认贪心解码更稳")
    parser.add_argument("--cpu_offload", action="store_true", help="显存不足时 offload 到 CPU（很慢）")
    parser.add_argument("--allow_download", action="store_true")
    args = parser.parse_args()

    local_files_only = setup_hf_env(args.allow_download)

    bnb_config = build_bnb_config(args.cpu_offload)
    load_kwargs = build_model_load_kwargs(args.cpu_offload, local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        **load_kwargs,
    )
    model = PeftModel.from_pretrained(model, args.lora, local_files_only=local_files_only)
    model.eval()

    inputs, prompt_len = build_prompt(tokenizer, args.question)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.9)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    answer = tokenizer.decode(
        output_ids[0, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = clean_answer(answer)

    print(f"\n问: {args.question}")
    print(f"\n答: {answer}")


if __name__ == "__main__":
    main()
