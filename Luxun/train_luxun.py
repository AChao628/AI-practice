"""
鲁迅文风 LoRA 微调（CodeLlama-13B + QLoRA）

数据格式：Luxun/train.jsonl，每行 {"messages": [system, user, assistant, ...]}

用法:
  # 默认使用本地缓存的 codellama/CodeLlama-13b-hf
  python Luxun/train_luxun.py

  # 推理测试
  HF_HUB_OFFLINE=1 python Luxun/infer_luxun.py --lora ./Luxun/lora_luxun
"""
import argparse
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 使用本地已缓存的 CodeLlama-13B）
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "codellama/CodeLlama-13b-hf"
DEFAULT_DATA = "Luxun/train.jsonl"       # 训练集路径
DEFAULT_OUTPUT = "./Luxun/lora_luxun"    # LoRA 权重输出目录

# httpx 不支持 socks:// 代理，会触发 "Unknown scheme for proxy URL"
_SOCKS_PROXY_KEYS = (
    "ALL_PROXY", "all_proxy",
    "SOCKS_PROXY", "socks_proxy",
    "SOCKS5_PROXY", "socks5_proxy",
)


# 13B 4bit 加载约需 8GB 显存（另需余量给训练激活）
_MIN_LOAD_VRAM_GB = 8.0


def setup_hf_env(allow_download: bool) -> bool:
    """
    配置 Hugging Face 加载环境。
    - 默认离线：强制 HF_HUB_OFFLINE=1，避免无缓存时误触网络
    - 清除 socks 代理：防止 httpx 报错（与 val.py 同类问题）
    返回 local_files_only 标志。
    """
    for key in _SOCKS_PROXY_KEYS:
        val = os.environ.get(key, "")
        if val.lower().startswith("socks"):
            os.environ.pop(key, None)

    if allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return False

    os.environ["HF_HUB_OFFLINE"] = "1"
    return True


def assert_model_cached(model_name: str) -> None:
    """离线模式下检查 config.json 是否在本地缓存，缺失则提前报错。"""
    cached = try_to_load_from_cache(model_name, "config.json")
    if cached is None:
        print(
            f"\n错误: 本地未找到模型 {model_name!r}。\n"
            "请确认 ~/.cache/huggingface/hub 中已有该模型，或去掉 socks 代理后下载:\n"
            "  unset all_proxy ALL_PROXY socks_proxy SOCKS_PROXY socks5_proxy SOCKS5_PROXY\n"
            "  python Luxun/train_luxun.py --allow_download\n",
            file=sys.stderr,
        )
        sys.exit(1)


def build_bnb_config(cpu_offload: bool) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )


def build_model_load_kwargs(cpu_offload: bool, local_files_only: bool) -> dict:
    """构建 from_pretrained 参数；默认整模上 GPU，避免 device_map=auto 误 offload 到 CPU。"""
    kwargs = {"local_files_only": local_files_only}
    if not torch.cuda.is_available():
        return kwargs

    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"GPU 可用显存: {free_gb:.1f} / {total_gb:.1f} GB")

    if cpu_offload:
        print("已启用 CPU offload（训练会明显变慢）")
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{max(1, int(free_gb))}GiB", "cpu": "48GiB"}
        return kwargs

    if free_gb < _MIN_LOAD_VRAM_GB:
        print(
            f"\n错误: 显存不足（仅剩 {free_gb:.1f} GB，加载 13B 4bit 约需 {_MIN_LOAD_VRAM_GB:.0f} GB）。\n"
            "请先结束占用 GPU 的其他进程:\n"
            "  nvidia-smi          # 查看 PID\n"
            "  kill <PID>          # 结束旧 python 进程\n"
            "或显存实在不够时启用 CPU offload:\n"
            "  python Luxun/train_luxun.py --cpu_offload\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # 固定到单卡，防止 auto 把层卸到 CPU 触发 bnb 报错
    kwargs["device_map"] = {"": 0}
    return kwargs


def _messages_to_instruction(messages):
    """将 messages 转为 ### 指令 / ### 回复 格式；忽略 system，仅保留 user 作指令。"""
    user = assistant = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    prompt = f"### 指令: {user}\n### 回复: "
    return prompt, assistant


def _drop_system_messages(messages):
    return [m for m in messages if m["role"] != "system"]


def _supports_chat_mask(tokenizer) -> bool:
    """判断 tokenizer 是否支持 assistant token 掩码（需内置 {% generation %} 模板）。"""
    if not getattr(tokenizer, "chat_template", None):
        return False
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        return True
    except Exception:
        return False


def preprocess_instruction_batch(examples, tokenizer, max_length=1024):
    """指令格式预处理：user 为 prompt，仅 assistant 回复参与 loss（不含 system）。"""
    prompts, responses, texts = [], [], []
    for messages in examples["messages"]:
        prompt, response = _messages_to_instruction(messages)
        prompts.append(prompt)
        responses.append(response)
        texts.append(prompt + response)

    batch = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )
    offset_mapping = batch.pop("offset_mapping")

    labels = []
    for i, input_ids in enumerate(batch["input_ids"]):
        attn = batch["attention_mask"][i]
        prompt_char_len = len(prompts[i])
        row_labels = list(input_ids)
        for j in range(len(row_labels)):
            if attn[j] == 0:
                row_labels[j] = -100
                continue
            span = offset_mapping[i][j]
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            cs = span[0]
            if cs is not None and cs < prompt_char_len:
                row_labels[j] = -100
        labels.append(row_labels)

    batch["labels"] = labels
    return batch


def preprocess_chat_batch(examples, tokenizer, max_length=1024):
    """
    将 messages 对话格式转为模型训练所需的 input_ids / labels。

    关键点：只对 assistant 的回复计算 loss。
    - system、user 部分 labels 设为 -100（system 在预处理时已剔除）
    - padding 位置同样设为 -100
    这样模型只学习"如何以鲁迅文风回答"，而不是复述用户问题。
    """
    # apply_chat_template 按模型内置模板拼接多轮对话
    # return_assistant_tokens_mask=True 会返回布尔掩码，标记哪些 token 属于 assistant
    batch = tokenizer.apply_chat_template(
        [_drop_system_messages(msgs) for msgs in examples["messages"]],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        max_length=max_length,
        truncation=True,          # 超长样本截断，避免 OOM
        padding="max_length",     # 批内对齐到同一长度
    )
    assistant_masks = batch.pop("assistant_masks")

    labels = []
    for input_ids, attn_mask, asst_mask in zip(
        batch["input_ids"], batch["attention_mask"], assistant_masks
    ):
        row_labels = []
        for tok_id, mask_bit, asst_bit in zip(input_ids, attn_mask, asst_mask):
            # mask_bit=0 → padding；asst_bit=False → 非 assistant 区间
            if mask_bit == 0 or not asst_bit:
                row_labels.append(-100)
            else:
                row_labels.append(tok_id)  # 仅 assistant token 参与 loss
        labels.append(row_labels)

    batch["labels"] = labels
    return batch


def main():
    # -----------------------------------------------------------------------
    # 命令行参数
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data_file", type=str, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="单条样本最大 token 数，鲁迅式长回复建议 1024")
    parser.add_argument("--num_train_epochs", type=int, default=15,
                        help="训练轮数；数据仅 49 条，需多训几轮")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="LoRA 常用学习率，比全量微调可略大")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每卡 batch size；13B 量化后通常只能设 1")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数；等效 batch = 1 × 8 = 8")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA 低秩矩阵的秩，越大表达能力越强、显存越多")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA 缩放系数，通常设为 r 的 2 倍")
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="显存不足时将部分层 offload 到 CPU（很慢，仅作兜底）",
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="允许从 Hugging Face Hub 下载；默认仅用本地缓存",
    )
    args = parser.parse_args()

    # 处理代理与离线模式（修复 socks:// 导致 httpx 崩溃）
    local_files_only = setup_hf_env(args.allow_download)
    if local_files_only:
        assert_model_cached(args.model_name)

    # -----------------------------------------------------------------------
    # 1. 4bit 量化配置（QLoRA 核心：把 13B 模型压到约 8GB 显存）
    # -----------------------------------------------------------------------
    bnb_config = build_bnb_config(args.cpu_offload)
    load_kwargs = build_model_load_kwargs(args.cpu_offload, local_files_only)

    # -----------------------------------------------------------------------
    # 2. 加载底座模型与分词器
    # -----------------------------------------------------------------------
    print(f"加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        **load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=local_files_only,
    )
    # Llama 系列默认无 pad_token，借用 eos_token 做 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # 训练时右侧 padding，与推理 left padding 不同

    # -----------------------------------------------------------------------
    # 3. 配置并注入 LoRA 适配器（只训练少量参数，底座权重冻结）
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # 与 train.off2.py 一致，在 Q/V 投影上插入 LoRA
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,              # 防止小数据集过拟合
        bias="none",                    # 不训练 bias
        task_type="CAUSAL_LM",          # 因果语言模型（自回归生成）
    )
    # 量化模型训练前需做 kbit 兼容处理（开启梯度检查点、关闭不必要缓存等）
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数占比（通常 < 1%）

    # -----------------------------------------------------------------------
    # 4. 加载并预处理训练数据
    # -----------------------------------------------------------------------
    print(f"加载数据: {args.data_file}")
    data = load_dataset("json", data_files={"train": args.data_file})

    use_chat = _supports_chat_mask(tokenizer)
    if use_chat:
        print("数据格式: chat template（messages + assistant mask）")
    else:
        print("数据格式: ### 指令 / ### 回复（当前模型无可用 chat 模板）")

    def tokenize_fn(examples):
        if use_chat:
            return preprocess_chat_batch(examples, tokenizer, max_length=args.max_length)
        return preprocess_instruction_batch(examples, tokenizer, max_length=args.max_length)

    tokenized = data["train"].map(
        tokenize_fn,
        batched=True,                                       # 批量处理加速
        remove_columns=data["train"].column_names,          # 去掉原始 messages 列
    )

    # -----------------------------------------------------------------------
    # 5. 训练超参数
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        print(
            "\n错误: 未检测到 CUDA GPU，13B 模型无法在 CPU 上训练。\n"
            "请确认: nvidia-smi 正常、PyTorch 为 CUDA 版、驱动已安装。\n",
            file=sys.stderr,
        )
        sys.exit(1)

    use_bf16 = torch.cuda.is_bf16_supported()
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = (len(tokenized) + effective_batch - 1) // effective_batch
    warmup_steps = max(1, int(steps_per_epoch * args.num_train_epochs * 0.05))
    print(f"GPU: {torch.cuda.get_device_name(0)}, bf16={use_bf16}, warmup_steps={warmup_steps}")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,          # 前 5% 步数线性升温学习率
        lr_scheduler_type="cosine",         # 余弦退火，后期缓慢收敛
        save_strategy="epoch",              # 每轮结束保存 checkpoint
        logging_steps=5,                    # 每 5 步打印一次 loss
        bf16=use_bf16,                      # Ampere+ GPU 用 bf16，否则退回 fp16
        fp16=not use_bf16,
        optim="paged_adamw_8bit",           # 8bit 分页优化器，进一步省显存
        report_to="none",                   # 不上报到 wandb 等
        remove_unused_columns=False,        # 保留自定义 labels 列
    )

    # -----------------------------------------------------------------------
    # 6. 启动训练
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
    )

    trainer.train()

    # -----------------------------------------------------------------------
    # 7. 保存 LoRA 权重与分词器（体积小，通常几十到几百 MB）
    # -----------------------------------------------------------------------
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA 已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
