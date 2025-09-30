from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from datetime import datetime
from torch import optim, nn
import argparse
import warnings
import random
import torch
import time
import math
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_base import TOMConfig, TOMForCausalLM
from dataset import PretrainSampler, PretrainDataset

warnings.filterwarnings("ignore")


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, item in enumerate(train_loader):
        X = item["X"].to(args.device)
        Y = item["Y"].to(args.device)
        loss_mask = item["loss_mask"].to(args.device)

        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=getattr(torch, args.dtype)):
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )

        loss = (loss * loss_mask).sum() / loss_mask.sum()
        loss += res.aux_loss
        loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            print(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

            if wandb is not None:
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60
                        - spend_time // 60,
                    }
                )

        if (step + 1) % args.save_interval == 0:
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.out_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth"

            state_dict = model.state_dict()
            state_dict = {k.replace("model._orig_mod.", "model."): v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = TOMForCausalLM(lm_config).to(args.device)
    print(
        f"LLM训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
    )
    return model, tokenizer


def init_dataset(tokenizer):
    train_ds = load_dataset("json", data_files=args.data_path, split="all")
    train_ds = train_ds.map(
        PretrainSampler(tokenizer, max_length=args.max_seq_len),
        batched=False,
        num_proc=os.cpu_count(),
    )
    train_ds.set_format(type="torch", columns=["X", "Y", "loss_mask"])
    print(f"训练数据量: {len(train_ds) / 1024 / 1024:.2f}M")
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOM Pretraining")
    parser.add_argument("--out-dir", type=str, default="output")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="TOM-Pretrain")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-hidden-layers", default=8, type=int)
    parser.add_argument("--max-seq-len", default=512, type=int)
    parser.add_argument("--use-moe", default=False, type=bool)
    parser.add_argument("--data-path", type=str, default="dataset/pretrain_hq.jsonl")
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--use-compile", action="store_true")
    args = parser.parse_args()

    lm_config = TOMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"TOM-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    if args.use_compile and "cuda" in args.device:
        model.model = torch.compile(model.model)

    train_loader = init_dataset(tokenizer)

    scaler = torch.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    print(f"开始训练时间: {datetime.now()}")
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
    print(f"结束训练时间: {datetime.now()}")
