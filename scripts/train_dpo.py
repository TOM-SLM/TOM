from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
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
from dataset import DPOSampler

warnings.filterwarnings('ignore')


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, mask, beta):
    print("ref_probs:", ref_probs)
    print("probs:", probs)
    print("mask:", mask)
    print("beta:", beta)
    # ref_probs 和 probs 都是 shape: (batch_size, seq_len)
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    print("logits:", logits)
    loss = -F.logsigmoid(beta * logits)
    print("loss:", loss)
    return loss.mean()


def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.no_grad():
            ref_outputs = ref_model(x)
            ref_logits = ref_outputs.logits
            
        with torch.autocast(device_type="cuda", dtype=getattr(torch, args.dtype)):
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            outputs = model(x)

        probs = logits_to_probs(outputs.logits, y)
        probs = probs * mask
        loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
        return print("loss:", loss)
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
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if wandb is not None:
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0:
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.out_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'

            state_dict = model.state_dict()
            state_dict = {k.replace("model._orig_mod.", "model."): v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('model')
    model = TOMForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.out_dir}/sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 初始化参考模型
    ref_model = TOMForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    ckp = f'{args.out_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'
    if os.path.exists(ckp):
        print(f'加载已有rlhf模型参数: {ckp}')
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

    print(f'LLM总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


def init_dataset(tokenizer):
    train_ds = load_dataset("json", data_files=args.data_path, split="all")
    train_ds = train_ds.map(
        DPOSampler(tokenizer, max_length=args.max_seq_len),
        batched=False,
        num_proc=os.cpu_count(),
    )
    train_ds.set_format(type="torch", columns=['x_chosen', 'y_chosen', 'mask_chosen', 'x_rejected', 'y_rejected', 'mask_rejected'])
    print(f"训练数据量: {len(train_ds) / 1024 / 1024:.2f}M")
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOM RLHF")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning-rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="TOM-RLHF-SFT")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument('--hidden-size', default=512, type=int)
    parser.add_argument('--num-hidden-layers', default=8, type=int)
    parser.add_argument('--max-seq-len', default=1024, type=int)
    parser.add_argument('--use-moe', default=False, type=bool)
    parser.add_argument("--data-path", type=str, default="dataset/dpo.jsonl")
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--use-compile", action="store_true")

    args = parser.parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.use_wandb:
        import wandb

        args.wandb_run_name = f"TOM-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    
    lm_config = TOMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )

    model, ref_model, tokenizer = init_model(lm_config)
    if args.use_compile:
        model.model = torch.compile(model.model)

    train_loader = init_dataset(tokenizer)

    scaler = torch.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    print(f"训练开始时间: {datetime.now()}")
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
    print(f"训练结束时间: {datetime.now()}")
