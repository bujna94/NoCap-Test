import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

SAVE_CHECKPOINTS = os.environ.get("SAVE_CHECKPOINTS", "").lower() in {"1", "true", "yes"}

with open(sys.argv[0]) as f:
    code = f.read()

USE_LR = 0.00367
WARMUP = 256
WARMDOWN = 1024
EMA_DECAY = 0.998
EMA_EVERY = 4
MICRO_BATCH = 64
VOCAB_PADDED = 50304
BATCH_SCALE = 1.5
INTERMEDIATE_VAL_BATCHES = 1
VAL_EVERY_OVERRIDE = 1024
LOG_EVERY = 512

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_cached[None, :seq_len, None, :], self.sin_cached[None, :seq_len, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


def fast_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, max_seq_len=1024)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = fast_gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.vocab_size, config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, fused=False):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=fused
        )
        return optimizer


def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


VAL_TOKENS = 1_048_576


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}")

    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bin", type=str, default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--input_val_bin", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model", type=str, default="d12")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--warmdown_iters", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_loss_every", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_wandb", action="store_true")
    args = parser.parse_args()

    B = MICRO_BATCH
    T = args.sequence_length

    assert args.model in {"d12", "d24", "d36", "d48"}
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    orig_total_grad_accum = args.grad_accumulation_steps
    assert orig_total_grad_accum % ddp_world_size == 0
    orig_per_gpu_accum = orig_total_grad_accum // ddp_world_size
    orig_batch = args.batch_size

    tokens_per_gpu_orig = orig_batch * orig_per_gpu_accum * T
    tokens_per_gpu_new = int(tokens_per_gpu_orig * BATCH_SCALE)

    new_grad_accum = tokens_per_gpu_new // (B * T)
    if new_grad_accum < 1:
        new_grad_accum = 1

    actual_tokens_per_step = B * T * new_grad_accum * ddp_world_size
    expected_tokens_per_step = orig_batch * orig_total_grad_accum * T
    scaled_tokens_per_step = int(expected_tokens_per_step * BATCH_SCALE)

    grad_accumulation_steps = new_grad_accum

    # Scale num_iterations down proportionally to maintain same total tokens
    total_tokens_target = expected_tokens_per_step * args.num_iterations
    new_num_iterations = total_tokens_target // actual_tokens_per_step
    # Ensure we don't lose tokens
    if new_num_iterations * actual_tokens_per_step < total_tokens_target:
        new_num_iterations += 1
    num_iterations = new_num_iterations

    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print0(f"using device: {device}")
    print0(f"Micro-batch: {B}, Grad accum per GPU: {grad_accumulation_steps}")
    print0(f"Tokens per step: {actual_tokens_per_step:,} (orig: {expected_tokens_per_step:,}, scale: {BATCH_SCALE})")
    print0(f"Num iterations: {num_iterations} (orig: {args.num_iterations})")
    print0(f"Total tokens: {num_iterations * actual_tokens_per_step:,} (target: {total_tokens_target:,})")

    tokens_per_iter = actual_tokens_per_step
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val
    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )
    x, y = train_loader.next_batch()

    num_vocab = VOCAB_PADDED
    model_config = {
        "d12": GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(vocab_size=num_vocab, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=num_vocab, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=num_vocab, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True
    print0("compiling the model...")
    model = torch.compile(model)

    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=USE_LR,
        betas=(0.9, 0.95),
        device_type=device,
        fused=True,
    )

    # Adjust warmup/warmdown for new iteration count
    warmup_iters = min(WARMUP, num_iterations // 5)
    warmdown_iters = min(WARMDOWN, num_iterations // 3)

    def get_lr(it):
        assert it <= num_iterations
        if it < warmup_iters:
            return USE_LR * (it + 1) / warmup_iters
        elif it < num_iterations - warmdown_iters:
            return USE_LR
        else:
            decay_ratio = (num_iterations - it) / warmdown_iters
            return USE_LR * decay_ratio

    ema_params = [p.data.clone() for p in raw_model.parameters()]

    run_id = str(uuid.uuid4())

    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        with open(logfile, "w") as f:
            pass

    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    val_loss_every = VAL_EVERY_OVERRIDE if args.val_loss_every > 0 else 0

    for step in range(num_iterations + 1):
        last_step = step == num_iterations

        if val_loss_every > 0 and (step % val_loss_every == 0 or last_step):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()

            is_final = last_step

            if is_final:
                val_loader.reset()
                with torch.no_grad():
                    val_loss_raw = 0.0
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss_raw += loss
                    dist.all_reduce(val_loss_raw, op=dist.ReduceOp.AVG)
                    val_loss_raw /= val_steps

                orig_params = []
                for p, ema_p in zip(raw_model.parameters(), ema_params):
                    orig_params.append(p.data)
                    p.data = ema_p

                val_loader.reset()
                with torch.no_grad():
                    val_loss_ema = 0.0
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss_ema += loss
                    dist.all_reduce(val_loss_ema, op=dist.ReduceOp.AVG)
                    val_loss_ema /= val_steps

                for p, orig_p in zip(raw_model.parameters(), orig_params):
                    p.data = orig_p

                val_loss = min(val_loss_raw.item(), val_loss_ema.item())
                print0(f"step:{step}/{num_iterations} | val loss raw {val_loss_raw:.6f} | val loss ema {val_loss_ema:.6f} | best {val_loss:.6f}")
                if master_process and logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))
            else:
                val_loader.reset()
                num_val = INTERMEDIATE_VAL_BATCHES
                with torch.no_grad():
                    val_loss = 0.0
                    for _ in range(num_val):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss += loss
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    val_loss /= num_val
                print0(f"step:{step}/{num_iterations} | val loss {val_loss:.6f} (approx)")
                if master_process and logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        model.train()
        for micro_step in range(grad_accumulation_steps):
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / grad_accumulation_steps
            x, y = train_loader.next_batch()
            loss.backward()

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % EMA_EVERY == 0:
            with torch.no_grad():
                for p, ema_p in zip(raw_model.parameters(), ema_params):
                    ema_p.lerp_(p.data, 1.0 - EMA_DECAY ** EMA_EVERY)

        if step % LOG_EVERY == 0 or step < 5:
            torch.cuda.synchronize()
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(
                f"step:{step}/{num_iterations} | lr {lr:.6f} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms"
            )

    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    destroy_process_group()
