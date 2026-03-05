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

PEAK_LR = 0.003
WARMUP_STEPS = 256
WARMDOWN_STEPS = 1024
EMA_DECAY = 0.998
INTERNAL_BATCH_SIZE = 32
PADDED_VOCAB = 50304
INTERMEDIATE_VAL_BATCHES = 4
LOG_EVERY = 128


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

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
        x = F.gelu(x, approximate='tanh')
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
        self.lm_head.weight = self.transformer.wte.weight

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
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
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

    B_orig = args.batch_size
    T = args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}

    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print(f"using device: {device}")

    torch.set_float32_matmul_precision('high')

    assert args.grad_accumulation_steps % ddp_world_size == 0
    orig_grad_accum_per_gpu = args.grad_accumulation_steps // ddp_world_size
    total_tokens_per_step = B_orig * orig_grad_accum_per_gpu * T * ddp_world_size

    B = INTERNAL_BATCH_SIZE
    new_grad_accum = (total_tokens_per_step) // (B * T * ddp_world_size)
    if new_grad_accum < 1:
        new_grad_accum = 1
        B = total_tokens_per_step // (T * ddp_world_size * new_grad_accum)
    grad_accumulation_steps = new_grad_accum

    tokens_per_iter = B * T * ddp_world_size * grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,} (B={B}, grad_accum={grad_accumulation_steps})")

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val
    val_loader = DistributedDataLoader(args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    num_vocab = PADDED_VOCAB
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
        learning_rate=PEAK_LR,
        betas=(0.9, 0.95),
        device_type=device,
        fused=True,
    )

    def get_lr(it):
        assert it <= args.num_iterations
        if it < WARMUP_STEPS:
            return PEAK_LR * (it + 1) / WARMUP_STEPS
        elif it < args.num_iterations - WARMDOWN_STEPS:
            return PEAK_LR
        else:
            decay_ratio = (args.num_iterations - it) / WARMDOWN_STEPS
            return PEAK_LR * decay_ratio

    ema_params = [p.data.clone() for p in raw_model.parameters()]

    def update_ema():
        with torch.no_grad():
            for ema_p, p in zip(ema_params, raw_model.parameters()):
                ema_p.lerp_(p.data, 1.0 - EMA_DECAY)

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

    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()

            if last_step:
                orig_params = [p.data.clone() for p in raw_model.parameters()]
                for ema_p, p in zip(ema_params, raw_model.parameters()):
                    p.data.copy_(ema_p)

                with torch.no_grad():
                    val_loss = 0.0
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss += loss
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    val_loss /= val_steps

                for orig_p, p in zip(orig_params, raw_model.parameters()):
                    p.data.copy_(orig_p)
                val_loader.reset()
                with torch.no_grad():
                    val_loss_raw = 0.0
                    for _ in range(val_steps):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss_raw += loss
                    dist.all_reduce(val_loss_raw, op=dist.ReduceOp.AVG)
                    val_loss_raw /= val_steps

                if val_loss_raw < val_loss:
                    val_loss = val_loss_raw

                del orig_params
            else:
                num_val_batches = INTERMEDIATE_VAL_BATCHES
                with torch.no_grad():
                    val_loss = 0.0
                    for _ in range(num_val_batches):
                        x_val, y_val = val_loader.next_batch()
                        _, loss = model(x_val, y_val, return_logits=False)
                        val_loss += loss
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    val_loss /= num_val_batches

            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                if logfile is not None:
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

        update_ema()

        if step % LOG_EVERY == 0 or step < 2:
            torch.cuda.synchronize()
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(
                f"step:{step}/{args.num_iterations} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms"
            )
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d trn:0\n" % step)

    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if SAVE_CHECKPOINTS and master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    destroy_process_group()
