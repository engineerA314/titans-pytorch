#!/usr/bin/env python3
"""
End-to-end training benchmark: manual_grad=True vs False

Runs actual training steps on enwik8 with TitansLMM model,
measuring wall-clock tok/sec and peak GPU memory.

Usage:
    python bench_e2e_training.py --model 0.5B --n-steps 50
    python bench_e2e_training.py --model all
"""

import argparse
import gc
import gzip
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from titans_pytorch import TitansLMM, MemoryMLP


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
CONFIGS = {
    '0.5B': {
        'dim': 896, 'depth': 4, 'heads': 14, 'dim_head': 64,
        'batch_size': 1, 'seq_len': 512,
    },
    '3B': {
        'dim': 2048, 'depth': 2, 'heads': 32, 'dim_head': 64,
        'batch_size': 1, 'seq_len': 256,
    },
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ByteDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start:start + self.seq_len + 1].long()


def load_enwik8(path):
    with gzip.open(path) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(cfg, use_manual_grad, device):
    neural_memory_model = MemoryMLP(
        dim=cfg['dim_head'],
        depth=2,
    )

    model = TitansLMM(
        num_tokens=256,
        dim=cfg['dim'],
        depth=cfg['depth'],
        num_persist_mem_tokens=4,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=cfg['dim_head'],
            heads=cfg['heads'],
            qk_rmsnorm=True,
            momentum=True,
            momentum_order=1,
            default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False,
            use_manual_grad=use_manual_grad,
        ),
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Benchmark one config
# ---------------------------------------------------------------------------
def benchmark_one(cfg_name, use_manual_grad, data_path, n_warmup, n_steps, device):
    cfg = CONFIGS[cfg_name]
    label = 'manual_grad' if use_manual_grad else 'vmap(grad)'
    dtype = torch.bfloat16

    try:
        model = build_model(cfg, use_manual_grad, device)
        num_params = sum(p.numel() for p in model.parameters())

        # Data
        raw = load_enwik8(data_path)
        ds = ByteDataset(raw, cfg['seq_len'])
        loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                            num_workers=0, drop_last=True)
        data_iter = iter(loader)

        optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        model.train()

        def get_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                return next(data_iter)

        # Warmup
        for _ in range(n_warmup):
            batch = get_batch().to(device)
            with torch.amp.autocast('cuda', dtype=dtype):
                loss = model(batch, return_loss=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

        torch.cuda.synchronize()

        # Measure peak memory
        reset_memory()
        batch = get_batch().to(device)
        with torch.amp.autocast('cuda', dtype=dtype):
            loss = model(batch, return_loss=True)
        loss.backward()
        optim.step()
        optim.zero_grad()
        peak_mem = get_peak_gb()

        # Measure throughput
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        total_tokens = 0
        for step in range(n_steps):
            batch = get_batch().to(device)
            with torch.amp.autocast('cuda', dtype=dtype):
                loss = model(batch, return_loss=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()
            total_tokens += cfg['batch_size'] * cfg['seq_len']

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        tok_per_sec = total_tokens / elapsed
        ms_per_step = elapsed / n_steps * 1000

        del model, optim, loader, ds, raw
        reset_memory()

        return {
            'config': cfg_name,
            'backend': label,
            'num_params': num_params,
            'peak_mem_gb': round(peak_mem, 3),
            'tok_per_sec': round(tok_per_sec),
            'ms_per_step': round(ms_per_step, 1),
            'n_steps': n_steps,
            'oom': False,
        }

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            reset_memory()
            return {
                'config': cfg_name, 'backend': label,
                'num_params': 0, 'peak_mem_gb': None,
                'tok_per_sec': None, 'ms_per_step': None,
                'n_steps': n_steps, 'oom': True,
            }
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='E2E Training Benchmark: manual_grad')
    parser.add_argument('--model', choices=['0.5B', '3B', 'all'], default='all')
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--n-steps', type=int, default=50)
    parser.add_argument('--data-path', type=str, default='./data/enwik8.gz')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__),
                                   'bench_e2e_training_results.json')

    if not torch.cuda.is_available():
        print("ERROR: CUDA required."); sys.exit(1)

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    models = ['0.5B', '3B'] if args.model == 'all' else [args.model]

    print("=" * 80)
    print("     Titans-PyTorch: End-to-End Training Benchmark")
    print("=" * 80)
    print(f"  GPU         : {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Precision   : bfloat16 (autocast)")
    print(f"  Warmup      : {args.n_warmup} steps")
    print(f"  Measurement : {args.n_steps} steps")
    print("=" * 80)

    all_results = []

    for model_name in models:
        cfg = CONFIGS[model_name]
        toks_per_step = cfg['batch_size'] * cfg['seq_len']
        print(f"\n{'─'*72}")
        print(f"  Config: {model_name}  |  dim={cfg['dim']}, depth={cfg['depth']}, "
              f"heads={cfg['heads']}")
        print(f"  Batch={cfg['batch_size']}, Seq={cfg['seq_len']}  "
              f"|  Tokens/step = {toks_per_step:,}")
        print(f"{'─'*72}\n")

        pair = []
        for use_manual in [False, True]:
            label = 'manual_grad' if use_manual else 'vmap(grad)'
            print(f"  [{label}] training {args.n_steps} steps ...", end=' ', flush=True)
            r = benchmark_one(model_name, use_manual, args.data_path,
                              args.n_warmup, args.n_steps, device)
            pair.append(r)
            all_results.append(r)

            if r['oom']:
                print("OOM!")
            else:
                print(f"done  (peak={r['peak_mem_gb']:.3f} GB, "
                      f"{r['tok_per_sec']:,} tok/s, "
                      f"{r['ms_per_step']:.0f} ms/step)")

        # Table
        print(f"\n  {'Backend':<16} | {'Params':>10} | {'Peak Mem':>10} | "
              f"{'ms/step':>10} | {'tok/sec':>12}")
        print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
        for r in pair:
            if r['oom']:
                print(f"  {r['backend']:<16} | {'':>10} | {'OOM':>10} | "
                      f"{'OOM':>10} | {'OOM':>12}")
            else:
                pstr = f"{r['num_params']/1e6:.1f}M"
                print(f"  {r['backend']:<16} | {pstr:>10} | {r['peak_mem_gb']:>7.3f} GB | "
                      f"{r['ms_per_step']:>7.0f} ms | {r['tok_per_sec']:>12,}")

        # Comparison
        v, m = pair
        if not v['oom'] and not m['oom']:
            mem_pct = (1 - m['peak_mem_gb'] / v['peak_mem_gb']) * 100
            speed_pct = (m['tok_per_sec'] / v['tok_per_sec'] - 1) * 100
            print(f"\n  >> E2E Improvement (manual_grad over vmap):")
            print(f"     Peak memory : {v['peak_mem_gb']:.3f} -> "
                  f"{m['peak_mem_gb']:.3f} GB  ({mem_pct:+.1f}%)")
            print(f"     Throughput  : {v['tok_per_sec']:,} -> "
                  f"{m['tok_per_sec']:,} tok/s  ({speed_pct:+.1f}%)")

    # Save
    output = {
        'benchmark': 'titans_e2e_training',
        'gpu': gpu_name, 'gpu_mem_gb': round(gpu_mem, 1),
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
