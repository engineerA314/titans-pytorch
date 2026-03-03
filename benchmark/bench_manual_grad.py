#!/usr/bin/env python3
"""
Benchmark: Manual Gradient vs vmap(grad()) in Titans NeuralMemory

Measures peak GPU memory and throughput for the NeuralMemory module
with use_manual_grad=True vs False, for 0.5B and 3B model configurations.

The manual gradient optimization replaces PyTorch's vmap(grad()) with
hand-written batched matmul chain-rule (torch.bmm), avoiding per-sample
computation graph overhead.

Usage:
    python bench_manual_grad.py                          # all configs
    python bench_manual_grad.py --model 0.5B             # 0.5B only
    python bench_manual_grad.py --model 3B --seq-len 512 # 3B, seq=512
    python bench_manual_grad.py --n-iters 50             # more iterations

Run on A100 80GB for best results.
"""

import argparse
import json
import gc
import sys
import os
import time
from datetime import datetime

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from titans_pytorch.neural_memory import NeuralMemory


# ---------------------------------------------------------------------------
# Model configurations matching Qwen2 family
# ---------------------------------------------------------------------------
CONFIGS = {
    '0.5B': {
        'dim': 896,
        'heads': 14,
        'dim_head': 64,
        'chunk_size': 64,
        'batch_size': 4,
        'model_kwargs': {'depth': 2, 'expansion_factor': 4.},
    },
    '3B': {
        'dim': 2048,
        'heads': 32,
        'dim_head': 64,
        'chunk_size': 64,
        'batch_size': 2,
        'model_kwargs': {'depth': 2, 'expansion_factor': 4.},
    },
}


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------
def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def benchmark_one(config_name, seq_len, use_manual_grad, n_warmup=5, n_iters=20):
    """Run one benchmark configuration. Returns dict of results."""
    cfg = CONFIGS[config_name]
    device = 'cuda'
    dtype = torch.bfloat16

    label = 'manual_grad' if use_manual_grad else 'vmap(grad)'

    try:
        # Create NeuralMemory module
        mem = NeuralMemory(
            dim=cfg['dim'],
            heads=cfg['heads'],
            dim_head=cfg['dim_head'],
            chunk_size=cfg['chunk_size'],
            use_manual_grad=use_manual_grad,
            momentum=True,
            momentum_order=1,
            qk_rmsnorm=True,
            pre_rmsnorm=True,
            use_accelerated_scan=False,
            default_model_kwargs=cfg['model_kwargs'],
        ).to(device).to(dtype)

        B = cfg['batch_size']
        T = seq_len
        D = cfg['dim']

        # --- Warmup ---
        for _ in range(n_warmup):
            x = torch.randn(B, T, D, device=device, dtype=dtype)
            out, _state = mem(x)
            loss = out.sum()
            loss.backward()
            mem.zero_grad()

        torch.cuda.synchronize()

        # --- Measure peak memory (single fwd+bwd) ---
        reset_memory()

        x = torch.randn(B, T, D, device=device, dtype=dtype)
        out, _state = mem(x)
        loss = out.sum()
        loss.backward()

        peak_mem = get_peak_memory_gb()
        mem.zero_grad()

        # --- Measure timing ---
        fwd_times = []
        bwd_times = []

        for _ in range(n_iters):
            x = torch.randn(B, T, D, device=device, dtype=dtype)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, _state = mem(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss = out.sum()

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            fwd_times.append(t1 - t0)
            bwd_times.append(t3 - t2)

            mem.zero_grad()

        fwd_ms = sum(fwd_times) / len(fwd_times) * 1000
        bwd_ms = sum(bwd_times) / len(bwd_times) * 1000
        total_ms = fwd_ms + bwd_ms
        tok_per_sec = (B * T) / (total_ms / 1000)

        # Cleanup
        del mem, x, out, _state, loss
        reset_memory()

        return {
            'config': config_name,
            'backend': label,
            'seq_len': seq_len,
            'batch_size': B,
            'peak_mem_gb': round(peak_mem, 4),
            'fwd_ms': round(fwd_ms, 2),
            'bwd_ms': round(bwd_ms, 2),
            'total_ms': round(total_ms, 2),
            'tok_per_sec': round(tok_per_sec),
            'oom': False,
        }

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            reset_memory()
            return {
                'config': config_name,
                'backend': label,
                'seq_len': seq_len,
                'batch_size': cfg['batch_size'],
                'peak_mem_gb': None,
                'fwd_ms': None,
                'bwd_ms': None,
                'total_ms': None,
                'tok_per_sec': None,
                'oom': True,
            }
        raise


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def fmt(val, suffix='', width=10):
    if val is None:
        return f"{'OOM':>{width}}"
    if isinstance(val, float):
        return f"{val:>{width - len(suffix)}.3f}{suffix}"
    return f"{val:>{width - len(suffix)}}{suffix}"


def print_table(results_pair):
    """Print comparison table for a pair of (vmap, manual) results."""
    header = (f"  {'Backend':<16} | {'Peak Mem':>10} | {'Fwd':>10} | "
              f"{'Bwd':>10} | {'Total':>10} | {'tok/sec':>12}")
    sep = (f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-"
           f"{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    print(header)
    print(sep)

    for r in results_pair:
        if r['oom']:
            print(f"  {r['backend']:<16} | {'OOM':>10} | {'OOM':>10} | "
                  f"{'OOM':>10} | {'OOM':>10} | {'OOM':>12}")
        else:
            print(f"  {r['backend']:<16} | {r['peak_mem_gb']:>7.3f} GB | "
                  f"{r['fwd_ms']:>7.1f} ms | {r['bwd_ms']:>7.1f} ms | "
                  f"{r['total_ms']:>7.1f} ms | {r['tok_per_sec']:>12,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Titans NeuralMemory: manual_grad vs vmap(grad())')
    parser.add_argument('--model', choices=['0.5B', '3B', 'all'], default='all')
    parser.add_argument('--seq-len', type=int, nargs='+', default=[512, 2048])
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--n-iters', type=int, default=20)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__),
                                   'bench_manual_grad_results.json')

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    models = ['0.5B', '3B'] if args.model == 'all' else [args.model]

    print("=" * 80)
    print("     Titans-PyTorch: Manual Gradient vs vmap(grad()) Benchmark")
    print("=" * 80)
    print(f"  GPU         : {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Warmup      : {args.n_warmup} iters")
    print(f"  Measurement : {args.n_iters} iters")
    print("=" * 80)

    all_results = []

    for model_name in models:
        cfg = CONFIGS[model_name]
        for seq_len in args.seq_len:
            print(f"\n{'─'*72}")
            print(f"  Config: {model_name}  |  dim={cfg['dim']}, heads={cfg['heads']}, "
                  f"dim_head={cfg['dim_head']}, chunk={cfg['chunk_size']}")
            print(f"  Batch={cfg['batch_size']}, Seq={seq_len}  "
                  f"|  Tokens/step = {cfg['batch_size'] * seq_len:,}")
            print(f"{'─'*72}\n")

            # Run both variants
            pair = []
            for use_manual in [False, True]:
                label = 'manual_grad' if use_manual else 'vmap(grad)'
                print(f"  [{label}] running ...", end=' ', flush=True)
                r = benchmark_one(model_name, seq_len, use_manual,
                                  args.n_warmup, args.n_iters)
                pair.append(r)
                all_results.append(r)

                if r['oom']:
                    print("OOM!")
                else:
                    print(f"done  (peak={r['peak_mem_gb']:.3f} GB, "
                          f"total={r['total_ms']:.1f} ms)")

            print()
            print_table(pair)

            # Print improvement summary
            vmap_r, manual_r = pair
            if not vmap_r['oom'] and not manual_r['oom']:
                mem_delta = manual_r['peak_mem_gb'] - vmap_r['peak_mem_gb']
                mem_pct = (1 - manual_r['peak_mem_gb'] / vmap_r['peak_mem_gb']) * 100
                speed_pct = (manual_r['tok_per_sec'] / vmap_r['tok_per_sec'] - 1) * 100
                latency_pct = (1 - manual_r['total_ms'] / vmap_r['total_ms']) * 100

                print(f"\n  >> Improvement (manual_grad over vmap):")
                print(f"     Peak memory : {vmap_r['peak_mem_gb']:.3f} -> "
                      f"{manual_r['peak_mem_gb']:.3f} GB  ({mem_pct:+.1f}%)")
                print(f"     Throughput  : {vmap_r['tok_per_sec']:,} -> "
                      f"{manual_r['tok_per_sec']:,} tok/s  ({speed_pct:+.1f}%)")
                print(f"     Latency     : {vmap_r['total_ms']:.1f} -> "
                      f"{manual_r['total_ms']:.1f} ms  ({latency_pct:+.1f}%)")

    # Save JSON
    output = {
        'benchmark': 'titans_manual_grad',
        'gpu': gpu_name,
        'gpu_mem_gb': round(gpu_mem, 1),
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'configs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in CONFIGS.items()},
        'results': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
