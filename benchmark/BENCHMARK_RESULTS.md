# Titans-PyTorch: Manual Gradient Benchmark Results

## Overview

Benchmarking the **manual gradient optimization** (`use_manual_grad=True`) vs PyTorch's `vmap(grad())` in the `NeuralMemory` module. The manual gradient replaces PyTorch's per-sample computation graph with hand-written batched matmul chain-rule (`torch.bmm`), eliminating vmap overhead.

## Environment

| Item | Detail |
|------|--------|
| GPU | NVIDIA A100-SXM4-80GB |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Precision | bfloat16 |
| Date | 2026-02-24 |

## Model Configurations

| Config | dim | heads | dim_head | chunk_size | MLP depth | expansion |
|--------|-----|-------|----------|------------|-----------|-----------|
| 0.5B (Qwen2-0.5B) | 896 | 14 | 64 | 64 | 2 | 4x |
| 3B (Qwen2.5-3B) | 2048 | 32 | 64 | 64 | 2 | 4x |

## Results

### 0.5B Configuration

| Seq Len | Batch | Backend | Peak Mem (GB) | Fwd (ms) | Bwd (ms) | Total (ms) | tok/sec |
|---------|-------|---------|---------------|----------|----------|------------|---------|
| 512 | 4 | vmap(grad) | 0.630 | 15.1 | 15.0 | 30.1 | 68,071 |
| 512 | 4 | **manual_grad** | 0.700 | 14.2 | 16.6 | 30.9 | 66,379 |
| 2048 | 4 | vmap(grad) | 2.371 | 26.0 | 47.4 | 73.4 | 111,562 |
| 2048 | 4 | **manual_grad** | 2.657 | 21.5 | 39.3 | 60.8 | **134,664** |

### 3B Configuration

| Seq Len | Batch | Backend | Peak Mem (GB) | Fwd (ms) | Bwd (ms) | Total (ms) | tok/sec |
|---------|-------|---------|---------------|----------|----------|------------|---------|
| 512 | 2 | vmap(grad) | 0.728 | 15.5 | 15.8 | 31.3 | 32,762 |
| 512 | 2 | **manual_grad** | 0.809 | 13.6 | 14.8 | 28.5 | **35,973** |
| 2048 | 2 | vmap(grad) | 2.724 | 20.6 | 32.9 | 53.5 | 76,525 |
| 2048 | 2 | **manual_grad** | 3.053 | 18.8 | 34.0 | 52.9 | 77,480 |

### Improvement Summary

| Config | Seq Len | Throughput Change | Latency Change | Memory Change |
|--------|---------|-------------------|----------------|---------------|
| 0.5B | 512 | -2.5% | -2.5% | -11.1% |
| 0.5B | 2048 | **+20.7%** | **+17.2%** | -12.1% |
| 3B | 512 | **+9.8%** | **+8.9%** | -11.1% |
| 3B | 2048 | +1.2% | +1.2% | -12.1% |

## Analysis

1. **Throughput**: Manual gradient provides up to **+20.7% throughput improvement** at longer sequences (0.5B, seq=2048), where the grad computation becomes a larger fraction of the total step. At shorter sequences, the improvement is marginal.

2. **Memory**: Manual gradient uses ~12% **more** peak memory. This is because the manual backward caches all intermediate activations and pre-GELU values for the chain-rule, while vmap(grad()) can share some graph structure.

3. **Scaling**: The benefit is most pronounced when there are many chunks per sequence (seq=2048 / chunk=64 = 32 chunks), as each chunk requires a separate grad computation. At seq=512 (8 chunks), the overhead is less significant.

4. **Note**: This benchmark measures the isolated `NeuralMemory` cell. In a full Titans model (TitansLMM/MAC/MAL), the memory cell is one component among attention, feedforward, and embedding layers. The end-to-end training impact is smaller than the isolated cell improvement.

---

## End-to-End Training Results

Full model training on **enwik8** dataset with TitansLMM (embedding + NeuralMemory layers + LM head), AdamW optimizer (lr=1e-3), gradient clipping (max_norm=1.0), bfloat16 autocast. 50 training steps measured after 5 warmup steps.

### Model Configurations

| Config | dim | depth | heads | dim_head | batch_size | seq_len |
|--------|-----|-------|-------|----------|------------|---------|
| 0.5B | 896 | 4 | 14 | 64 | 1 | 512 |
| 3B | 2048 | 2 | 32 | 64 | 1 | 256 |

### Results

| Config | Backend | Params | Peak Mem (GB) | ms/step | tok/sec |
|--------|---------|--------|---------------|---------|---------|
| 0.5B | vmap(grad) | 40.2M | 32.328 | 400.3 | 1,279 |
| 0.5B | **manual_grad** | 40.2M | **30.408** | **394.4** | **1,298** |
| 3B | vmap(grad) | 103.4M | 20.477 | 227.6 | 1,125 |
| 3B | **manual_grad** | 103.4M | **19.493** | **223.4** | **1,146** |

### E2E Improvement Summary

| Config | Peak Memory | Throughput | Latency |
|--------|-------------|------------|---------|
| 0.5B | 32.3 -> 30.4 GB (**-5.9%**) | 1,279 -> 1,298 tok/s (**+1.5%**) | 400.3 -> 394.4 ms (**+1.5%**) |
| 3B | 20.5 -> 19.5 GB (**-4.8%**) | 1,125 -> 1,146 tok/s (**+1.9%**) | 227.6 -> 223.4 ms (**+1.8%**) |

### E2E Analysis

1. **Memory reduction: ~5% in full model.** The isolated cell shows ~12% more memory for manual_grad, but in e2e training the manual_grad actually uses **less** total memory because it avoids building vmap computation graphs that interact with autograd across the full model backward pass.

2. **Throughput: +1.5-1.9% improvement.** The grad computation is a small fraction of the total training step (which includes embedding, attention-free mixing, LM head, optimizer step), so the isolated 20% cell-level speedup translates to a modest but consistent e2e gain.

3. **Consistency**: Both configs show the same trend — manual_grad is strictly better in both memory and throughput at the e2e level, validating the optimization for production use.

4. **Note on batch size**: TitansLMM's per-token gradient computation is memory-intensive. Batch size was set to 1 to avoid OOM on a single A100 80GB. In multi-GPU training with model parallelism, larger batches become feasible.

## Reproduction

```bash
# Isolated cell benchmark
cd titans-pytorch
CUDA_VISIBLE_DEVICES=0 python3 benchmark/bench_manual_grad.py --model all --seq-len 512 2048

# End-to-end training benchmark (requires enwik8.gz in ./data/)
CUDA_VISIBLE_DEVICES=0 python3 benchmark/bench_e2e_training.py --model all --n-steps 50
```
