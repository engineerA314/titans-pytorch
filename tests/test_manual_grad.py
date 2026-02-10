"""
TDD tests for manual batched matmul gradient computation.
Replaces vmap(grad(forward_and_loss)) with torch.bmm chain-rule.
"""

import math
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, vmap, grad

from titans_pytorch.memory_models import MemoryMLP, ResidualNorm


# ============================================================================
# Reference: vmap(grad()) implementation (ground truth)
# ============================================================================

def make_vmap_grad_fn(model, loss_fn):
    """Create the reference vmap(grad(forward_and_loss)) function."""
    def forward_and_loss(params, inputs, loss_weights, target):
        pred = functional_call(model, params, inputs)
        losses = loss_fn(pred, target)
        return (losses * loss_weights).sum(), losses

    return vmap(grad(forward_and_loss, has_aux=True), in_dims=(0, 0, 0, 0))


def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim=-1)


# ============================================================================
# Test 1: GELU backward correctness
# ============================================================================

class TestGELUBackward:
    def test_gelu_backward_matches_autograd(self):
        """_gelu_backward should match torch.autograd.grad of F.gelu."""
        from titans_pytorch.neural_memory import _gelu_backward

        torch.manual_seed(42)
        x = torch.randn(8, 16, 64, requires_grad=True)
        grad_output = torch.randn_like(x)

        # Autograd reference
        y = F.gelu(x)
        (autograd_result,) = torch.autograd.grad(y, x, grad_output)

        # Manual
        manual_result = _gelu_backward(x.detach(), grad_output)

        assert torch.allclose(manual_result, autograd_result, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(manual_result - autograd_result).abs().max().item()}"

    def test_gelu_backward_various_shapes(self):
        """Test GELU backward with different tensor shapes."""
        from titans_pytorch.neural_memory import _gelu_backward

        for shape in [(4, 8, 32), (1, 1, 16), (16, 64, 128)]:
            x = torch.randn(*shape, requires_grad=True)
            grad_output = torch.randn_like(x)
            y = F.gelu(x)
            (ref,) = torch.autograd.grad(y, x, grad_output)
            manual = _gelu_backward(x.detach(), grad_output)
            assert torch.allclose(manual, ref, atol=1e-5, rtol=1e-5)

    def test_gelu_backward_zeros(self):
        """GELU backward at zero should be 0.5 * grad_output."""
        from titans_pytorch.neural_memory import _gelu_backward

        x = torch.zeros(2, 4, 8)
        grad_output = torch.ones_like(x)
        result = _gelu_backward(x, grad_output)
        # gelu'(0) = Phi(0) + 0*phi(0) = 0.5
        assert torch.allclose(result, torch.full_like(result, 0.5), atol=1e-5)


# ============================================================================
# Test 2: Manual grad for MemoryMLP (no ResidualNorm)
# ============================================================================

class TestManualGradMemoryMLP:
    @pytest.fixture
    def setup_depth2(self):
        """Create depth-2 MemoryMLP with test data."""
        torch.manual_seed(42)
        dim, chunk_size, batch = 64, 8, 4
        model = MemoryMLP(dim, depth=2, expansion_factor=2.)

        # Create per-sample weight copies (as in store_memories)
        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)
        loss_weights = torch.rand(batch, chunk_size).abs() + 0.1

        return model, params, keys, values, loss_weights, dim

    def test_depth2_grads_match_vmap(self, setup_depth2):
        """Manual gradient for depth-2 MLP should match vmap(grad())."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        model, params, keys, values, loss_weights, dim = setup_depth2

        # Reference: vmap(grad())
        ref_fn = make_vmap_grad_fn(model, default_loss_fn)
        ref_grads, ref_losses = ref_fn(params, keys, loss_weights, values)

        # Manual
        manual_grads, manual_losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        # Compare losses
        assert torch.allclose(manual_losses, ref_losses, atol=1e-5, rtol=1e-5), \
            f"Loss max diff: {(manual_losses - ref_losses).abs().max().item()}"

        # Compare gradients for each parameter
        for name in params:
            assert torch.allclose(manual_grads[name], ref_grads[name], atol=1e-5, rtol=1e-5), \
                f"Grad '{name}' max diff: {(manual_grads[name] - ref_grads[name]).abs().max().item()}"

    def test_depth3_grads_match_vmap(self):
        """Manual gradient for depth-3 MLP should match vmap(grad())."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 32, 4, 6
        model = MemoryMLP(dim, depth=3, expansion_factor=2.)

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)
        loss_weights = torch.rand(batch, chunk_size).abs() + 0.1

        ref_fn = make_vmap_grad_fn(model, default_loss_fn)
        ref_grads, ref_losses = ref_fn(params, keys, loss_weights, values)

        manual_grads, manual_losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        assert torch.allclose(manual_losses, ref_losses, atol=1e-5, rtol=1e-5)
        for name in params:
            assert torch.allclose(manual_grads[name], ref_grads[name], atol=1e-4, rtol=1e-4), \
                f"Grad '{name}' max diff: {(manual_grads[name] - ref_grads[name]).abs().max().item()}"

    def test_depth1_grads_match_vmap(self):
        """Manual gradient for depth-1 (linear) MLP should match vmap(grad())."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 64, 8, 4
        model = MemoryMLP(dim, depth=1, expansion_factor=1.)

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)
        loss_weights = torch.ones(batch, chunk_size)

        ref_fn = make_vmap_grad_fn(model, default_loss_fn)
        ref_grads, ref_losses = ref_fn(params, keys, loss_weights, values)

        manual_grads, manual_losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        assert torch.allclose(manual_losses, ref_losses, atol=1e-5, rtol=1e-5)
        for name in params:
            assert torch.allclose(manual_grads[name], ref_grads[name], atol=1e-5, rtol=1e-5), \
                f"Grad '{name}' max diff: {(manual_grads[name] - ref_grads[name]).abs().max().item()}"


# ============================================================================
# Test 3: Manual grad with ResidualNorm
# ============================================================================

class TestManualGradResidualNorm:
    def test_depth2_with_residnorm_grads_match(self):
        """Manual gradient for ResidualNorm(MemoryMLP) should match vmap(grad())."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 64, 8, 4
        inner = MemoryMLP(dim, depth=2, expansion_factor=2.)
        model = ResidualNorm(dim, inner)

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)
        loss_weights = torch.rand(batch, chunk_size).abs() + 0.1

        ref_fn = make_vmap_grad_fn(model, default_loss_fn)
        ref_grads, ref_losses = ref_fn(params, keys, loss_weights, values)

        manual_grads, manual_losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        assert torch.allclose(manual_losses, ref_losses, atol=1e-5, rtol=1e-5), \
            f"Loss max diff: {(manual_losses - ref_losses).abs().max().item()}"

        for name in params:
            assert torch.allclose(manual_grads[name], ref_grads[name], atol=1e-4, rtol=1e-4), \
                f"Grad '{name}' max diff: {(manual_grads[name] - ref_grads[name]).abs().max().item()}"

    def test_depth3_with_residnorm_grads_match(self):
        """Manual gradient for ResidualNorm(MemoryMLP depth=3) should match vmap."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 32, 4, 6
        inner = MemoryMLP(dim, depth=3, expansion_factor=2.)
        model = ResidualNorm(dim, inner)

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)
        loss_weights = torch.rand(batch, chunk_size).abs() + 0.1

        ref_fn = make_vmap_grad_fn(model, default_loss_fn)
        ref_grads, ref_losses = ref_fn(params, keys, loss_weights, values)

        manual_grads, manual_losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        assert torch.allclose(manual_losses, ref_losses, atol=1e-5, rtol=1e-5)
        for name in params:
            assert torch.allclose(manual_grads[name], ref_grads[name], atol=1e-4, rtol=1e-4), \
                f"Grad '{name}' max diff: {(manual_grads[name] - ref_grads[name]).abs().max().item()}"


# ============================================================================
# Test 4: Loss weighting correctness
# ============================================================================

class TestManualGradLossWeights:
    def test_uniform_weights_equal_unweighted(self):
        """With uniform loss_weights=1.0, weighted and unweighted grads should scale correctly."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 32, 4, 4
        model = MemoryMLP(dim, depth=2, expansion_factor=2.)

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)

        # loss_weights = 1.0 everywhere
        lw_ones = torch.ones(batch, chunk_size)
        # loss_weights = 2.0 everywhere
        lw_twos = torch.full((batch, chunk_size), 2.0)

        grads_1, _ = _manual_memory_mlp_grad(model, params, keys, lw_ones, values, default_loss_fn)
        grads_2, _ = _manual_memory_mlp_grad(model, params, keys, lw_twos, values, default_loss_fn)

        for name in params:
            assert torch.allclose(grads_2[name], grads_1[name] * 2, atol=1e-5, rtol=1e-5), \
                f"Scaling mismatch for '{name}'"

    def test_zero_weight_gives_zero_grad(self):
        """loss_weights=0 should produce zero gradients."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 32, 4, 4
        model = MemoryMLP(dim, depth=2, expansion_factor=2.)

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}
        keys = torch.randn(batch, chunk_size, dim)
        values = torch.randn(batch, chunk_size, dim)
        lw_zero = torch.zeros(batch, chunk_size)

        grads, _ = _manual_memory_mlp_grad(model, params, keys, lw_zero, values, default_loss_fn)
        for name in params:
            assert torch.allclose(grads[name], torch.zeros_like(grads[name]), atol=1e-7), \
                f"Non-zero grad with zero weights for '{name}'"


# ============================================================================
# Test 5: Outer backward (training gradient flow)
# ============================================================================

class TestOuterBackward:
    def test_manual_grad_supports_outer_backward(self):
        """
        The manual gradient computation should be differentiable w.r.t.
        the inputs (keys, values) so that outer training gradients flow
        through projections (to_keys, to_values, etc.).
        """
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 32, 4, 4
        model = MemoryMLP(dim, depth=2, expansion_factor=2.)

        # Keys and values require grad (simulating projection output)
        keys = torch.randn(batch, chunk_size, dim, requires_grad=True)
        values = torch.randn(batch, chunk_size, dim, requires_grad=True)
        loss_weights = torch.rand(batch, chunk_size).abs() + 0.1

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}

        grads, losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        # Sum all gradients and backprop - should flow to keys/values
        total = sum(g.sum() for g in grads.values()) + losses.sum()
        total.backward()

        assert keys.grad is not None, "No gradient flowed to keys"
        assert values.grad is not None, "No gradient flowed to values"
        assert not torch.all(keys.grad == 0), "Keys gradient is all zeros"
        assert not torch.all(values.grad == 0), "Values gradient is all zeros"

    def test_outer_backward_with_residnorm(self):
        """Outer backward should work with ResidualNorm wrapping too."""
        from titans_pytorch.neural_memory import _manual_memory_mlp_grad

        torch.manual_seed(42)
        dim, chunk_size, batch = 32, 4, 4
        inner = MemoryMLP(dim, depth=2, expansion_factor=2.)
        model = ResidualNorm(dim, inner)

        keys = torch.randn(batch, chunk_size, dim, requires_grad=True)
        values = torch.randn(batch, chunk_size, dim, requires_grad=True)
        loss_weights = torch.rand(batch, chunk_size).abs() + 0.1

        params = {k: v.unsqueeze(0).expand(batch, *v.shape).clone()
                  for k, v in dict(model.named_parameters()).items()}

        grads, losses = _manual_memory_mlp_grad(
            model, params, keys, loss_weights, values, default_loss_fn,
        )

        total = sum(g.sum() for g in grads.values()) + losses.sum()
        total.backward()

        assert keys.grad is not None
        assert values.grad is not None


# ============================================================================
# Test 6: Full NeuralMemory integration
# ============================================================================

class TestFullNeuralMemoryIntegration:
    def test_manual_vs_vmap_output_equivalence(self):
        """
        Full NeuralMemory with use_manual_grad=True should produce
        identical output to use_manual_grad=False (vmap).
        """
        from titans_pytorch import NeuralMemory

        dim, seq_len = 64, 32

        # Create two identical modules (same seed → same init weights)
        torch.manual_seed(42)
        mem_manual = NeuralMemory(dim=dim, chunk_size=8, use_manual_grad=True)
        torch.manual_seed(42)
        mem_vmap = NeuralMemory(dim=dim, chunk_size=8, use_manual_grad=False)

        torch.manual_seed(99)
        seq = torch.randn(2, seq_len, dim)

        out_manual, _ = mem_manual(seq)
        out_vmap, _ = mem_vmap(seq)

        assert torch.allclose(out_manual, out_vmap, atol=1e-4, rtol=1e-4), \
            f"Output max diff: {(out_manual - out_vmap).abs().max().item()}"

    def test_manual_grad_with_momentum(self):
        """Manual grad should work correctly with momentum enabled."""
        from titans_pytorch import NeuralMemory

        dim, seq_len = 64, 32

        torch.manual_seed(42)
        mem_manual = NeuralMemory(dim=dim, chunk_size=8, momentum=True, momentum_order=1, use_manual_grad=True)
        torch.manual_seed(42)
        mem_vmap = NeuralMemory(dim=dim, chunk_size=8, momentum=True, momentum_order=1, use_manual_grad=False)

        torch.manual_seed(99)
        seq = torch.randn(2, seq_len, dim)

        out_manual, _ = mem_manual(seq)
        out_vmap, _ = mem_vmap(seq)

        assert torch.allclose(out_manual, out_vmap, atol=1e-4, rtol=1e-4), \
            f"Output max diff: {(out_manual - out_vmap).abs().max().item()}"

    def test_manual_grad_multi_head(self):
        """Manual grad should work with multiple heads."""
        from titans_pytorch import NeuralMemory

        dim, seq_len = 64, 32

        torch.manual_seed(42)
        mem_manual = NeuralMemory(dim=dim, heads=4, chunk_size=8, use_manual_grad=True)
        torch.manual_seed(42)
        mem_vmap = NeuralMemory(dim=dim, heads=4, chunk_size=8, use_manual_grad=False)

        torch.manual_seed(99)
        seq = torch.randn(2, seq_len, dim)

        out_manual, _ = mem_manual(seq)
        out_vmap, _ = mem_vmap(seq)

        assert torch.allclose(out_manual, out_vmap, atol=1e-4, rtol=1e-4), \
            f"Output max diff: {(out_manual - out_vmap).abs().max().item()}"

    def test_manual_grad_unsupported_model_falls_back(self):
        """Non-MemoryMLP models should auto-fallback to vmap."""
        from titans_pytorch import NeuralMemory
        from titans_pytorch.memory_models import MemoryAttention

        torch.manual_seed(42)
        dim, seq_len = 64, 32

        # MemoryAttention is not supported by manual grad
        mem = NeuralMemory(
            dim=dim,
            chunk_size=8,
            model=MemoryAttention(dim),
            use_manual_grad=True,  # should auto-fallback
        )

        seq = torch.randn(2, seq_len, dim)
        # Should not raise
        out, state = mem(seq)
        assert out.shape == (2, seq_len, dim)


# ============================================================================
# Test 7: Speed benchmark (informational, does not fail)
# ============================================================================

class TestManualGradSpeed:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_manual_faster_than_vmap_gpu(self):
        """Benchmark: manual grad should be faster than vmap on GPU."""
        from titans_pytorch import NeuralMemory
        import time

        torch.manual_seed(42)
        dim, seq_len = 128, 256
        device = 'cuda'

        mem_manual = NeuralMemory(dim=dim, chunk_size=16, use_manual_grad=True).to(device)
        mem_vmap = NeuralMemory(dim=dim, chunk_size=16, use_manual_grad=False).to(device)
        mem_vmap.load_state_dict(mem_manual.state_dict())

        seq = torch.randn(4, seq_len, dim, device=device)

        # Warmup
        for _ in range(3):
            mem_manual(seq)
            mem_vmap(seq)
        torch.cuda.synchronize()

        # Benchmark manual
        n_iters = 10
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            mem_manual(seq)
        torch.cuda.synchronize()
        time_manual = time.perf_counter() - t0

        # Benchmark vmap
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            mem_vmap(seq)
        torch.cuda.synchronize()
        time_vmap = time.perf_counter() - t0

        speedup = time_vmap / time_manual
        print(f"\n  Manual: {time_manual/n_iters*1000:.1f}ms, Vmap: {time_vmap/n_iters*1000:.1f}ms, Speedup: {speedup:.2f}x")

        # Informational - we expect manual to be faster but don't fail if not
        # (CPU-only environments may not show improvement)

    def test_manual_faster_than_vmap_cpu(self):
        """Benchmark: manual grad vs vmap on CPU (informational)."""
        from titans_pytorch import NeuralMemory
        import time

        torch.manual_seed(42)
        dim, seq_len = 64, 64

        mem_manual = NeuralMemory(dim=dim, chunk_size=8, use_manual_grad=True)
        mem_vmap = NeuralMemory(dim=dim, chunk_size=8, use_manual_grad=False)
        mem_vmap.load_state_dict(mem_manual.state_dict())

        seq = torch.randn(2, seq_len, dim)

        # Warmup
        for _ in range(2):
            mem_manual(seq)
            mem_vmap(seq)

        n_iters = 5
        t0 = time.perf_counter()
        for _ in range(n_iters):
            mem_manual(seq)
        time_manual = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iters):
            mem_vmap(seq)
        time_vmap = time.perf_counter() - t0

        speedup = time_vmap / time_manual
        print(f"\n  CPU Manual: {time_manual/n_iters*1000:.1f}ms, Vmap: {time_vmap/n_iters*1000:.1f}ms, Speedup: {speedup:.2f}x")
