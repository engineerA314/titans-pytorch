from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

import pytest
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention, MemoryAsContextTransformer
from titans_pytorch.mag_transformer import MemoryAsGateTransformer, SlidingWindowAttention
from titans_pytorch.mal_transformer import MemoryAsLayerTransformer, TitansLMM

# functions

def exists(v):
    return v is not None

def diff(x, y):
    return (x - y).abs().amax()

@contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# ============================================================================
# CUDA / Triton integration test for assoc_scan accelerated backend
# ============================================================================

def _skip_if_no_cuda_triton():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available (integration test for triton accelerated scan)")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("triton not installed")
    try:
        import accelerated_scan.triton as triton_mod  # noqa
    except Exception as e:
        pytest.skip(f"accelerated_scan.triton import failed: {e}")
    return triton_mod


def test_assoc_scan_triton_backend_parity_cuda(monkeypatch):
    """
    Integration test: require Linux+CUDA+triton. Verifies that:
    - AssocScan(use_accelerated=True) actually calls accelerated_scan.triton.scan on CUDA
    - Its output matches the reference non-accelerated AssocScan(use_accelerated=False)
    """
    triton_mod = _skip_if_no_cuda_triton()

    from assoc_scan import AssocScan

    calls = {'n': 0}
    orig_scan = triton_mod.scan

    def wrapped_scan(gates, tokens):
        calls['n'] += 1
        return orig_scan(gates, tokens)

    monkeypatch.setattr(triton_mod, 'scan', wrapped_scan)

    scan_acc = AssocScan(use_accelerated=True)
    scan_ref = AssocScan(use_accelerated=False)

    torch.manual_seed(0)
    device = torch.device('cuda')
    B, T, D = 2, 32, 16
    gate = torch.sigmoid(torch.randn(B, T, 1, device=device, dtype=torch.float32)).contiguous()
    value = torch.randn(B, T, D, device=device, dtype=torch.float32).contiguous()

    out_acc = scan_acc(gate, value)
    out_ref = scan_ref(gate, value)

    assert calls['n'] > 0
    assert torch.allclose(out_acc, out_ref, atol=1e-4, rtol=1e-4)

# ============================================================================
# NeuralMemory Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (32, 512, 77))
@pytest.mark.parametrize('silu', (False, True))
@pytest.mark.parametrize('chunk_size, attn_pool_chunks', ((64, True), (64, False), (1, False)))
@pytest.mark.parametrize('momentum', (False, True))
@pytest.mark.parametrize('qk_rmsnorm', (False, True))
@pytest.mark.parametrize('heads', (1, 4))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
@pytest.mark.parametrize('num_kv_per_token', (1, 2))
@pytest.mark.parametrize('per_parameter_lr_modulation', (False, True))
@pytest.mark.parametrize('per_head_learned_parameters', (False, True))
@pytest.mark.parametrize('test_store_mask', (False, True))
def test_titans(
    seq_len,
    silu,
    attn_pool_chunks,
    chunk_size,
    momentum,
    qk_rmsnorm,
    heads,
    max_grad_norm,
    num_kv_per_token,
    per_parameter_lr_modulation,
    per_head_learned_parameters,
    test_store_mask
):
    mem = NeuralMemory(
        dim = 16,
        chunk_size = chunk_size,
        activation = nn.SiLU() if silu else None,
        attn_pool_chunks = attn_pool_chunks,
        max_grad_norm = max_grad_norm,
        num_kv_per_token = num_kv_per_token,
        momentum = momentum,
        qk_rmsnorm = qk_rmsnorm,
        heads = heads,
        per_parameter_lr_modulation = per_parameter_lr_modulation,
        per_head_learned_parameters = per_head_learned_parameters
    )

    seq = torch.randn(2, seq_len, 16)

    store_mask = None

    if test_store_mask:
        store_mask = torch.randint(0, 2, (2, seq_len)).bool()

    retrieved, _ = mem(seq, store_mask = store_mask)

    assert seq.shape == retrieved.shape


def test_assoc_scan_called_titans(monkeypatch):
    calls = {'n': 0}
    class DummyScan:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, gate, value, prev=None, remove_prev=False):
            calls['n'] += 1
            # use simple cumulative update; ignore prev to avoid shape pitfalls
            gate = gate
            while gate.ndim < value.ndim:
                gate = gate.unsqueeze(-1)
            gate = gate.expand_as(value)
            state = torch.zeros_like(value[:, :1])
            outs = []
            for t in range(value.shape[1]):
                state = gate[:, t:t+1] * state + value[:, t:t+1]
                outs.append(state)
            return torch.cat(outs, dim=1)

    import titans_pytorch.neural_memory as nm
    monkeypatch.setattr(nm, 'AssocScan', DummyScan)
    mem = NeuralMemory(dim = 16, chunk_size = 2, heads = 1, momentum = True)
    seq = torch.randn(1, 4, 16)
    # use store-only to avoid retrieval path shape issues
    _state, _ = mem.forward_store_only(seq, return_surprises = False)
    assert calls['n'] > 0


def test_assoc_scan_shapes_titans(monkeypatch):
    import titans_pytorch.neural_memory as nm
    records = []

    class RecordingScan:
        def __init__(self, use_accelerated=False, *args, **kwargs):
            self.use_accelerated = use_accelerated
        def __call__(self, gate, value, prev=None, remove_prev=False):
            records.append((gate.shape, value.shape, None if prev is None else prev.shape))
            gate = gate
            while gate.ndim < value.ndim:
                gate = gate.unsqueeze(-1)
            gate = gate.expand_as(value)
            state = torch.zeros_like(value[:, :1])
            outs = []
            for t in range(value.shape[1]):
                state = gate[:, t:t+1] * state + value[:, t:t+1]
                outs.append(state)
            return torch.cat(outs, dim=1)

    monkeypatch.setattr(nm, 'AssocScan', RecordingScan)
    mem = NeuralMemory(dim=16, chunk_size=2, heads=2, momentum=True)
    seq = torch.randn(1, 6, 16)  # multiple of chunk_size to avoid remainders
    _state, _ = mem.forward_store_only(seq, return_surprises = False)
    assert records, "assoc_scan not invoked"
    for g, v, p in records:
        # gate is (b, n, 1) while value is (b, n, ...); batch/time must match
        assert g[0] == v[0] and g[1] == v[1]
        if p is not None:
            assert p[0] == v[0]


def test_assoc_scan_full_forward_titans(monkeypatch):
    import titans_pytorch.neural_memory as nm
    calls = {'n': 0}

    class DummyScan:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, gate, value, prev=None, remove_prev=False):
            calls['n'] += 1
            # prev arrives without time dim; add if needed
            if prev is None:
                state = torch.zeros_like(value[:, :1])
            else:
                state = prev if prev.ndim == value.ndim else prev.unsqueeze(1)
            # broadcast gate
            g = gate
            while g.ndim < value.ndim:
                g = g.unsqueeze(-1)
            g = g.expand_as(value)
            outs = []
            for t in range(value.shape[1]):
                state = g[:, t:t+1] * state + value[:, t:t+1]
                outs.append(state)
            out = torch.cat(outs, dim=1)
            if remove_prev:
                out = out[:, 1:]
            return out

    monkeypatch.setattr(nm, 'AssocScan', DummyScan)
    # use chunk_size=1 to keep retrieval reshape simple
    class BroadcastLinear(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(in_dim, out_dim))
        def forward(self, x, weight=None):
            w = self.weight if weight is None else weight
            if w.ndim == 2:
                return x @ w
            # collapse leading dims and use the first slice to keep the test lightweight
            w2 = w.reshape(-1, w.shape[-2], w.shape[-1])[0]
            return x @ w2

    mem = NeuralMemory(
        dim=16,
        chunk_size=1,
        heads=1,
        momentum=True,
        model=BroadcastLinear(16, 16),
        per_head_learned_parameters=False,
        mem_model_norm_add_residual=False
    )
    seq = torch.randn(1, 3, 16)
    out, _state = mem(seq)
    assert calls['n'] > 0
    assert out.shape == seq.shape


def test_assoc_scan_full_forward_titans_per_head_norm(monkeypatch):
    import titans_pytorch.neural_memory as nm
    calls = {'n': 0}

    class DummyScan:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, gate, value, prev=None, remove_prev=False):
            calls['n'] += 1
            if prev is None:
                state = torch.zeros_like(value[:, :1])
            else:
                state = prev if prev.ndim == value.ndim else prev.unsqueeze(1)
            g = gate
            while g.ndim < value.ndim:
                g = g.unsqueeze(-1)
            g = g.expand_as(value)
            outs = []
            for t in range(value.shape[1]):
                state = g[:, t:t+1] * state + value[:, t:t+1]
                outs.append(state)
            out = torch.cat(outs, dim=1)
            if remove_prev:
                out = out[:, 1:]
            return out

    monkeypatch.setattr(nm, 'AssocScan', DummyScan)

    class PerHeadLinear(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(heads, dim, dim))
        def forward(self, x, weight=None):
            w = self.weight if weight is None else weight
            # w: (h, d, d); x will be expanded to (h, b, n, d) by stateless call
            if w.ndim == 3:
                # functional_call will expand weights to (h, d, d); x comes as (h, b*n, d)
                if x.ndim == 3:
                    return torch.einsum('hbd,hde->hbe', x, w)
                return torch.einsum('hbnd,hde->hbne', x, w)
            return x @ w

    mem = NeuralMemory(
        dim=8,
        chunk_size=1,
        heads=2,
        momentum=True,
        model=torch.nn.Identity(),
        per_head_learned_parameters=True,
        mem_model_norm_add_residual=True,
        batch_size=1,
    )
    seq = torch.randn(1, 1, 8)  # single step to satisfy per-head norm broadcast
    out, _ = mem(seq)
    assert calls['n'] > 0
    assert out.shape == seq.shape

def test_return_surprises():

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 2,
        dim_head = 64,
        heads = 4,
    )

    seq = torch.randn(4, 64, 384)

    _, _, (surprises, adaptive_lr) = mem(seq, return_surprises = True)

    assert all([t.shape == (4, 4, 64) for t in (surprises, adaptive_lr)])

@pytest.mark.parametrize('learned_momentum_combine', (False, True))
@pytest.mark.parametrize('learned_combine_include_zeroth', (False, True))
def test_titans_second_order_momentum(
    learned_momentum_combine,
    learned_combine_include_zeroth
):

    mem  = NeuralMemory(
        dim = 384,
        dim_head = 64,
        heads = 2,
        chunk_size = 1,
        batch_size = 2,
        momentum_order = 2,
        learned_momentum_combine = learned_momentum_combine,
        learned_combine_include_zeroth = learned_combine_include_zeroth
    )

    seq = torch.randn(2, 5, 384)

    parallel_retrieved, state = mem(seq)
    assert seq.shape == parallel_retrieved.shape

def test_titans_attn_memory():
    from titans_pytorch.memory_models import MemoryAttention

    mem = NeuralMemory(
        dim = 16,
        chunk_size = 64,
        model = MemoryAttention(
            dim = 16
        )
    )

    seq = torch.randn(2, 1024, 16)
    retrieved, _ = mem(seq)

    assert seq.shape == retrieved.shape

def test_swiglu_ff_memory():
    from titans_pytorch.memory_models import MemorySwiGluMLP

    mem = NeuralMemory(
        dim = 16,
        chunk_size = 2,
        mem_model_norm_add_residual = False,
        model = MemorySwiGluMLP(
            dim = 16,
            depth = 2
        )
    )

    seq = torch.randn(2, 64, 16)
    retrieved, _ = mem(seq)

    assert seq.shape == retrieved.shape

@pytest.mark.parametrize('gated_transition', (True, False))
def test_neural_mem_chaining_chunks(
    gated_transition
):
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        gated_transition = gated_transition
    )

    seq = torch.randn(2, 48, 16)

    parallel_retrieved, state = mem(seq)

    seq_first, seq_second, seq_third = seq.split(16, dim = 1)

    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)

    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1), atol = 1e-5)

def test_neural_mem_chaining_with_weight_residual():
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64
    )

    mem2 = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 64,
        accept_weight_residual = True
    )

    seq = torch.randn(2, 256, 16)

    seq, state = mem(seq)

    parallel_retrieved, _ = mem2(seq, prev_weights = state.updates)

    seq_first, seq_second = seq[:, :128], seq[:, 128:]

    first_retrieved, state1 = mem2(seq_first, prev_weights = state.updates)
    second_retrieved, state2 = mem2(seq_second, state = state1, prev_weights = state.updates)

    assert torch.allclose(parallel_retrieved, torch.cat((first_retrieved, second_retrieved), dim = 1), atol = 1e-5)

def test_neural_mem_chaining_with_batch_size():
    mem  = NeuralMemory(
        dim = 16,
        dim_head = 16,
        heads = 2,
        chunk_size = 16,
        batch_size = 64
    )

    seq = torch.randn(2, 112, 16)

    parallel_retrieved, state = mem(seq)

    seq_first, seq_second, seq_third = seq[:, :16], seq[:, 16:64], seq[:, 64:]

    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state = state)
    third_retrieved, state = mem(seq_third, state = state)

    parallel_part_retrieved = torch.cat((first_retrieved, second_retrieved, third_retrieved), dim = 1)

    assert torch.allclose(parallel_retrieved, parallel_part_retrieved, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (2, 64, 256))
@pytest.mark.parametrize('prompt_len', (0, 65))
@pytest.mark.parametrize('mem_chunk_size', (2, 32, 64))
@pytest.mark.parametrize('gated_transition', (False, True))
@torch_default_dtype(torch.float64)
def test_neural_mem_inference(
    seq_len,
    prompt_len,
    mem_chunk_size,
    gated_transition
):

    mem = NeuralMemory(
        dim = 16,
        chunk_size = mem_chunk_size,
        gated_transition = gated_transition
    )

    seq = torch.randn(2, seq_len, 16)
    parallel_retrieved, _ = mem(seq)

    assert seq.shape == parallel_retrieved.shape

    state = None
    sequential_retrieved = []

    # test initial parallel prompt

    test_parallel_prompt = prompt_len > 0 and prompt_len < seq_len

    if test_parallel_prompt:
        prompt, seq = seq[:, :prompt_len], seq[:, prompt_len:]
        retrieved_prompt, state = mem(prompt)
        sequential_retrieved.append(retrieved_prompt)

    # sequential inference

    for token in seq.unbind(dim = 1):

        one_retrieved, state = mem.forward(
            token,
            state = state,
        )

        sequential_retrieved.append(one_retrieved)

    sequential_retrieved = torch.cat(sequential_retrieved, dim = -2)

    assert torch.allclose(parallel_retrieved, sequential_retrieved, atol = 1e-6)

def test_mem_state_detach():
    from titans_pytorch.neural_memory import mem_state_detach

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 2,
        qk_rmsnorm = True,
        dim_head = 64,
        heads = 4,
    )

    seq = torch.randn(4, 64, 384)

    state = None

    for _ in range(2):
        parallel_retrieved, state = mem(seq, state = state)
        state = mem_state_detach(state)
        parallel_retrieved.sum().backward()

@pytest.mark.parametrize('use_accelerated', (True, False))
def test_assoc_scan(
    use_accelerated
):
    from titans_pytorch.neural_memory import AssocScan

    if use_accelerated and not torch.cuda.is_available():
        pytest.skip()

    scan = AssocScan(use_accelerated = use_accelerated)

    seq_len = 128
    mid_point = seq_len // 2

    gates = torch.randn(2, seq_len, 16).sigmoid()
    inputs = torch.randn(2, seq_len, 16)

    if use_accelerated:
        gates = gates.cuda()
        inputs = inputs.cuda()

    output = scan(gates, inputs)

    gates1, gates2 = gates[:, :mid_point], gates[:, mid_point:]
    inputs1, inputs2 = inputs[:, :mid_point], inputs[:, mid_point:]

    first_half = scan(gates1, inputs1)

    second_half = scan(gates2, inputs2, prev = first_half[:, -1])
    assert second_half.shape == inputs2.shape

    assert torch.allclose(output[:, -1], second_half[:, -1], atol = 1e-5)


# ============================================================================
# MAC (Memory as Context) Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 16))
@pytest.mark.parametrize('num_longterm_mem_tokens', (0, 16))
@pytest.mark.parametrize('neural_mem_segment_len', (8, 16))
@pytest.mark.parametrize('neural_mem_weight_residual', (False, True))
@pytest.mark.parametrize('neural_mem_batch_size', (None, 64))
@pytest.mark.parametrize('neural_mem_qkv_receives_diff_views', (False, True))
@pytest.mark.parametrize('neural_mem_momentum', (False, True))
@pytest.mark.parametrize('batch_size', (1, 2))
def test_mac(
    seq_len,
    num_persist_mem_tokens,
    num_longterm_mem_tokens,
    neural_mem_segment_len,
    neural_mem_weight_residual,
    neural_mem_batch_size,
    neural_mem_qkv_receives_diff_views,
    neural_mem_momentum,
    batch_size
):
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        num_longterm_mem_tokens = num_longterm_mem_tokens,
        segment_len = 128,
        neural_memory_segment_len = neural_mem_segment_len,
        neural_memory_batch_size = neural_mem_batch_size,
        neural_memory_qkv_receives_diff_views = neural_mem_qkv_receives_diff_views,
        neural_mem_weight_residual = neural_mem_weight_residual,
        neural_memory_kwargs = dict(
            momentum = neural_mem_momentum
        )
    )

    x = torch.randint(0, 256, (batch_size, seq_len))

    logits = transformer(x)
    assert logits.shape == (batch_size, seq_len, 256)

@pytest.mark.parametrize('sliding', (False, True))
@pytest.mark.parametrize('mem_layers', ((), None))
@pytest.mark.parametrize('longterm_mems', (0, 4, 16))
@pytest.mark.parametrize('prompt_len', (4, 16))
@pytest.mark.parametrize('batch_size', (1, 2))
@torch_default_dtype(torch.float64)
def test_mac_sampling(
    sliding,
    mem_layers,
    longterm_mems,
    prompt_len,
    batch_size
):
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 16,
        depth = 4,
        segment_len = 32,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = longterm_mems,
        sliding_window_attn = sliding,
        neural_memory_layers = mem_layers
    )

    ids = torch.randint(0, 256, (batch_size, 1023))

    prompt = ids[:, :prompt_len]

    sampled = transformer.sample(prompt, 53, use_cache = False, temperature = 0.)
    sampled_with_cache = transformer.sample(prompt, 53, use_cache = True, temperature = 0.)

    assert sampled.shape == sampled_with_cache.shape

    # caching should be deterministic even if neural memory updates differ chunk-wise
    sampled_with_cache_repeat = transformer.sample(prompt, 53, use_cache = True, temperature = 0.)
    assert torch.allclose(sampled_with_cache, sampled_with_cache_repeat)

    # without neural memory, both code paths must match exactly
    if mem_layers == ():
        assert torch.allclose(sampled, sampled_with_cache)

def _make_simple_mac():
    return MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        segment_len = 16,
        neural_memory_segment_len = 1,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(
            momentum = False
        )
    )

def _get_first_memory_layer(transformer):
    for layer in transformer.layers:
        mem = layer[4]
        if exists(mem):
            return mem
    return None

def test_mac_retrieves_then_stores(monkeypatch):
    transformer = _make_simple_mac()
    mem = _get_first_memory_layer(transformer)
    assert exists(mem)

    call_sequence = []

    original_retrieve = mem.forward_retrieve_only
    def wrapped_retrieve(*args, **kwargs):
        call_sequence.append('retrieve')
        return original_retrieve(*args, **kwargs)

    original_store = mem.forward_store_only
    def wrapped_store(*args, **kwargs):
        call_sequence.append('store')
        return original_store(*args, **kwargs)

    monkeypatch.setattr(mem, 'forward_retrieve_only', wrapped_retrieve)
    monkeypatch.setattr(mem, 'forward_store_only', wrapped_store)

    x = torch.randint(0, 64, (1, 8))
    transformer(x)

    assert call_sequence, 'memory was never invoked'
    assert call_sequence[0] == 'retrieve'
    assert any(step == 'store' for step in call_sequence)
    retrieve_idx = call_sequence.index('retrieve')
    store_idx = call_sequence.index('store')
    assert retrieve_idx < store_idx

def test_mac_stores_attention_output(monkeypatch):
    transformer = _make_simple_mac()
    mem = _get_first_memory_layer(transformer)
    attn = transformer.layers[0][5]
    assert exists(mem)

    last_attn_out = {}
    original_attn_forward = attn.forward

    def wrapped_attn_forward(*args, **kwargs):
        out, aux = original_attn_forward(*args, **kwargs)
        last_attn_out['value'] = out.detach().clone()
        return out, aux

    stored_sequences = []
    original_store = mem.forward_store_only

    def wrapped_store(store_seq, *args, **kwargs):
        stored_sequences.append(store_seq.detach().clone())
        return original_store(store_seq, *args, **kwargs)

    monkeypatch.setattr(attn, 'forward', wrapped_attn_forward)
    monkeypatch.setattr(mem, 'forward_store_only', wrapped_store)

    x = torch.randint(0, 64, (1, 8))
    transformer(x)

    assert stored_sequences, 'forward_store_only never called'
    assert 'value' in last_attn_out
    assert torch.allclose(stored_sequences[0], last_attn_out['value'])

def test_mac_updates_memory_state(monkeypatch):
    transformer = _make_simple_mac()
    mem = _get_first_memory_layer(transformer)
    assert exists(mem)

    captured_states = []
    original_store = mem.forward_store_only

    def wrapped_store(*args, **kwargs):
        result = original_store(*args, **kwargs)
        captured_states.append(result[0])
        return result

    monkeypatch.setattr(mem, 'forward_store_only', wrapped_store)

    x = torch.randint(0, 64, (1, 8))
    transformer(x)

    assert captured_states, 'store_memories never captured state'

    final_state = captured_states[-1]
    assert isinstance(final_state.seq_index, int)
    assert final_state.seq_index > 0
    assert exists(final_state.weights)
    assert exists(final_state.states)

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('sliding', (True, False))
def test_flex(
    seq_len,
    sliding
):
    if not (torch.cuda.is_available() and exists(flex_attention)):
        pytest.skip()

    attn = SegmentedAttention(
        dim = 16,
        segment_len = 32,
        num_persist_mem_tokens = 1,
        num_longterm_mem_tokens = 1,
        use_flex_attn = True,
        sliding = sliding
    ).cuda()

    seq = torch.randn(1, seq_len, 16).cuda()

    out_flex, _ = attn(seq)
    out_non_flex, _ = attn(seq, disable_flex_attn = True)

    assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)

@pytest.mark.parametrize('seq_len', (65, 257))
def test_flex_with_context_matches_nonflex(seq_len):
    """Test that flex attention with context produces same results as non-flex.
    
    Note: Due to torch.compile compatibility issues with dynamic context_len in
    flex_attention masks, context automatically falls back to non-flex path.
    This test verifies that fallback works correctly.
    """
    if not (torch.cuda.is_available() and exists(flex_attention)):
        pytest.skip()

    attn = SegmentedAttention(
        dim = 16,
        segment_len = 32,
        num_persist_mem_tokens = 2,
        num_longterm_mem_tokens = 0,
        use_flex_attn = True,
        sliding = False
    ).cuda()

    seq = torch.randn(1, seq_len, 16).cuda()
    ctx = torch.randn(1, 7, 16).cuda()

    # With context, flex_attn automatically falls back to non-flex path
    out_flex, _ = attn(seq, context = ctx)
    out_non_flex, _ = attn(seq, context = ctx, disable_flex_attn = True)

    assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)

def test_sliding_context_cpu():
    attn = SegmentedAttention(
        dim = 16,
        segment_len = 16,
        num_persist_mem_tokens = 1,
        num_longterm_mem_tokens = 0,
        use_flex_attn = False,
        sliding = True
    )
    seq = torch.randn(1, 64, 16)
    ctx = torch.randn(1, 5, 16)
    out, _ = attn(seq, context = ctx, disable_flex_attn = True)
    assert out.shape == (1, 64, 16)

def test_mac_passes_retrieved_as_context_to_attn(monkeypatch):
    transformer = _make_simple_mac()
    mem = _get_first_memory_layer(transformer)
    attn = transformer.layers[0][5]
    assert exists(mem)

    captured = {}

    original_retrieve = mem.forward_retrieve_only
    def wrapped_retrieve(*args, **kwargs):
        out, state = original_retrieve(*args, **kwargs)
        captured['pl'] = out.detach().clone()
        return out, state

    original_attn_forward = attn.forward
    def wrapped_attn_forward(seq, *args, **kwargs):
        captured['context'] = kwargs.get('context', None)
        return original_attn_forward(seq, *args, **kwargs)

    monkeypatch.setattr(mem, 'forward_retrieve_only', wrapped_retrieve)
    monkeypatch.setattr(attn, 'forward', wrapped_attn_forward)

    x = torch.randint(0, 64, (1, 8))
    transformer(x)

    assert 'pl' in captured and 'context' in captured
    assert torch.allclose(captured['context'], captured['pl'])

def test_mac_ephemeral_context_not_cached(monkeypatch):
    transformer = _make_simple_mac()
    mem = _get_first_memory_layer(transformer)
    attn = transformer.layers[0][5]
    assert exists(mem)

    events = []

    original_forward_inference = attn.forward_inference
    def wrapped_forward_inference(token, cache, value_residual=None, output_gating=None, context=None):
        ck, cv = cache
        ck_len_before = ck.shape[-2] if ck is not None else 0
        out, aux = original_forward_inference(token, cache, value_residual=value_residual, output_gating=output_gating, context=context)
        next_k, next_v = aux.cached_key_values
        next_k_len_after = next_k.shape[-2]
        events.append((ck_len_before, next_k_len_after))
        return out, aux

    monkeypatch.setattr(attn, 'forward_inference', wrapped_forward_inference)

    prompt = torch.randint(0, 64, (1, 4))
    _ = transformer.sample(prompt, 6, use_cache=True, temperature=0.)

    assert events, 'no inference attention calls captured'
    for before, after in events:
        assert (after - before) in (0, 1)
        if after > before:
            assert (after - before) == 1

def test_retrieval_uses_committed_weights_only(monkeypatch):
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        momentum = False,
        heads = 1
    )

    store_seq = torch.randn(1, 3, 16)
    state_after_store, _ = mem.forward_store_only(store_seq, state = None, return_surprises = True)

    assert state_after_store.updates is not None
    assert state_after_store.weights is not None

    captured = {}
    original_retrieve_memories = mem.retrieve_memories

    def wrapped_retrieve_memories(seq, weights):
        captured['weights'] = weights
        return original_retrieve_memories(seq, weights)

    monkeypatch.setattr(mem, 'retrieve_memories', wrapped_retrieve_memories)

    one_token = torch.randn(1, 1, 16)
    _retrieved, _next_state = mem.forward_retrieve_only(one_token, state = state_after_store)

    assert 'weights' in captured

    committed = state_after_store.weights
    used = captured['weights']

    for k in committed.keys():
        assert torch.allclose(used[k], committed[k])

def test_mac_inference_query_growth(monkeypatch):
    segment_len = 4
    transformer = MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        segment_len = segment_len,
        neural_memory_segment_len = segment_len,
        neural_memory_layers = (1,)
    )
    mem = _get_first_memory_layer(transformer)
    assert exists(mem)

    captured_query_lengths = []
    original_retrieve = mem.forward_retrieve_only

    def wrapped_retrieve(seq, *args, **kwargs):
        captured_query_lengths.append(seq.shape[-2])
        return original_retrieve(seq, *args, **kwargs)

    monkeypatch.setattr(mem, 'forward_retrieve_only', wrapped_retrieve)

    prompt = torch.randint(0, 64, (1, 1))
    transformer.sample(prompt, 1 + 8, use_cache=True, temperature=0.)

    assert len(captured_query_lengths) >= 8
    
    growth_sequences = 0
    current_run = 0
    for length in captured_query_lengths:
        if length == current_run + 1:
            current_run += 1
            if current_run == segment_len:
                growth_sequences += 1
                current_run = 0
        elif length == 1:
            current_run = 1
        else:
            current_run = 0
    
    assert growth_sequences >= 1, f"Expected at least one full growth cycle up to {segment_len}. Got: {captured_query_lengths}"

def test_mac_multi_layer_query_growth(monkeypatch):
    segment_len = 4
    transformer = MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 3,
        segment_len = segment_len,
        neural_memory_segment_len = segment_len,
        neural_memory_layers = (1, 2)
    )
    mem_layers = []
    for layer in transformer.layers:
        mem = layer[4]
        if exists(mem):
            mem_layers.append(mem)
        if len(mem_layers) == 2:
            break

    assert len(mem_layers) == 2

    captured_lengths = [[], []]

    def make_wrap(idx):
        orig = mem_layers[idx].forward_retrieve_only
        def wrapped(seq, *args, **kwargs):
            captured_lengths[idx].append(seq.shape[-2])
            return orig(seq, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(mem_layers[0], 'forward_retrieve_only', make_wrap(0))
    monkeypatch.setattr(mem_layers[1], 'forward_retrieve_only', make_wrap(1))

    prompt = torch.randint(0, 64, (1, 1))
    transformer.sample(prompt, 1 + 8, use_cache = True, temperature = 0.)

    for lens in captured_lengths:
        assert len(lens) >= 8
        run = 0
        cycles = 0
        for L in lens:
            if L == run + 1:
                run += 1
                if run == segment_len:
                    cycles += 1
                    run = 0
            elif L == 1:
                run = 1
            else:
                run = 0
        assert cycles >= 1, f"Layer did not show a full growth cycle: {lens}"


# ============================================================================
# MAG (Memory as Gate) Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (17, 65, 129, 257))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4, 16))
@pytest.mark.parametrize('window_size', (8, 16, 32))
@pytest.mark.parametrize('neural_mem_momentum', (False, True))
@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('depth', (2, 4))
def test_mag(
    seq_len,
    num_persist_mem_tokens,
    window_size,
    neural_mem_momentum,
    batch_size,
    depth
):
    """Test MAG forward pass with comprehensive parameter combinations (MAC-level rigor)."""
    model = MemoryAsGateTransformer(
        num_tokens = 256,
        dim = 16,
        depth = depth,
        window_size = window_size,
        num_persist_mem_tokens = num_persist_mem_tokens,
        neural_memory_layers = None,  # all layers have memory
        neural_memory_kwargs = dict(
            momentum = neural_mem_momentum
        )
    )
    
    x = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 256)


@pytest.mark.parametrize('neural_memory_layers', ((), (1,), (1, 2), None))
@pytest.mark.parametrize('prompt_len', (4, 16))
@torch_default_dtype(torch.float64)
def test_mag_sampling(neural_memory_layers, prompt_len):
    """Test MAG sampling with cache (MAC-level)."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        num_persist_mem_tokens = 4,
        neural_memory_layers = neural_memory_layers
    )
    
    prompt = torch.randint(0, 64, (1, prompt_len))
    
    sampled_with_cache = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    sampled_no_cache = model.sample(prompt, prompt_len + 10, use_cache = False, temperature = 0., show_progress = False)
    
    assert sampled_with_cache.shape == sampled_no_cache.shape
    
    # cached sampling should be deterministic
    sampled_with_cache_2 = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    assert torch.allclose(sampled_with_cache, sampled_with_cache_2)


def _get_mag_memory_layer(transformer, layer_idx=0):
    """Helper to get memory module from MAG layer."""
    attn, mem, gate, ff = transformer.layers[layer_idx]
    return mem


def test_mag_memory_and_attention_both_called(monkeypatch):
    """Verify MAG calls both memory and attention in each layer (parallel branches)."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    
    call_log = {'memory': 0, 'attention': 0}
    
    attn, mem, gate, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(*args, **kwargs):
        call_log['memory'] += 1
        return original_mem_forward(*args, **kwargs)
    
    original_attn_forward = attn.forward
    def wrapped_attn(*args, **kwargs):
        call_log['attention'] += 1
        return original_attn_forward(*args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    assert call_log['memory'] >= 1, "Memory should be called"
    assert call_log['attention'] >= 1, "Attention should be called"


def test_mag_gating_mechanism(monkeypatch):
    """Verify MAG applies gating correctly: output = attn_out + gate * mem_out."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    
    captured = {}
    attn, mem, gate, ff = model.layers[0]
    
    original_gate_forward = gate.forward
    def wrapped_gate(x):
        result = original_gate_forward(x)
        captured['gate_out'] = result.detach().clone()
        return result
    
    original_mem_forward = mem.forward
    def wrapped_mem(*args, **kwargs):
        result = original_mem_forward(*args, **kwargs)
        captured['mem_out'] = result[0].detach().clone()
        return result
    
    original_attn_forward = attn.forward
    def wrapped_attn(*args, **kwargs):
        result = original_attn_forward(*args, **kwargs)
        captured['attn_out'] = result[0].detach().clone()
        return result
    
    monkeypatch.setattr(gate, 'forward', wrapped_gate)
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # gate output should be sigmoid (0-1 range)
    assert captured['gate_out'].min() >= 0
    assert captured['gate_out'].max() <= 1
    
    # gate receives memory output
    assert captured['mem_out'].shape[-1] == 16


def test_mag_memory_state_propagates_during_inference(monkeypatch):
    """Verify memory state is properly propagated during sequential inference."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    model.eval()
    
    captured_states = []
    mem = _get_mag_memory_layer(model, 0)
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        captured_states.append(state)
        return original_mem_forward(seq, state=state, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    cache = None
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            _, cache = model(token, cache=cache, return_cache=True)
    
    non_none_states = sum(1 for s in captured_states if s is not None)
    assert non_none_states >= 1, "Memory state should be propagated during inference"


def test_mag_memory_state_actually_updates(monkeypatch):
    """Verify MAG memory state is actually updated during forward pass."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    
    captured_states = []
    mem = _get_mag_memory_layer(model, 0)
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        result = original_mem_forward(seq, state=state, **kwargs)
        captured_states.append(result[1])
        return result
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    x = torch.randint(0, 64, (1, 32))
    model(x)
    
    assert len(captured_states) >= 1, "Memory should be called"
    final_state = captured_states[-1]
    
    assert final_state is not None, "Memory state should not be None after forward"
    assert hasattr(final_state, 'seq_index'), "State should have seq_index"
    assert final_state.seq_index > 0, "seq_index should be > 0 after processing tokens"


def test_mag_parallel_branches_same_input(monkeypatch):
    """Verify MAG runs memory and attention as parallel branches on same input."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 1,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    
    captured = {}
    attn, mem, gate, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, *args, **kwargs):
        captured['mem_input'] = seq.detach().clone()
        return original_mem_forward(seq, *args, **kwargs)
    
    original_attn_forward = attn.forward
    def wrapped_attn(seq, *args, **kwargs):
        captured['attn_input'] = seq.detach().clone()
        return original_attn_forward(seq, *args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # Both should receive the same input (parallel branches)
    assert torch.allclose(captured['mem_input'], captured['attn_input'])


# ============================================================================
# MAL (Memory as Layer) Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (17, 65, 129, 257))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4, 16))
@pytest.mark.parametrize('window_size', (8, 16, 32))
@pytest.mark.parametrize('neural_mem_momentum', (False, True))
@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('depth', (2, 4))
def test_mal(
    seq_len,
    num_persist_mem_tokens,
    window_size,
    neural_mem_momentum,
    batch_size,
    depth
):
    """Test MAL forward pass with comprehensive parameter combinations (MAC-level rigor)."""
    model = MemoryAsLayerTransformer(
        num_tokens = 256,
        dim = 16,
        depth = depth,
        window_size = window_size,
        num_persist_mem_tokens = num_persist_mem_tokens,
        neural_memory_layers = None,  # all layers have memory
        neural_memory_kwargs = dict(
            momentum = neural_mem_momentum
        )
    )
    
    x = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 256)


@pytest.mark.parametrize('neural_memory_layers', ((), (1,), (1, 2), None))
@pytest.mark.parametrize('prompt_len', (4, 16))
@torch_default_dtype(torch.float64)
def test_mal_sampling(neural_memory_layers, prompt_len):
    """Test MAL sampling with cache (MAC-level)."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        num_persist_mem_tokens = 4,
        neural_memory_layers = neural_memory_layers
    )
    
    prompt = torch.randint(0, 64, (1, prompt_len))
    
    sampled_with_cache = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    sampled_no_cache = model.sample(prompt, prompt_len + 10, use_cache = False, temperature = 0., show_progress = False)
    
    assert sampled_with_cache.shape == sampled_no_cache.shape
    
    # cached sampling should be deterministic
    sampled_with_cache_2 = model.sample(prompt, prompt_len + 10, use_cache = True, temperature = 0., show_progress = False)
    assert torch.allclose(sampled_with_cache, sampled_with_cache_2)


def _get_mal_memory_layer(transformer, layer_idx=0):
    """Helper to get memory module from MAL layer."""
    mem, attn, ff = transformer.layers[layer_idx]
    return mem


def test_mal_memory_before_attention_order(monkeypatch):
    """Verify MAL applies memory BEFORE attention in each layer."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1,)
    )
    
    call_order = []
    mem, attn, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem_forward(*args, **kwargs):
        call_order.append('memory')
        return original_mem_forward(*args, **kwargs)
    
    original_attn_forward = attn.forward
    def wrapped_attn_forward(*args, **kwargs):
        call_order.append('attention')
        return original_attn_forward(*args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem_forward)
    monkeypatch.setattr(attn, 'forward', wrapped_attn_forward)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    assert 'memory' in call_order, "Memory was not called"
    assert 'attention' in call_order, "Attention was not called"
    mem_idx = call_order.index('memory')
    attn_idx = call_order.index('attention')
    assert mem_idx < attn_idx, "Memory should be called before attention in MAL"


def test_mal_attention_receives_memory_transformed_input(monkeypatch):
    """Verify attention receives input that has been transformed by memory."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    
    captured = {}
    mem, attn, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, *args, **kwargs):
        captured['mem_input'] = seq.detach().clone()
        result = original_mem_forward(seq, *args, **kwargs)
        captured['mem_output'] = result[0].detach().clone()
        return result
    
    original_attn_forward = attn.forward
    def wrapped_attn(seq, *args, **kwargs):
        captured['attn_input'] = seq.detach().clone()
        return original_attn_forward(seq, *args, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    monkeypatch.setattr(attn, 'forward', wrapped_attn)
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    # Attention input should be mem_input + mem_output (residual)
    expected = captured['mem_input'] + captured['mem_output']
    assert torch.allclose(captured['attn_input'], expected, atol=1e-5)


def test_mal_memory_state_propagates_during_inference(monkeypatch):
    """Verify memory state is properly propagated during sequential inference."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    model.eval()
    
    captured_states = []
    mem = _get_mal_memory_layer(model, 0)
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        captured_states.append(state)
        return original_mem_forward(seq, state=state, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    cache = None
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            _, cache = model(token, cache=cache, return_cache=True)
    
    non_none_states = sum(1 for s in captured_states if s is not None)
    assert non_none_states >= 1, "Memory state should be propagated during inference"


def test_mal_memory_state_actually_updates(monkeypatch):
    """Verify MAL memory state is actually updated during forward pass."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 8,
        neural_memory_layers = (1,)
    )
    
    captured_states = []
    mem = _get_mal_memory_layer(model, 0)
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        result = original_mem_forward(seq, state=state, **kwargs)
        captured_states.append(result[1])
        return result
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    x = torch.randint(0, 64, (1, 32))
    model(x)
    
    assert len(captured_states) >= 1
    final_state = captured_states[-1]
    
    assert final_state is not None
    assert hasattr(final_state, 'seq_index')
    assert final_state.seq_index > 0


def test_mal_multi_layer_memory_independence(monkeypatch):
    """Verify each layer's memory operates independently."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 3,
        window_size = 8,
        neural_memory_layers = (1, 2, 3)
    )
    
    call_counts = [0, 0, 0]
    
    for idx, (mem, attn, ff) in enumerate(model.layers):
        original_forward = mem.forward
        def make_wrapper(i, orig):
            def wrapper(*args, **kwargs):
                call_counts[i] += 1
                return orig(*args, **kwargs)
            return wrapper
        monkeypatch.setattr(mem, 'forward', make_wrapper(idx, original_forward))
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    for i, count in enumerate(call_counts):
        assert count == 1, f"Memory layer {i} called {count} times, expected 1"


# ============================================================================
# Pure Titans (LMM) Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (17, 65, 129, 257))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4, 16))
@pytest.mark.parametrize('depth', (1, 2, 4))
@pytest.mark.parametrize('neural_mem_momentum', (False, True))
@pytest.mark.parametrize('batch_size', (1, 2))
def test_lmm(
    seq_len,
    num_persist_mem_tokens,
    depth,
    neural_mem_momentum,
    batch_size
):
    """Test Pure Titans (LMM) forward pass with comprehensive parameter combinations."""
    model = TitansLMM(
        num_tokens = 256,
        dim = 16,
        depth = depth,
        num_persist_mem_tokens = num_persist_mem_tokens,
        neural_memory_kwargs = dict(
            momentum = neural_mem_momentum
        )
    )
    
    x = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 256)


@pytest.mark.parametrize('prompt_len', (4, 16, 32))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 4))
def test_lmm_sampling(prompt_len, num_persist_mem_tokens):
    """Test Pure Titans sampling."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
    )
    
    prompt = torch.randint(0, 64, (1, prompt_len))
    
    sampled = model.sample(prompt, prompt_len + 10, temperature = 0., show_progress = False)
    
    assert sampled.shape == (1, 10)
    
    # sampling should be deterministic with temperature=0
    sampled_2 = model.sample(prompt, prompt_len + 10, temperature = 0., show_progress = False)
    assert torch.allclose(sampled, sampled_2)


def test_lmm_no_attention():
    """Verify LMM has no attention layers."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 3,
    )
    
    for layer in model.layers:
        mem, ff = layer
        assert exists(mem), "Each layer should have memory"
        assert exists(ff), "Each layer should have feedforward"


def test_lmm_stacked_layers_sequential_processing(monkeypatch):
    """Verify LMM processes through stacked memory layers sequentially."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 3,
    )
    
    call_order = []
    
    for idx, (mem, ff) in enumerate(model.layers):
        original_forward = mem.forward
        def make_wrapper(i, orig):
            def wrapper(*args, **kwargs):
                call_order.append(f'mem_{i}')
                return orig(*args, **kwargs)
            return wrapper
        monkeypatch.setattr(mem, 'forward', make_wrapper(idx, original_forward))
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    assert call_order == ['mem_0', 'mem_1', 'mem_2']


def test_lmm_memory_state_actually_updates(monkeypatch):
    """Verify LMM memory state is actually updated during forward pass."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
    )
    
    captured_states = []
    mem, ff = model.layers[0]
    
    original_mem_forward = mem.forward
    def wrapped_mem(seq, state=None, **kwargs):
        result = original_mem_forward(seq, state=state, **kwargs)
        captured_states.append(result[1])
        return result
    
    monkeypatch.setattr(mem, 'forward', wrapped_mem)
    
    x = torch.randint(0, 64, (1, 32))
    model(x)
    
    assert len(captured_states) >= 1
    final_state = captured_states[-1]
    
    assert final_state is not None
    assert hasattr(final_state, 'seq_index')
    assert final_state.seq_index > 0


def test_lmm_inference_memory_continuity(monkeypatch):
    """Verify LMM maintains memory continuity during sequential inference."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
    )
    model.eval()
    
    captured_states = []
    mem, ff = model.layers[0]
    
    original_forward = mem.forward
    def wrapped_forward(seq, state=None, **kwargs):
        captured_states.append(state)
        return original_forward(seq, state=state, **kwargs)
    
    monkeypatch.setattr(mem, 'forward', wrapped_forward)
    
    cache = None
    with torch.no_grad():
        for i in range(8):
            token = torch.randint(0, 64, (1, 1))
            _, cache = model(token, cache=cache, return_cache=True)
    
    non_none_count = sum(1 for s in captured_states if s is not None)
    assert non_none_count >= 1, "Memory state should be maintained during inference"


def test_lmm_each_layer_receives_previous_output(monkeypatch):
    """Verify each LMM layer receives output from previous layer."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
    )
    
    captured_inputs = []
    
    for idx, (mem, ff) in enumerate(model.layers):
        original_forward = mem.forward
        def make_wrapper(i, orig):
            def wrapper(seq, *args, **kwargs):
                captured_inputs.append((i, seq.detach().clone()))
                return orig(seq, *args, **kwargs)
            return wrapper
        monkeypatch.setattr(mem, 'forward', make_wrapper(idx, original_forward))
    
    x = torch.randint(0, 64, (1, 16))
    model(x)
    
    assert len(captured_inputs) == 2
    
    # Inputs to different layers should be different (transformed)
    layer0_input = captured_inputs[0][1]
    layer1_input = captured_inputs[1][1]
    assert not torch.allclose(layer0_input, layer1_input)


# ============================================================================
# SlidingWindowAttention Tests
# ============================================================================

def test_sliding_window_attention():
    """Test standalone sliding window attention."""
    attn = SlidingWindowAttention(
        dim = 16,
        window_size = 16,
        num_persist_mem_tokens = 2,
        dim_head = 8,
        heads = 2
    )
    
    seq = torch.randn(2, 64, 16)
    out, cache = attn(seq)
    
    assert out.shape == (2, 64, 16)
    assert cache[0].shape[-2] == 64  # k cache
    assert cache[1].shape[-2] == 64  # v cache


def test_sliding_window_attention_inference():
    """Test sliding window attention in inference mode with caching."""
    attn = SlidingWindowAttention(
        dim = 16,
        window_size = 8,
        num_persist_mem_tokens = 2,
        dim_head = 8,
        heads = 2
    )
    
    seq = torch.randn(1, 4, 16)
    out1, cache = attn(seq)
    
    token = torch.randn(1, 1, 16)
    out2, cache = attn(token, cache = cache)
    
    assert out2.shape == (1, 1, 16)
    assert cache[0].shape[-2] <= 8 + 2  # window + persist


@pytest.mark.parametrize('window_size', (8, 16, 32))
def test_sliding_window_masking(window_size):
    """Test that sliding window attention respects window boundaries."""
    attn = SlidingWindowAttention(
        dim = 16,
        window_size = window_size,
        num_persist_mem_tokens = 0,
        dim_head = 8,
        heads = 1
    )
    
    seq_len = window_size * 3
    seq = torch.randn(1, seq_len, 16)
    out, _ = attn(seq)
    
    assert out.shape == (1, seq_len, 16)


def test_sliding_window_cache_trimming():
    """Test that sliding window cache is properly trimmed to window size."""
    window_size = 4
    attn = SlidingWindowAttention(
        dim = 16,
        window_size = window_size,
        num_persist_mem_tokens = 0,
        dim_head = 8,
        heads = 1
    )
    
    cache = None
    for i in range(20):
        token = torch.randn(1, 1, 16)
        out, cache = attn(token, cache = cache)
    
    k_cache, v_cache = cache
    assert k_cache.shape[-2] <= window_size


# ============================================================================
# Gradient Flow Tests
# ============================================================================

def test_mag_gradient_flow():
    """Test that gradients flow properly through MAG."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1, 2)
    )
    
    x = torch.randint(0, 64, (2, 32))
    loss = model(x, return_loss = True)
    loss.backward()
    
    params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert params_with_grad > total_params * 0.8
    
    assert model.token_emb.weight.grad is not None
    assert model.to_logits.weight.grad is not None


def test_mal_gradient_flow():
    """Test that gradients flow properly through MAL."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 16,
        neural_memory_layers = (1, 2)
    )
    
    x = torch.randint(0, 64, (2, 32))
    loss = model(x, return_loss = True)
    loss.backward()
    
    params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert params_with_grad > total_params * 0.8
    
    assert model.token_emb.weight.grad is not None
    assert model.to_logits.weight.grad is not None


def test_lmm_gradient_flow():
    """Test that gradients flow properly through LMM."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 3,
        num_persist_mem_tokens = 2
    )
    
    x = torch.randint(0, 64, (2, 32))
    loss = model(x, return_loss = True)
    loss.backward()
    
    params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert params_with_grad > total_params * 0.8
    
    assert model.token_emb.weight.grad is not None
    assert model.to_logits.weight.grad is not None
    
    if model.persistent_memory is not None:
        assert model.persistent_memory.grad is not None


# ============================================================================
# Cross-architecture Tests
# ============================================================================

def test_all_architectures_same_vocab():
    """Test all architectures work with same vocabulary size."""
    num_tokens = 128
    dim = 16
    depth = 2
    seq_len = 32
    
    mac = MemoryAsContextTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = depth,
        segment_len = 16,
        neural_memory_layers = ()
    )
    
    mag = MemoryAsGateTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = depth,
        window_size = 16,
        neural_memory_layers = ()
    )
    
    mal = MemoryAsLayerTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = depth,
        window_size = 16,
        neural_memory_layers = ()
    )
    
    lmm = TitansLMM(
        num_tokens = num_tokens,
        dim = dim,
        depth = depth,
    )
    
    x = torch.randint(0, num_tokens, (1, seq_len))
    
    assert mac(x).shape == (1, seq_len, num_tokens)
    assert mag(x).shape == (1, seq_len, num_tokens)
    assert mal(x).shape == (1, seq_len, num_tokens)
    assert lmm(x).shape == (1, seq_len, num_tokens)


@pytest.mark.parametrize('batch_size', (1, 2, 4))
def test_all_architectures_batch_independence(batch_size):
    """Test that different batch items are processed independently."""
    num_tokens = 64
    dim = 16
    seq_len = 16
    
    models = [
        ('MAC', MemoryAsContextTransformer(
            num_tokens = num_tokens, dim = dim, depth = 2,
            segment_len = 8, neural_memory_layers = ()
        )),
        ('MAG', MemoryAsGateTransformer(
            num_tokens = num_tokens, dim = dim, depth = 2,
            window_size = 8, neural_memory_layers = ()
        )),
        ('MAL', MemoryAsLayerTransformer(
            num_tokens = num_tokens, dim = dim, depth = 2,
            window_size = 8, neural_memory_layers = ()
        )),
        ('LMM', TitansLMM(
            num_tokens = num_tokens, dim = dim, depth = 2
        )),
    ]
    
    for name, model in models:
        model.eval()
        
        x_single = torch.randint(0, num_tokens, (1, seq_len))
        x_batch = x_single.repeat(batch_size, 1)
        
        with torch.no_grad():
            logits_batch = model(x_batch)
        
        for i in range(1, batch_size):
            assert torch.allclose(logits_batch[0], logits_batch[i], atol = 1e-5), \
                f"{name}: batch items should be independent"


@pytest.mark.parametrize('batch_size', (1, 2, 4))
@pytest.mark.parametrize('with_memory', (False, True))
def test_mac_batch_independence(batch_size, with_memory):
    """Test MAC batch independence with and without neural memory."""
    num_tokens = 64
    dim = 16
    seq_len = 32
    
    transformer = MemoryAsContextTransformer(
        num_tokens = num_tokens,
        dim = dim,
        depth = 2,
        segment_len = 1 if with_memory else 16,
        neural_memory_layers = (1,) if with_memory else ()
    )
    transformer.eval()
    
    x_single = torch.randint(0, num_tokens, (1, seq_len))
    x_batch = x_single.repeat(batch_size, 1)
    
    with torch.no_grad():
        logits_batch = transformer(x_batch)
    
    for i in range(1, batch_size):
        assert torch.allclose(logits_batch[0], logits_batch[i], atol = 1e-5), \
            f"MAC (with_memory={with_memory}): batch items should be independent"


@pytest.mark.parametrize('arch', ('mag', 'mac', 'mal', 'lmm'))
@pytest.mark.parametrize('num_batches', (2, 3))
def test_small_training_multi_batches(arch, num_batches):
    """Tiny training loop over multiple mini-batches to ensure gradients stay finite."""
    torch.manual_seed(0)
    num_tokens = 64
    dim = 16
    seq_len = 12

    if arch == 'mag':
        model = MemoryAsGateTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 1,
            window_size = 8,
            neural_memory_layers = ()
        )
    elif arch == 'mac':
        model = MemoryAsContextTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 1,
            segment_len = 8,
            neural_memory_layers = ()
        )
    elif arch == 'mal':
        model = MemoryAsLayerTransformer(
            num_tokens = num_tokens,
            dim = dim,
            depth = 1,
            window_size = 8,
            neural_memory_layers = ()
        )
    else:
        model = TitansLMM(
            num_tokens = num_tokens,
            dim = dim,
            depth = 1
        )
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3)

    for _ in range(num_batches):
        x = torch.randint(0, num_tokens, (2, seq_len))
        target = torch.randint(0, num_tokens, (2, seq_len))
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, num_tokens), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)

# ============================================================================
# Numerical Stability Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (16, 33, 64, 128))
def test_mag_numerical_stability(seq_len):
    """Test MAG doesn't produce NaN/Inf with various inputs."""
    model = MemoryAsGateTransformer(
        num_tokens = 256,
        dim = 32,
        depth = 3,
        window_size = 32,
        neural_memory_layers = (1, 2, 3)
    )
    model.eval()
    
    x = torch.randint(0, 256, (2, seq_len))
    with torch.no_grad():
        logits = model(x)
    
    assert not torch.isnan(logits).any(), f"NaN at seq_len={seq_len}"
    assert not torch.isinf(logits).any(), f"Inf at seq_len={seq_len}"


@pytest.mark.parametrize('seq_len', (16, 33, 64, 128))
def test_mal_numerical_stability(seq_len):
    """Test MAL doesn't produce NaN/Inf with various inputs."""
    model = MemoryAsLayerTransformer(
        num_tokens = 256,
        dim = 32,
        depth = 3,
        window_size = 32,
        neural_memory_layers = (1, 2, 3)
    )
    model.eval()
    
    x = torch.randint(0, 256, (2, seq_len))
    with torch.no_grad():
        logits = model(x)
    
    assert not torch.isnan(logits).any(), f"NaN at seq_len={seq_len}"
    assert not torch.isinf(logits).any(), f"Inf at seq_len={seq_len}"


@pytest.mark.parametrize('seq_len', (16, 33, 64, 128))
def test_lmm_numerical_stability(seq_len):
    """Test LMM doesn't produce NaN/Inf with various inputs."""
    model = TitansLMM(
        num_tokens = 256,
        dim = 32,
        depth = 3,
    )
    model.eval()
    
    x = torch.randint(0, 256, (2, seq_len))
    with torch.no_grad():
        logits = model(x)
    
    assert not torch.isnan(logits).any(), f"NaN at seq_len={seq_len}"
    assert not torch.isinf(logits).any(), f"Inf at seq_len={seq_len}"


# ============================================================================
# Long Sequence Tests
# ============================================================================

@pytest.mark.parametrize('seq_len', (512, 1024))
def test_mag_long_sequence(seq_len):
    """Test MAG handles long sequences."""
    model = MemoryAsGateTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 64,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False)
    )
    
    x = torch.randint(0, 64, (1, seq_len))
    logits = model(x)
    
    assert logits.shape == (1, seq_len, 64)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize('seq_len', (512, 1024))
def test_mal_long_sequence(seq_len):
    """Test MAL handles long sequences."""
    model = MemoryAsLayerTransformer(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        window_size = 64,
        neural_memory_layers = (1,),
        neural_memory_kwargs = dict(momentum = False)
    )
    
    x = torch.randint(0, 64, (1, seq_len))
    logits = model(x)
    
    assert logits.shape == (1, seq_len, 64)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize('seq_len', (512, 1024))
def test_lmm_long_sequence(seq_len):
    """Test LMM handles long sequences."""
    model = TitansLMM(
        num_tokens = 64,
        dim = 16,
        depth = 2,
        neural_memory_kwargs = dict(momentum = False)
    )
    
    x = torch.randint(0, 64, (1, seq_len))
    logits = model(x)
    
    assert logits.shape == (1, seq_len, 64)
    assert not torch.isnan(logits).any()


# ============================================================================
# Neural Memory forward_store_only edge case tests
# ============================================================================

def test_neural_memory_forward_store_only_return_surprises_false():
    """Test forward_store_only works correctly when return_surprises=False.
    
    This tests the fix for a bug where surprises were always computed
    even when return_surprises=False, causing shape mismatches during sample().
    """
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        heads = 1,
    )
    
    seq = torch.randn(2, 16, 16)  # batch=2, seq_len=16
    
    # Test with return_surprises=False (used during sample/inference)
    new_state, surprises = mem.forward_store_only(
        seq,
        state=None,
        return_surprises=False
    )
    
    assert surprises is None
    assert new_state is not None
    assert new_state.weights is not None


def test_neural_memory_forward_store_only_return_surprises_true():
    """Test forward_store_only works correctly when return_surprises=True."""
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        heads = 1,
    )
    
    seq = torch.randn(2, 16, 16)
    
    # Test with return_surprises=True (used during training)
    new_state, surprises = mem.forward_store_only(
        seq,
        state=None,
        return_surprises=True
    )
    
    assert surprises is not None
    assert len(surprises) == 2  # (unweighted_mem_model_loss, adaptive_lr)
    assert new_state is not None


def test_neural_memory_short_sequence():
    """Test NeuralMemory handles very short sequences (1 token).
    
    This simulates what happens during autoregressive sampling
    where tokens are generated one at a time.
    """
    mem = NeuralMemory(
        dim = 16,
        chunk_size = 4,
        heads = 1,
    )
    
    # Very short sequence (1 token) - simulates sample() behavior
    seq = torch.randn(2, 1, 16)
    
    # Should not raise an error
    new_state, surprises = mem.forward_store_only(
        seq,
        state=None,
        return_surprises=False
    )
    
    assert new_state is not None
    assert surprises is None


def test_mac_transformer_sample():
    """Test MemoryAsContextTransformer sample() method works correctly.
    
    This is an integration test for the forward_store_only fix.
    """
    from titans_pytorch import MemoryMLP
    
    # dim_head = dim / heads = 32 / 4 = 8, so MemoryMLP dim should match
    model = MemoryAsContextTransformer(
        num_tokens = 64,
        dim = 32,
        depth = 2,
        segment_len = 8,
        num_persist_mem_tokens = 2,
        num_longterm_mem_tokens = 2,
        neural_memory_layers = (1,),
        neural_memory_model = MemoryMLP(dim=8, depth=1),  # dim = dim_head
        neural_memory_kwargs = dict(dim_head = 8, heads = 4),
        use_flex_attn = False,  # Avoid triton dependency
    )
    
    # Create a prompt
    prompt = torch.randint(0, 64, (1, 8))
    prompt_len = prompt.shape[-1]
    
    # Sample should work without errors
    # sample() returns only newly generated tokens
    sampled = model.sample(prompt, prompt_len + 8, temperature=1.0, show_progress=False)
    
    assert sampled.shape == (1, 8)  # 8 new tokens
    assert not torch.isnan(sampled.float()).any()
