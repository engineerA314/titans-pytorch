<img src="./fig2.png" width="400px"></img>

<img src="./fig1.png" width="400px"></img>

## Titans - Pytorch (Paper-Aligned Fork)

Unofficial implementation of [Titans](https://arxiv.org/abs/2501.00663) in Pytorch.

**Note:** This fork diverges from the original repository to strictly follow the architectures described in the paper. While the original implementation explores experimental variations (like residual memory mixing), this fork aims for architectural correctness based on the paper's specifications. This implementation includes all four variants proposed in the Titans paper:

| Architecture          | Description           | Attention Type       | Memory Integration                    |
| --------------------- | --------------------- | -------------------- | ------------------------------------- |
| **MAC**               | Memory as Context     | Block/Sliding Window | Memory prepended as attention context |
| **MAG**               | Memory as Gate        | Sliding Window       | Memory combined via learned gating    |
| **MAL**               | Memory as Layer       | Sliding Window       | Memory applied before attention       |
| **Pure Titans (LMM)** | Long-term Memory only | None                 | Standalone memory sequence model      |

### Key Architectural Alignments

Below are the critical changes made to align with the MAC architecture:

1.  **Memory as Context (Concat instead of Residual)**

    - **Paper:** The long-term memory $p_L$ serves as context for the current segment.
    - **Implementation:** The retrieved memory is **prepended** to the attention Key/Values as context. It is _not_ added as a residual to the input, and it is _ephemeral_ (not stored in the KV cache).

2.  **Strict Segment-wise Operation**

    - **Retrieval:** Enforced to use only **committed weights** from the previous segment. Uncommitted updates within the current segment are ignored during retrieval to ensure $p_L$ remains fixed and stable for the entire segment.
    - **Storage:** Explicitly separated `forward_store_only` path to handle memory updates independently from retrieval, preserving partial chunks correctly.

3.  **Inference Consistency (Segment Buffering)**

    - **Issue:** Standard token-by-token autoregressive decoding provides only 1 token of information to the memory, degrading performance compared to the training phase where the memory sees a full segment.
    - **Solution:** Implemented **variable-length query buffering**. During inference, the model buffers inputs within the current segment (growing from 1 to `segment_len`). This ensures the memory module receives the same "view" of the segment as it does during training, resetting only at segment boundaries.

4.  **Context-Aware Flex Attention**
    - Updated masking logic to support `flex_attention` even when $p_L$ context is present during training, maintaining training speed without sacrificing architectural correctness.

---

## Appreciation

- [Eryk](https://github.com/sentialx) for sharing his early experimental results with me, positive for 2 layer MLP

## Install

```bash
$ pip install titans-pytorch
```

## Usage

```python
import torch
from titans_pytorch import NeuralMemory

mem = NeuralMemory(
    dim = 384,
    chunk_size = 64 # set to smaller chunk size for better perf on smaller sequence lengths (but more memory usage)
).cuda()

seq = torch.randn(2, 1024, 384).cuda()
retrieved, mem_state = mem(seq)

assert seq.shape == retrieved.shape
```

### MAC (Memory as Context)

Segment-based architecture where memory is prepended as context to attention.

```python
import torch
from titans_pytorch import MemoryAsContextTransformer

transformer = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 2,
    segment_len = 128,              # segment/window size
    num_persist_mem_tokens = 4,
    num_longterm_mem_tokens = 16,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True)
loss.backward()

# after much training
sampled = transformer.sample(token_ids[:, :4], 512)
```

### MAG (Memory as Gate)

Sliding window attention combined with neural memory via gating.

```python
import torch
from titans_pytorch import MemoryAsGateTransformer

transformer = MemoryAsGateTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 2,
    window_size = 64,               # sliding window size
    num_persist_mem_tokens = 4,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True)
loss.backward()

sampled = transformer.sample(token_ids[:, :4], 512)
```

### MAL (Memory as Layer)

Memory applied as a layer before sliding window attention.

```python
import torch
from titans_pytorch import MemoryAsLayerTransformer

transformer = MemoryAsLayerTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 2,
    window_size = 64,               # sliding window size
    num_persist_mem_tokens = 4,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True)
loss.backward()

sampled = transformer.sample(token_ids[:, :4], 512)
```

### Pure Titans (LMM)

Long-term memory module only, without attention.

```python
import torch
from titans_pytorch import TitansLMM

model = TitansLMM(
    num_tokens = 256,
    dim = 256,
    depth = 4,                      # number of memory layers
    num_persist_mem_tokens = 4,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = model(token_ids, return_loss = True)
loss.backward()

sampled = model.sample(token_ids[:, :4], 512)
```

## Experiments

```bash
$ pip install .[examples]
```

Training scripts are provided for each architecture:

```bash
# MAC (Memory as Context)
$ python train_mac.py

# MAG (Memory as Gate)
$ python train_mag.py

# MAL (Memory as Layer)
$ python train_mal.py

# Pure Titans (LMM only)
$ python train_titans.py
```

## Citations

```bibtex
@inproceedings{Behrouz2024TitansLT,
    title   = {Titans: Learning to Memorize at Test Time},
    author  = {Ali Behrouz and Peilin Zhong and Vahab S. Mirrokni},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:275212078}
}
```

```bibtex
@article{Sun2024LearningT,
    title   = {Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
    author  = {Yu Sun and Xinhao Li and Karan Dalal and Jiarui Xu and Arjun Vikram and Genghan Zhang and Yann Dubois and Xinlei Chen and Xiaolong Wang and Oluwasanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.04620},
    url     = {https://api.semanticscholar.org/CorpusID:271039606}
}
```

```bibtex
@inproceedings{Yang2024GatedDN,
    title   = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
    author  = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274598177}
}
```

```bibtex
@inproceedings{Nguyen2024TurningUT,
    title   = {Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs},
    author  = {Minh Nguyen and Andrew Baker and Clement Neo and Allen Roush and Andreas Kirsch and Ravid Shwartz-Ziv},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270870613}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```

```bibtex
@article{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.17897},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@software{Kyrylov_Accelerated_Scan_2024,
    author  = {Kyrylov, Volodymyr},
    doi     = {10.5281/zenodo.10600962},
    title   = {Accelerated Scan},
    version = {0.1.2},
    year    = {2024}
}
```

```bibtex
@misc{wang2025testtimeregressionunifyingframework,
    title   = {Test-time regression: a unifying framework for designing sequence models with associative memory},
    author  = {Ke Alexander Wang and Jiaxin Shi and Emily B. Fox},
    year    = {2025},
    eprint  = {2501.12352},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2501.12352},
}
```

```bibtex
@misc{jordan2024muon,
    author  = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and
                    Franz Cesista and Laker Newhouse and Jeremy Bernstein},
    title   = {Muon: An optimizer for hidden layers in neural networks},
    year    = {2024},
    url     = {https://kellerjordan.github.io/posts/muon/}
}
```

```bibtex
@inproceedings{Zhang2025TestTimeTD,
    title   = {Test-Time Training Done Right},
    author  = {Tianyuan Zhang and Sai Bi and Yicong Hong and Kai Zhang and Fujun Luan and Songlin Yang and Kalyan Sunkavalli and William T. Freeman and Hao Tan},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:279071244}
}
```

```bibtex
@inproceedings{Behrouz2025ATLASLT,
    title  = {ATLAS: Learning to Optimally Memorize the Context at Test Time},
    author = {Ali Behrouz and Ze-Minghui Li and Praneeth Kacham and Majid Daliri and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab S. Mirrokni},
    year   = {2025},
    url    = {https://api.semanticscholar.org/CorpusID:278996373}
}
```
