# Self-Attention Mechanism Implementation

Complete implementation of Llama-3 style self-attention mechanisms in pure Go, following the "LLMs from Scratch" book by Sebastian Raschka.

## 📁 Project Structure

```
pkg/
├── tensor/
│   ├── tensor.go           # N-dimensional tensor operations
│   └── tensor_test.go      # Comprehensive tensor tests
├── model/
│   ├── config.go           # Model configuration
│   ├── rope.go             # Rotary Position Embeddings
│   ├── rope_test.go        # RoPE tests
│   ├── kv_cache.go         # Key-Value caching
│   ├── kv_cache_test.go    # KV cache tests
│   ├── testutil/
│   │   └── testutil.go     # Test utilities
│   └── attention/
│       ├── causal.go       # Basic causal self-attention
│       ├── multihead.go    # Multi-head attention
│       ├── grouped.go      # Grouped Query Attention (GQA)
│       ├── attention_test.go    # Unit tests
│       └── integration_test.go  # Python comparison tests

scripts/
└── generate_test_weights.py  # Python reference generator

notes/ch03/
└── attention_explanation.md  # Educational documentation
```

## 🧮 Components Implemented

### 1. Tensor Operations (`pkg/tensor/`)
- **N-dimensional tensor** with shape and strides
- **Core operations**: MatMul, Transpose, View, Softmax
- **Element-wise**: Add, Mul, Scale with broadcasting
- **Advanced**: Slice, Concatenate
- **Comprehensive tests**: 20 test cases, all passing

**Lines**: 1,809 (920 + 889)

### 2. RoPE - Rotary Position Embeddings (`pkg/model/rope.go`)
- **ComputeRoPE()**: Precomputes cos/sin frequencies
- **ApplyRoPE()**: Applies split-halves rotation to Q/K
- **Basic implementation** (no frequency scaling yet)
- **Educational documentation** with examples

**Lines**: 265 + 603 = 868

### 3. Attention Mechanisms (`pkg/model/attention/`)

#### Causal Self-Attention (`causal.go`)
Basic scaled dot-product with causal masking
- Single Q, K, V projection
- Causal mask (upper triangular → -inf)
- Shape: (batch, seq, d_in) → (batch, seq, d_out)

**Lines**: 112

#### Multi-Head Attention (`multihead.go`)
Parallel attention heads for richer representations
- Splits embedding into num_heads parallel computations
- Each head: (batch, seq, head_dim)
- Supports RoPE and masking
- Output projection combines heads

**Lines**: 153

#### Grouped Query Attention (`grouped.go`)
Memory-efficient attention with shared K/V
- **Configuration**: 12 query heads, 4 KV groups
- **Group size**: 3 query heads share 1 K/V head
- **Memory savings**: 67% reduction vs MHA
- **Ideal for**: Long sequences and KV caching

**Lines**: 179 + 372 = 551

### 4. KV Cache (`pkg/model/kv_cache.go`)
Efficient caching for autoregressive generation
- **Complexity**: O(N²) → O(N) per sequence
- **Pre-allocated** tensors for max_length
- **Update()**: Appends new K/V, returns full cache
- **Clear()**: Resets for new generation

**Memory savings calculation**:
```
Without cache: 1 + 2 + 3 + ... + N = O(N²)
With cache: 1 + 1 + 1 + ... + 1 = O(N)
For N=1000: 500,000 vs 1,000 operations
```

**Lines**: 249 + 566 = 815

### 5. Testing Infrastructure (`pkg/model/testutil/` & `scripts/`)

#### Python Reference Generator
- Generates test data using PyTorch
- Fixed seed (42) for reproducibility
- Saves binary tensors and JSON config

#### Test Utilities
- `LoadTensor()`: Load binary tensors from Python
- `TensorsEqual()`: Compare with tolerance (1e-5)
- `PrintTensor()`: Debug output with statistics

#### Integration Tests
Compare Go implementations against PyTorch:
- Causal self-attention
- Multi-head attention
- Grouped query attention
- RoPE application

**Lines**: 376 + 323 + 316 = 1,015

### 6. Documentation (`notes/ch03/`)
Educational guide covering:
1. Self-attention intuition (library analogy)
2. Query, Key, Value concepts
3. Mathematical foundation with formulas
4. Causal masking explanation
5. Multi-head attention visualizations
6. GQA memory savings (67% reduction)
7. RoPE position encoding
8. KV cache efficiency analysis

**Lines**: ~2,000 words with diagrams

## 📊 Statistics Summary

| Component | Code | Tests | Total |
|-----------|------|-------|-------|
| Tensor Operations | 920 | 889 | 1,809 |
| RoPE | 265 | 603 | 868 |
| Attention (3 variants) | 444 | 372 | 816 |
| KV Cache | 249 | 566 | 815 |
| Testing Infrastructure | - | 1,015 | 1,015 |
| Documentation | - | - | ~2,000 words |
| **TOTAL** | **1,878** | **3,445** | **~5,323** |

## 🚀 Usage Example

```go
package main

import (
    "gollm/pkg/model"
    "gollm/pkg/model/attention"
    "gollm/pkg/tensor"
)

func main() {
    // Configuration for 85M parameter model
    cfg := model.DefaultConfig()
    
    // Create GQA attention
    gqa := attention.NewGroupedQueryAttention(cfg)
    
    // Input tensor: (batch=2, seq=10, dim=768)
    x := tensor.NewTensor([]int{2, 10, 768})
    // ... fill with data
    
    // Create causal mask
    mask := attention.CreateCausalMask(10)
    
    // Create RoPE parameters
    rope := model.ComputeRoPE(cfg.HeadDim, cfg.ContextLength, cfg.RoPEBase)
    
    // Forward pass
    output, err := gqa.Forward(x, mask, rope)
    if err != nil {
        panic(err)
    }
    
    // Output shape: (2, 10, 768)
    fmt.Printf("Output shape: %v\n", output.Shape)
}
```

## 🧪 Running Tests

```bash
# Generate Python reference data
python3 scripts/generate_test_weights.py

# Run all tests
go test -v ./pkg/tensor/...
go test -v ./pkg/model/...

# Run specific test
go test -v ./pkg/model/attention -run TestGroupedQueryAttention
```

## 📚 Key Concepts Implemented

### 1. Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- Scaling by √d_k prevents softmax saturation
- Causal mask ensures autoregressive property

### 2. Multi-Head Attention
- 12 parallel attention heads
- Each head learns different patterns
- Increased model capacity

### 3. Grouped Query Attention (GQA)
- 12 query heads, 4 KV groups
- 3 query heads share 1 K/V pair
- **67% memory reduction** in KV cache
- Minimal quality impact

### 4. RoPE (Rotary Position Embeddings)
- Encode position by rotating Q/K vectors
- Split-halves style (not interleaved)
- Generalizes to longer sequences
- No learned parameters

### 5. KV Cache
- Store K/V tensors during generation
- Complexity: O(N²) → O(N)
- Essential for efficient inference
- Memory cost: ~2MB for default config

## 🎯 Testing Strategy

1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Compare with PyTorch reference
3. **Shape Tests**: Verify all tensor shapes
4. **Numerical Tests**: Tolerance 1e-5 for float32
5. **Fixed Seeds**: Reproducible test data (seed=42)

## 📖 References

- "LLMs from Scratch" by Sebastian Raschka
- "Attention Is All You Need" (Vaswani et al., 2017)
- Llama 3 paper (Meta AI, 2024)
- GQA paper (Ainslie et al., 2023)

## 🎓 Educational Value

This implementation prioritizes:
- **Clarity** over performance
- **Educational documentation** with examples
- **Pure Go** (no external dependencies)
- **Comprehensive tests** for reliability
- **Progressive complexity** (Causal → MHA → GQA)

## 🔮 Future Enhancements

- [ ] Advanced RoPE with frequency scaling
- [ ] Flash Attention optimization
- [ ] Quantization support (int8/int4)
- [ ] Multi-GPU support
- [ ] Additional attention variants (MQA, MLA)

## 📄 License

MIT License - Educational project for learning transformers from scratch.

---

**Total Implementation**: ~5,300 lines of code and tests
**Status**: ✅ Complete and tested
**Ready for**: Integration into full transformer model
