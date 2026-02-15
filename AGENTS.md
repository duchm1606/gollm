# AGENTS.md

This file defines the project structure, conventions, and best practices for AI agents working on **gollm**.

## Project Overview

**gollm** is an educational LLM implementation with:

- **Go** for inference (pure Go, minimal dependencies)
- **Python** for training (PyTorch-based)
- **Goal**: Learn by building from scratch, following the LLMs-from-scratch book

## Project Structure

```
gollm/
├── go/                          # Go inference implementation
│   ├── cmd/
│   │   └── inference/          # CLI tool for text generation
│   │       └── main.go
│   ├── pkg/
│   │   ├── tensor/             # Tensor operations (pure Go)
│   │   │   └── tensor.go
│   │   ├── tokenizer/          # BPE tokenization
│   │   │   └── tokenizer.go
│   │   ├── model/              # Transformer architecture
│   │   │   ├── attention.go    # GQA implementation
│   │   │   ├── feedforward.go  # SwiGLU implementation
│   │   │   ├── norm.go         # RMSNorm implementation
│   │   │   ├── rope.go         # RoPE implementation
│   │   │   ├── block.go        # Transformer block
│   │   │   └── model.go        # Complete LLaMA model
│   │   └── sampler/            # Sampling strategies
│   │       └── sampler.go
│   └── go.mod
├── scripts/                      # Python training scripts
│   ├── train.py                # Main training script
│   ├── model.py                # PyTorch model (mirror of Go)
│   └── export.py               # Export weights to Go format
├── configs/                     # Model configurations
│   └── gollm-85m.yaml
├── notes/                       # Personal study notes (by chapter)
│   ├── ch02/                   # Tokenization notes
│   ├── ch03/                   # Attention notes
│   ├── ch04/                   # Transformer notes
│   └── ch05/                   # Training notes
├── .docs/                       # Internal references (DO NOT COMMIT)
│   ├── LLMs-from-scratch/      # Book code
│   ├── nanochat/               # Reference implementation
│   └── llm.c/                  # C reference
├── README.md
└── AGENTS.md                   # This file
```

## Architecture Decisions

### Model Architecture (LLaMA 3 Style)

We implement modern LLaMA architecture with these key components:

1. **RoPE** (Rotary Position Embeddings)
   - Replaces learned positional embeddings
   - Applied to queries and keys in attention
   - See: `.docs/LLMs-from-scratch/pkg/llms_from_scratch/llama3.py` for reference

2. **RMSNorm** (Root Mean Square Layer Normalization)
   - Normalizes inputs to attention and feedforward layers
   - No learnable bias, just scale parameter
   - See: `ch05/07_gpt_to_llama/` for conversion guide

3. **Grouped Query Attention (GQA)**
   - Fewer key/value heads than query heads
   - Reduces memory during inference
   - See: `ch04/04_gqa/` for implementation

4. **SwiGLU Feedforward**
   - Two linear projections gated by SiLU activation
   - `SiLU(W_gate @ x) * (W_up @ x)` → W_down
   - See: Llama3.py reference for exact formula

### Target Model Specs

| Parameter       | Value  | Notes                    |
| --------------- | ------ | ------------------------ |
| Layers          | 12     | Start here, can scale up |
| Embedding dim   | 768    | emb_dim                  |
| Attention heads | 12     | n_heads                  |
| KV heads        | 4      | n_kv_groups              |
| Head dim        | 64     | emb_dim / n_heads        |
| Context length  | 2048   | context_length           |
| Vocab size      | 32,768 | vocab_size (tiktoken)    |
| Hidden dim      | 2048   | ~8/3 \* emb_dim          |
| Parameters      | ~85M   | Manageable for learning  |

## Coding Conventions

### Go Conventions

1. **Package Structure**
   - One concept per package (tensor, tokenizer, model, sampler)
   - Clear separation between public (exported) and private functions
   - Use interfaces where appropriate for testability

2. **Error Handling**
   - Always return errors, don't panic in library code
   - Panic only in truly exceptional cases (programmer errors)

   ```go
   // Good
   func (t *Tensor) Matmul(other *Tensor) (*Tensor, error) {
       if !canMultiply(t.Shape, other.Shape) {
           return nil, fmt.Errorf("incompatible shapes: %v and %v", t.Shape, other.Shape)
       }
       // ... implementation
   }

   // Bad - avoid panics in library code
   if !canMultiply(t.Shape, other.Shape) {
       panic("incompatible shapes")
   }
   ```

3. **Comments**
   - Export all public functions/types with godoc comments
   - Include shape information in tensor operation comments

   ```go
   // GroupedQueryAttention implements GQA as described in Llama 3.
   //
   // Shapes:
   //   - Input: (batch_size, seq_len, emb_dim)
   //   - Output: (batch_size, seq_len, emb_dim)
   //   - Q: (batch_size, n_heads, seq_len, head_dim)
   //   - K/V: (batch_size, n_kv_groups, seq_len, head_dim)
   type GroupedQueryAttention struct {
       // ...
   }
   ```

4. **Performance**
   - Avoid premature optimization
   - Prefer clarity over speed initially
   - Use slices and avoid allocations where obvious

5. **Testing**
   - Write tests for all core operations
   - Test against reference implementations (Python)
   - Use table-driven tests

### Python Conventions

1. **Mirror Go Architecture**
   - Python model structure must exactly match Go
   - Same layer names, same parameter shapes
   - This enables weight transfer

2. **Type Hints**
   - Use type hints throughout

   ```python
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       ...
   ```

3. **Documentation**
   - Docstrings for all classes and methods
   - Reference the corresponding Go implementation

## Implementation Phases

Follow the book chapters in order:

### Phase 1: Tokenization (Ch 2)

**Goal**: Working tokenizer

1. Implement BPE tokenizer in Go
2. Must be compatible with tiktoken
3. Support encoding and decoding
4. Handle special tokens (<|begin_of_text|>, etc.)

**Reference**: `.docs/LLMs-from-scratch/ch02/05_bpe-from-scratch/`

**Checkpoint**: Can encode/decode text matching tiktoken output

### Phase 2: Tensor Operations

**Goal**: Basic tensor math

1. N-dimensional tensor type
2. Matrix multiplication
3. Element-wise operations
4. Broadcasting
5. View/reshape operations

**Reference**: `.docs/LLMs-from-scratch/appendix-A/`

**Checkpoint**: Can perform basic neural network operations

### Phase 3: Attention (Ch 3)

**Goal**: Working attention mechanism

1. Self-attention (scaled dot-product)
2. Causal masking
3. Multi-head attention
4. Grouped Query Attention (GQA)

**Reference**: `.docs/LLMs-from-scratch/ch03/`

**Checkpoint**: Attention produces correct output shapes and values

### Phase 4: Model Components (Ch 4-5)

**Goal**: Complete LLaMA architecture

1. RoPE implementation
2. RMSNorm layer
3. SwiGLU feedforward
4. Transformer block
5. Complete model
6. Weight loading from Python

**References**:

- `.docs/LLMs-from-scratch/ch04/`
- `.docs/LLMs-from-scratch/ch05/07_gpt_to_llama/`
- `.docs/LLMs-from-scratch/pkg/llms_from_scratch/llama3.py`

**Checkpoint**: Can load weights and run forward pass

### Phase 5: Training & Inference

**Goal**: End-to-end working system

1. Python training script
2. Weight export format
3. KV-cache implementation
4. Sampling strategies
5. CLI inference tool

**References**:

- `.docs/LLMs-from-scratch/ch05/`
- `.docs/nanochat/nanochat/engine.py`

**Checkpoint**: Can train a model and generate text in Go

## Weight Format

Define a simple binary format for weight transfer:

```
File Structure:
1. Header (magic number, version, metadata)
2. Tensor descriptors (name, shape, dtype, offset)
3. Tensor data (raw bytes)
```

Requirements:

- Little-endian floats (f32)
- Clear naming convention matching PyTorch state_dict
- Versioned for future compatibility

## Testing Strategy

1. **Unit Tests**: Each component in isolation
2. **Integration Tests**: Full forward pass
3. **Numerical Tests**: Compare with PyTorch reference
4. **Shape Tests**: Verify all tensor shapes

Example test pattern:

```go
func TestAttention(t *testing.T) {
    // Setup
    config := Config{...}
    attn := NewGroupedQueryAttention(config)

    // Load reference weights from Python
    loadTestWeights(attn, "test_data/attn_weights.bin")

    // Run
    input := tensor.NewTensor([]int{2, 10, 768}) // batch=2, seq=10
    output, err := attn.Forward(input)

    // Verify
    expected := loadExpectedOutput("test_data/attn_output.bin")
    assertTensorEqual(t, output, expected, 1e-5)
}
```

## Common Pitfalls

1. **RoPE Implementation**
   - Split-halves vs interleaved styles (we use split-halves like HF)
   - Apply to both Q and K
   - Cache cos/sin values for efficiency

2. **GQA Attention**
   - Query heads must be divisible by KV heads
   - Repeat KV heads to match query heads before matmul
   - Shape is critical: (batch, heads, seq, head_dim)

3. **RMSNorm**
   - No bias term, only scale
   - Epsilon typically 1e-5 or 1e-6
   - Normalize before attention/FFN (pre-norm)

4. **Weight Loading**
   - Ensure shapes match exactly
   - Transpose matrices if needed (PyTorch vs Go conventions)
   - Handle tied weights (embedding = output head)

5. **Tokenization**
   - Handle BOS/EOS tokens correctly
   - Special tokens have specific IDs
   - Test roundtrip: decode(encode(text)) == text

## Resources

### Primary References

1. **LLMs-from-scratch book**
   - Main learning resource
   - Located in: `.docs/LLMs-from-scratch/`
   - Start with: `ch04/01_main-chapter-code/gpt.py`
   - Then: `ch05/07_gpt_to_llama/`
   - Reference: `pkg/llms_from_scratch/llama3.py`

2. **nanochat**
   - See: `.docs/nanochat/nanochat/gpt.py`
   - Modern implementation with Flash Attention
   - Good reference for kv-cache

3. **llm.c**
   - See: `.docs/llm.c/train_gpt2.c`
   - Clean C implementation
   - Good for understanding memory layout

### Useful Commands

```bash
# Run Go tests
cd go && go test ./...

# Format Go code
cd go && go fmt ./...

# Run Python training
cd python && python train.py --config ../configs/gollm-85m.yaml

# Test tokenizer
cd go && go run cmd/inference/main.go --test-tokenizer
```

## Questions?

When in doubt:

1. Check the reference implementation in `.docs/`
2. Prefer clarity over cleverness
3. Test against Python reference
4. Document any deviations from standard practice

Remember: This is an educational project. The goal is understanding, not state-of-the-art performance.
