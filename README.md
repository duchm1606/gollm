# gollm

A minimal LLM implementation in Go for inference, with Python for training. Inspired by [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch), [nanochat](https://github.com/karpathy/nanochat), and [llm.c](https://github.com/karpathy/llm.c).

## Purpose

**gollm** is a "thin LLM" - a clean, educational implementation that prioritizes:
- **Simplicity**: Readable code following the book's progression
- **Portability**: Pure Go inference engine (no heavy dependencies)
- **Compatibility**: Model weights trained in Python, loaded in Go
- **Modern architecture**: Based on LLaMA 3 design principles

## Architecture

Following the modern LLaMA 3 architecture:

- **RoPE** (Rotary Position Embeddings)
- **RMSNorm** (Root Mean Square Layer Normalization)
- **Grouped Query Attention** (GQA)
- **SwiGLU** feedforward activation
- **Byte-Pair Encoding** (tiktoken-compatible)
- **KV-Cache** for efficient autoregressive generation

## Project Structure

```
.
├── go/                     # Go implementation
│   ├── cmd/inference/     # CLI inference tool
│   └── pkg/
│       ├── tensor/        # Tensor operations
│       ├── tokenizer/     # BPE tokenization
│       ├── model/         # Transformer architecture
│       └── sampler/       # Sampling strategies
├── python/                # Python training scripts
├── configs/               # Model configurations
├── notes/                 # Personal study notes
│   ├── ch02/             # Tokenization notes
│   ├── ch03/             # Attention mechanism notes
│   ├── ch04/             # Transformer architecture notes
│   └── ch05/             # Training notes
└── .docs/                 # Internal reference materials (not for commit)
    ├── nanochat/          # nanochat reference
    ├── llm.c/             # llm.c reference
    └── LLMs-from-scratch/ # Book code & examples
```

## Learning Roadmap

Following the *Build a Large Language Model (From Scratch)* book progression:

### Phase 1: Foundations
**Goal**: Tokenizer and basic model structure

- [ ] BPE Tokenizer (tiktoken-compatible)
- [ ] Data loading and batching
- [ ] Basic tensor operations in Go

**Notes**: `notes/ch02/`

### Phase 2: Attention Mechanisms
**Goal**: Implement modern attention

- [ ] Self-attention mechanism
- [ ] Causal (masked) attention
- [ ] Multi-head attention → Grouped Query Attention (GQA)

**Notes**: `notes/ch03/`

### Phase 3: Transformer Architecture
**Goal**: Complete model implementation

- [ ] RMSNorm layer normalization
- [ ] RoPE (Rotary Position Embeddings)
- [ ] SwiGLU feedforward network
- [ ] Full transformer block
- [ ] Complete LLaMA model

**Notes**: `notes/ch04/`

### Phase 4: Training & Inference
**Goal**: End-to-end training and generation

- [ ] Python training script (mirrors Go architecture)
- [ ] Weight export format (Go-compatible)
- [ ] KV-cache for efficient inference
- [ ] Sampling strategies (greedy, temperature, top-k, top-p)
- [ ] Go CLI inference tool

**Notes**: `notes/ch05/`

### Phase 5: Finetuning (Optional)
**Goal**: Instruction following

- [ ] Supervised finetuning (SFT)
- [ ] Chat format handling
- [ ] Classification finetuning

**Notes**: `notes/ch06/`, `notes/ch07/`

## Target Model

Initial implementation targets a small compute-optimal model:

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Embedding dim | 768 |
| Attention heads | 12 |
| KV heads | 4 (GQA) |
| Context length | 2048 |
| Vocab size | 32,768 |
| **Parameters** | ~85M |

## Dependencies

**Go (inference)**:
- Go 1.23+
- Pure Go implementation (minimal external deps)

**Python (training)**:
- PyTorch 2.0+
- tiktoken (for tokenizer compatibility)

## Usage

### Training (Python)
```bash
python python/train.py --config configs/gollm-85m.yaml --data data/tinystories
```

### Inference (Go)
```bash
go run go/cmd/inference/main.go --model checkpoints/model.bin --prompt "Once upon a time"
```

## References

This project builds on:

1. **LLMs-from-scratch** - The primary learning resource by Sebastian Raschka
2. **nanochat** - Minimal PyTorch LLM training by Andrej Karpathy  
3. **llm.c** - Clean C/CUDA implementation by Andrej Karpathy

*Note: Reference materials are located in `.docs/` (internal folder, not committed)*

## License

MIT

## Contributing

This is an educational project. The goal is to understand LLMs by building them from scratch. Contributions should prioritize clarity and learning over performance optimizations.
