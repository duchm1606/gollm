# gollm - Go LLM from Scratch

A minimal LLM implementation in Go for inference, with training in Python. This is an educational project implementing modern LLaMA 3 architecture from scratch.

## Project Structure

```
gollm/
├── cmd/
│   └── tokcli/           # CLI tool for tokenizer operations
├── pkg/
│   └── tokenizer/        # BPE tokenizer implementation
├── data/
│   ├── corpus/          # Training text corpora
│   └── tokenizers/      # Trained tokenizer models
├── go.mod               # Go module
├── README.md            # This file
└── AGENTS.md            # Project guidelines
```

## BPE Tokenizer

Implements Byte-Pair Encoding tokenization compatible with LLaMA 3.

### Features

- **Training from scratch**: Build vocabulary from any text corpus
- **LLaMA 3 compatible**: Supports all LLaMA 3 special tokens
- **Save/Load**: LLaMA 3 tokenizer.model format
- **Educational**: Clean, well-documented code

### Quick Start

```bash
# Build the CLI
go build -o tokcli ./cmd/tokcli

# Train a tokenizer
./tokcli train --corpus data/corpus/your_corpus.txt --vocab-size 8192 --output data/tokenizers/my_tokenizer.model

# Encode text
./tokcli encode --tokenizer data/tokenizers/my_tokenizer.model "Hello world"

# Decode tokens
./tokcli decode --tokenizer data/tokenizers/my_tokenizer.model "[128000, 9906, 1917, 128001]"

# Test roundtrip
./tokcli test --tokenizer data/tokenizers/my_tokenizer.model

# Show tokenizer info
./tokcli info --tokenizer data/tokenizers/my_tokenizer.model
```

### Example: Train on "The Verdict"

```bash
# Download "The Verdict" and train
./tokcli train --corpus the_verdict.txt --vocab-size 8000 --output verdict_tokenizer.model
```

### Library Usage

```go
package main

import (
    "gollm/pkg/tokenizer"
)

func main() {
    // Create and train
    tok := tokenizer.NewTokenizer()
    corpus := []string{"Hello world", "Test text"}
    tok.Train(corpus, 1000)
    
    // Encode
    ids, _ := tok.Encode("Hello", tokenizer.EncodeOptions{BOS: true})
    // ids: [128000, 9906, ...]  (BOS + tokens)
    
    // Decode
    text := tok.Decode(ids)
    // text: "<|begin_of_text|>Hello"
    
    // Save
    tok.Save("tokenizer.model")
}
```

## Where to Put Your Corpus

Corpus files should go in `data/corpus/`:

```
data/
├── corpus/
│   ├── the_verdict.txt    # Your training text
│   └── *.txt              # Any other text files
└── tokenizers/
    └── *.model            # Generated tokenizer files
```

### Corpus Format

- Plain text files (UTF-8)
- Paragraphs separated by blank lines
- Each paragraph = one training document

### Example

```bash
# Put your corpus in data/corpus/
cp your_corpus.txt data/corpus/

# Train with recommended paths
./tokcli train \
  --corpus data/corpus/your_corpus.txt \
  --vocab-size 8192 \
  --output data/tokenizers/my_tokenizer.model

# Use it
./tokcli encode \
  --tokenizer data/tokenizers/my_tokenizer.model \
  "Hello world"
```

## Architecture

### BPE Algorithm

1. **Initialize**: 256 byte tokens (0-255)
2. **Preprocess**: Replace spaces with 'Ġ' (word boundary marker)
3. **Count pairs**: Find all adjacent token pairs in corpus
4. **Merge most frequent**: Create new token, apply to corpus
5. **Repeat**: Until target vocabulary size reached

### Vocabulary

```
ID 0-255:      Byte tokens (ASCII/UTF-8 bytes)
ID 256+:       Merged tokens (learned from data)
ID 128000+:    Special tokens
  - 128000:    <|begin_of_text|>
  - 128001:    <|end_of_text|>
  - 128006:    <|start_header_id|>
  - 128007:    <|end_header_id|>
  - 128008:    <|eom_id|> (end of message)
  - 128009:    <|eot_id|> (end of turn)
  - ...        256 reserved special tokens
```

## Testing

```bash
# Run all tests
go test -v ./pkg/tokenizer/

# Run specific tests
go test -v ./pkg/tokenizer/ -run TestRoundtrip
```

## Key Takeaways

1. **BPE is compression**: Starts with bytes, merges frequent pairs
2. **Vocabulary size matters**: Tradeoff between granularity and efficiency
3. **Special tokens**: Handle structure (BOS/EOS) and chat format
4. **Educational focus**: Clean code over performance

## References

- LLaMA 3: https://github.com/meta-llama/llama-models
- LLMs-from-scratch: Book by Sebastian Raschka
- Original BPE: "A New Algorithm for Data Compression" (Gage, 1994)

## License

MIT
