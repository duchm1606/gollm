// Package model provides the transformer model components for LLaMA-3 style architectures.
//
// This package implements the core attention mechanisms including:
//   - CausalSelfAttention: Basic scaled dot-product attention with causal masking
//   - MultiHeadAttention: Parallel attention heads for richer representations
//   - GroupedQueryAttention: Memory-efficient attention with shared K/V heads
//
// All implementations follow the patterns from "LLMs from Scratch" by Sebastian Raschka
// and are designed for educational purposes with pure Go.
package model

import "fmt"

// Config holds the model hyperparameters for a LLaMA-3 style transformer.
// These parameters define the architecture size and behavior.
type Config struct {
	// VocabSize is the size of the token vocabulary (e.g., 32,768 for LLaMA-3)
	VocabSize int

	// ContextLength is the maximum sequence length the model can process (e.g., 2048)
	ContextLength int

	// EmbeddingDim is the dimension of token embeddings (e.g., 768)
	EmbeddingDim int

	// NumHeads is the number of attention heads (e.g., 12)
	NumHeads int

	// NumKVGroups is the number of key-value groups for Grouped Query Attention (e.g., 4)
	// Must divide NumHeads evenly. Smaller values = more memory efficient.
	NumKVGroups int

	// HeadDim is the dimension per attention head (EmbeddingDim / NumHeads, e.g., 64)
	HeadDim int

	// Dropout is the dropout rate for regularization (0.0 for inference)
	Dropout float32

	// UseBias determines if linear layers use bias terms (false for LLaMA-3)
	UseBias bool

	// RoPEBase is the base value for RoPE frequency calculation (10,000 or 500,000)
	RoPEBase float32
}

// DefaultConfig returns a configuration for a small LLaMA-3 style model (~85M parameters)
// suitable for educational purposes and testing.
func DefaultConfig() Config {
	return Config{
		VocabSize:     32768,
		ContextLength: 2048,
		EmbeddingDim:  768,
		NumHeads:      12,
		NumKVGroups:   4,
		HeadDim:       64, // 768 / 12
		Dropout:       0.0,
		UseBias:       false,
		RoPEBase:      10000.0,
	}
}

// Validate checks if the configuration is valid and consistent.
// Returns an error if any parameters are incompatible.
func (c Config) Validate() error {
	if c.EmbeddingDim%c.NumHeads != 0 {
		return fmt.Errorf("embedding_dim (%d) must be divisible by num_heads (%d)",
			c.EmbeddingDim, c.NumHeads)
	}
	if c.NumHeads%c.NumKVGroups != 0 {
		return fmt.Errorf("num_heads (%d) must be divisible by num_kv_groups (%d)",
			c.NumHeads, c.NumKVGroups)
	}
	return nil
}

// GroupSize returns the number of query heads per KV group.
// For example, with 12 heads and 4 groups, group_size = 3.
func (c Config) GroupSize() int {
	return c.NumHeads / c.NumKVGroups
}
