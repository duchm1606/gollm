// Package model provides the transformer model components for GPT-2 architecture.
//
// This package implements the core transformer components following the GPT-2
// architecture from "LLMs from Scratch" by Sebastian Raschka.
//
// GPT-2 Key Features:
//   - LayerNorm with scale (gamma) and shift (beta)
//   - GELU activation
//   - Multi-Head Attention (not GQA)
//   - Learned positional embeddings (not RoPE)
package model

import "fmt"

// GPT2Config holds the model hyperparameters for GPT-2 architecture.
// These parameters define the architecture size and behavior.
type GPT2Config struct {
	// VocabSize is the size of the token vocabulary (50257 for GPT-2)
	VocabSize int

	// ContextLength is the maximum sequence length the model can process (1024 for GPT-2)
	ContextLength int

	// EmbeddingDim is the dimension of token embeddings (768 for GPT-2 124M)
	EmbeddingDim int

	// NumHeads is the number of attention heads (12 for GPT-2 124M)
	NumHeads int

	// NumLayers is the number of transformer blocks (12 for GPT-2 124M)
	NumLayers int

	// HeadDim is the dimension per attention head (EmbeddingDim / NumHeads, e.g., 64)
	HeadDim int

	// HiddenDim is the dimension of the feedforward layer (3072 for GPT-2 124M)
	HiddenDim int

	// Dropout is the dropout rate for regularization (0.1 for GPT-2)
	Dropout float32

	// QKVBias determines if Q/K/V projections use bias (false for GPT-2)
	QKVBias bool
}

// DefaultGPT2Config returns a configuration for GPT-2 124M model.
// This is the smallest GPT-2 model suitable for educational purposes.
func DefaultGPT2Config() GPT2Config {
	embDim := 768
	numHeads := 12
	return GPT2Config{
		VocabSize:     50257,
		ContextLength: 1024,
		EmbeddingDim:  embDim,
		NumHeads:      numHeads,
		NumLayers:     12,
		HeadDim:       embDim / numHeads, // 64
		HiddenDim:     3072,              // 4 * 768
		Dropout:       0.1,
		QKVBias:       false,
	}
}

// Validate checks if the configuration is valid and consistent.
// Returns an error if any parameters are incompatible.
func (c GPT2Config) Validate() error {
	if c.EmbeddingDim%c.NumHeads != 0 {
		return fmt.Errorf("embedding_dim (%d) must be divisible by num_heads (%d)",
			c.EmbeddingDim, c.NumHeads)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("vocab_size must be positive, got %d", c.VocabSize)
	}
	if c.ContextLength <= 0 {
		return fmt.Errorf("context_length must be positive, got %d", c.ContextLength)
	}
	if c.NumLayers <= 0 {
		return fmt.Errorf("num_layers must be positive, got %d", c.NumLayers)
	}
	return nil
}

// HeadDimension returns the dimension per attention head.
func (c GPT2Config) HeadDimension() int {
	return c.EmbeddingDim / c.NumHeads
}
