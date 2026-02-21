package model

import (
	"fmt"

	"gollm/pkg/tensor"
)

// FeedForward implements the feed-forward network used in GPT-2.
//
// Architecture:
//  1. Linear projection: x @ FC1 -> (batch, seq, hidden_dim)
//  2. GELU activation
//  3. Linear projection: @ FC2 -> (batch, seq, emb_dim)
//
// This is simpler than SwiGLU used in LLaMA and doesn't use gating.
type FeedForward struct {
	FC1 *tensor.Tensor // (emb_dim, hidden_dim)
	FC2 *tensor.Tensor // (hidden_dim, emb_dim)
}

// NewFeedForward creates a new feed-forward layer.
//
// Parameters:
//   - config: GPT2Config containing emb_dim and hidden_dim
//
// Returns:
//   - Initialized FeedForward with FC1 and FC2 weight matrices
func NewFeedForward(config GPT2Config) *FeedForward {
	return &FeedForward{
		FC1: tensor.NewTensor([]int{config.EmbeddingDim, config.HiddenDim}),
		FC2: tensor.NewTensor([]int{config.HiddenDim, config.EmbeddingDim}),
	}
}

// Forward computes the feed-forward transformation.
//
// Input shape: (batch, seq, emb_dim)
// Output shape: (batch, seq, emb_dim)
//
// Steps:
//  1. x @ FC1 -> (batch, seq, hidden_dim)
//  2. Apply GELU activation
//  3. @ FC2 -> (batch, seq, emb_dim)
func (ff *FeedForward) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if len(x.Shape) < 2 {
		return nil, fmt.Errorf("expected at least 2D input, got %dD", len(x.Shape))
	}

	// Verify input dimension
	lastDim := x.Shape[len(x.Shape)-1]
	if lastDim != ff.FC1.Shape[0] {
		return nil, fmt.Errorf("input dimension %d doesn't match FC1 input dimension %d",
			lastDim, ff.FC1.Shape[0])
	}

	// Step 1: First linear projection
	// x: (batch, seq, emb_dim) @ FC1: (emb_dim, hidden_dim) -> (batch, seq, hidden_dim)
	hidden, err := tensor.Matmul(x, ff.FC1)
	if err != nil {
		return nil, fmt.Errorf("failed to compute FC1 projection: %w", err)
	}

	// Step 2: Apply GELU activation
	// GELU is applied element-wise
	activated := hidden.GELU()

	// Step 3: Second linear projection
	// activated: (batch, seq, hidden_dim) @ FC2: (hidden_dim, emb_dim) -> (batch, seq, emb_dim)
	output, err := tensor.Matmul(activated, ff.FC2)
	if err != nil {
		return nil, fmt.Errorf("failed to compute FC2 projection: %w", err)
	}

	return output, nil
}
