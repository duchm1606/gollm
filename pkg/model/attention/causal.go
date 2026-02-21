// Package attention implements various attention mechanisms for transformer models.
//
// This package provides attention variants for GPT-2:
//   - CausalSelfAttention: Basic causal scaled dot-product attention
//   - MultiHeadAttention: Multi-head attention without RoPE (GPT-2 style)
package attention

import (
	"fmt"
	"math"

	"gollm/pkg/tensor"
)

// CausalSelfAttentionConfig holds configuration for CausalSelfAttention.
type CausalSelfAttentionConfig struct {
	DIn     int
	DOut    int
	Dropout float32
}

// CausalSelfAttention implements basic causal self-attention.
//
// This is the simplest form of attention where each position attends to all
// previous positions (causal masking). It uses single projection matrices
// for Query, Key, and Value.
//
// GPT-2 uses multi-head attention, but this is kept for educational purposes.
type CausalSelfAttention struct {
	WQuery *tensor.Tensor // (d_in, d_out)
	WKey   *tensor.Tensor // (d_in, d_out)
	WValue *tensor.Tensor // (d_in, d_out)
	DOut   int
	Scale  float32 // 1/sqrt(d_out)
}

// NewCausalSelfAttention creates a new causal self-attention layer.
func NewCausalSelfAttention(config CausalSelfAttentionConfig) *CausalSelfAttention {
	return &CausalSelfAttention{
		WQuery: tensor.NewTensor([]int{config.DIn, config.DOut}),
		WKey:   tensor.NewTensor([]int{config.DIn, config.DOut}),
		WValue: tensor.NewTensor([]int{config.DIn, config.DOut}),
		DOut:   config.DOut,
		Scale:  float32(1.0 / math.Sqrt(float64(config.DOut))),
	}
}

// Forward computes causal self-attention.
//
// Input shape: (batch, seq, d_in)
// Output shape: (batch, seq, d_out)
//
// Steps:
//  1. Compute Q, K, V projections
//  2. Compute attention scores: Q @ K^T / sqrt(d_out)
//  3. Apply causal mask
//  4. Apply softmax to get attention weights
//  5. Compute output: weights @ V
func (c *CausalSelfAttention) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if len(x.Shape) != 3 {
		return nil, fmt.Errorf("expected 3D input (batch, seq, d_in), got %dD with shape %v",
			len(x.Shape), x.Shape)
	}

	_, seqLen, dIn := x.Shape[0], x.Shape[1], x.Shape[2]

	if dIn != c.WQuery.Shape[0] {
		return nil, fmt.Errorf("input dimension %d doesn't match WQuery shape %v",
			dIn, c.WQuery.Shape)
	}

	// Step 1: Compute Q, K, V projections
	// Q = x @ WQuery: (batch, seq, d_in) @ (d_in, d_out) -> (batch, seq, d_out)
	Q, err := tensor.Matmul(x, c.WQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Q: %w", err)
	}

	// K = x @ WKey: (batch, seq, d_in) @ (d_in, d_out) -> (batch, seq, d_out)
	K, err := tensor.Matmul(x, c.WKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute K: %w", err)
	}

	// V = x @ WValue: (batch, seq, d_in) @ (d_in, d_out) -> (batch, seq, d_out)
	V, err := tensor.Matmul(x, c.WValue)
	if err != nil {
		return nil, fmt.Errorf("failed to compute V: %w", err)
	}

	// Step 2: Compute attention scores: Q @ K^T
	// K^T: (batch, d_out, seq)
	KT, err := K.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose K: %w", err)
	}

	// scores: (batch, seq, d_out) @ (batch, d_out, seq) -> (batch, seq, seq)
	scores, err := tensor.Matmul(Q, KT)
	if err != nil {
		return nil, fmt.Errorf("failed to compute attention scores: %w", err)
	}

	// Scale: scores / sqrt(d_out)
	scores = scores.Scale(c.Scale)

	// Step 3: Apply causal mask
	mask := tensor.CreateCausalMask(seqLen)
	scores = tensor.ApplyMask(scores, mask)

	// Step 4: Apply softmax to get attention weights (along last dimension)
	weights, err := tensor.Softmax(scores, len(scores.Shape)-1)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax: %w", err)
	}

	// Step 5: Compute output: weights @ V
	// weights: (batch, seq, seq) @ V: (batch, seq, d_out) -> (batch, seq, d_out)
	output, err := tensor.Matmul(weights, V)
	if err != nil {
		return nil, fmt.Errorf("failed to compute attention output: %w", err)
	}

	return output, nil
}
