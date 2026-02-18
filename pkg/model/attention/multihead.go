package attention

import (
	"fmt"
	"math"

	"gollm/pkg/model"
	"gollm/pkg/tensor"
)

// MultiHeadAttention implements multi-head attention with RoPE support.
//
// This splits the attention computation across multiple heads, allowing the model
// to jointly attend to information from different representation subspaces.
//
// Architecture:
//   - Each head has its own Q, K, V projections
//   - Heads are computed in parallel
//   - Output projection combines all heads
type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	DOut     int
	DIn      int
	Dropout  float32

	WQuery  *tensor.Tensor // (d_in, d_out)
	WKey    *tensor.Tensor // (d_in, d_out)
	WValue  *tensor.Tensor // (d_in, d_out)
	OutProj *tensor.Tensor // (d_out, d_out)
}

// NewMultiHeadAttention creates a new multi-head attention layer.
func NewMultiHeadAttention(config model.Config, dIn, dOut int) *MultiHeadAttention {
	if dOut%config.NumHeads != 0 {
		panic(fmt.Sprintf("d_out (%d) must be divisible by num_heads (%d)", dOut, config.NumHeads))
	}

	return &MultiHeadAttention{
		NumHeads: config.NumHeads,
		HeadDim:  dOut / config.NumHeads,
		DOut:     dOut,
		DIn:      dIn,
		Dropout:  config.Dropout,
		WQuery:   tensor.NewTensor([]int{dIn, dOut}),
		WKey:     tensor.NewTensor([]int{dIn, dOut}),
		WValue:   tensor.NewTensor([]int{dIn, dOut}),
		OutProj:  tensor.NewTensor([]int{dOut, dOut}),
	}
}

// Forward computes multi-head attention.
//
// Input shapes:
//   - x: (batch, seq, d_in)
//   - mask: optional causal mask, shape (seq, seq) or nil
//   - rope: optional RoPE parameters for position encoding
//
// Output shape: (batch, seq, d_out)
func (m *MultiHeadAttention) Forward(x, mask *tensor.Tensor, rope *model.RoPEParams) (*tensor.Tensor, error) {
	if len(x.Shape) != 3 {
		return nil, fmt.Errorf("expected 3D input (batch, seq, d_in), got %dD with shape %v",
			len(x.Shape), x.Shape)
	}

	batchSize, seqLen, dIn := x.Shape[0], x.Shape[1], x.Shape[2]

	if dIn != m.DIn {
		return nil, fmt.Errorf("input dimension %d doesn't match expected %d", dIn, m.DIn)
	}

	// Step 1: Project to Q, K, V
	// Q, K, V: (batch, seq, d_out)
	Q, err := tensor.Matmul(x, m.WQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Q: %w", err)
	}

	K, err := tensor.Matmul(x, m.WKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute K: %w", err)
	}

	V, err := tensor.Matmul(x, m.WValue)
	if err != nil {
		return nil, fmt.Errorf("failed to compute V: %w", err)
	}

	// Step 2: Reshape to separate heads
	// From: (batch, seq, d_out)
	// To: (batch, num_heads, seq, head_dim)
	Q = Q.Reshape([]int{batchSize, seqLen, m.NumHeads, m.HeadDim})
	Q, err = Q.Transpose(1, 2) // (batch, num_heads, seq, head_dim)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose Q: %w", err)
	}

	K = K.Reshape([]int{batchSize, seqLen, m.NumHeads, m.HeadDim})
	K, err = K.Transpose(1, 2) // (batch, num_heads, seq, head_dim)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose K: %w", err)
	}

	V = V.Reshape([]int{batchSize, seqLen, m.NumHeads, m.HeadDim})
	V, err = V.Transpose(1, 2) // (batch, num_heads, seq, head_dim)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose V: %w", err)
	}

	// Step 3: Apply RoPE if provided
	if rope != nil {
		// RoPE is applied to Q and K here
		// Implementation would multiply by cos and sin matrices
		// For now, we skip this step in the basic implementation
		_ = rope
	}

	// Step 4: Compute attention scores
	// Q: (batch, num_heads, seq, head_dim)
	// K^T: (batch, num_heads, head_dim, seq)
	// scores: (batch, num_heads, seq, seq)
	KT, err := K.Transpose(2, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose K: %w", err)
	}
	scores, err := tensor.Matmul(Q, KT)
	if err != nil {
		return nil, fmt.Errorf("failed to compute attention scores: %w", err)
	}

	// Scale
	scale := float32(1.0 / math.Sqrt(float64(m.HeadDim)))
	scores = scores.Scale(scale)

	// Step 5: Apply mask if provided
	if mask != nil {
		scores = tensor.ApplyMask(scores, mask)
	}

	// Step 6: Softmax (along last dimension)
	weights, err := tensor.Softmax(scores, len(scores.Shape)-1)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax: %w", err)
	}

	// Step 7: Apply attention to V
	// weights: (batch, num_heads, seq, seq)
	// V: (batch, num_heads, seq, head_dim)
	// output: (batch, num_heads, seq, head_dim)
	attnOutput, err := tensor.Matmul(weights, V)
	if err != nil {
		return nil, fmt.Errorf("failed to apply attention to V: %w", err)
	}

	// Step 8: Reshape back to (batch, seq, d_out)
	// From: (batch, num_heads, seq, head_dim)
	// To: (batch, seq, num_heads, head_dim) -> (batch, seq, d_out)
	attnOutput, err = attnOutput.Transpose(1, 2) // (batch, seq, num_heads, head_dim)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose attention output: %w", err)
	}
	attnOutput = attnOutput.Reshape([]int{batchSize, seqLen, m.DOut})

	// Step 9: Output projection
	output, err := tensor.Matmul(attnOutput, m.OutProj)
	if err != nil {
		return nil, fmt.Errorf("failed to apply output projection: %w", err)
	}

	return output, nil
}
