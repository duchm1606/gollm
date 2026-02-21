package attention

import (
	"fmt"
	"math"

	"gollm/pkg/tensor"
)

// MultiHeadAttention implements multi-head attention for GPT-2.
//
// This splits the attention computation across multiple heads, allowing the model
// to jointly attend to information from different representation subspaces.
//
// Architecture:
//   - Each head has its own Q, K, V projections
//   - Heads are computed in parallel
//   - Output projection combines all heads
//   - Optional bias terms (configurable via QKVBias)
//   - Dropout applied after softmax
//
// Unlike LLaMA, GPT-2 does NOT use RoPE (uses learned positional embeddings instead).
type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	DOut     int
	DIn      int
	Dropout  float32
	QKVBias  bool

	WQuery  *tensor.Tensor // (d_in, d_out)
	WKey    *tensor.Tensor // (d_in, d_out)
	WValue  *tensor.Tensor // (d_in, d_out)
	OutProj *tensor.Tensor // (d_out, d_out)

	// Bias terms (optional, based on QKVBias config)
	BiasQ *tensor.Tensor // (d_out,)
	BiasK *tensor.Tensor // (d_out,)
	BiasV *tensor.Tensor // (d_out,)
}

// MultiHeadAttentionConfig holds configuration for MultiHeadAttention.
type MultiHeadAttentionConfig struct {
	NumHeads int
	DIn      int
	DOut     int
	Dropout  float32
	QKVBias  bool
}

// NewMultiHeadAttention creates a new multi-head attention layer.
func NewMultiHeadAttention(config MultiHeadAttentionConfig) *MultiHeadAttention {
	if config.DOut%config.NumHeads != 0 {
		panic(fmt.Sprintf("d_out (%d) must be divisible by num_heads (%d)", config.DOut, config.NumHeads))
	}

	mha := &MultiHeadAttention{
		NumHeads: config.NumHeads,
		HeadDim:  config.DOut / config.NumHeads,
		DOut:     config.DOut,
		DIn:      config.DIn,
		Dropout:  config.Dropout,
		QKVBias:  config.QKVBias,
		WQuery:   tensor.NewTensor([]int{config.DIn, config.DOut}),
		WKey:     tensor.NewTensor([]int{config.DIn, config.DOut}),
		WValue:   tensor.NewTensor([]int{config.DIn, config.DOut}),
		OutProj:  tensor.NewTensor([]int{config.DOut, config.DOut}),
	}

	// Initialize bias terms if needed
	if config.QKVBias {
		mha.BiasQ = tensor.NewTensor([]int{config.DOut})
		mha.BiasK = tensor.NewTensor([]int{config.DOut})
		mha.BiasV = tensor.NewTensor([]int{config.DOut})
	}

	return mha
}

// Forward computes multi-head attention.
//
// Input shapes:
//   - x: (batch, seq, d_in)
//   - mask: optional causal mask, shape (seq, seq) or nil
//   - training: if true, apply dropout; if false, skip dropout
//
// Output shape: (batch, seq, d_out)
//
// Note: GPT-2 uses learned positional embeddings, NOT RoPE.
func (m *MultiHeadAttention) Forward(x, mask *tensor.Tensor, training bool) (*tensor.Tensor, error) {
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
	if m.QKVBias {
		Q = addBias(Q, m.BiasQ)
	}

	K, err := tensor.Matmul(x, m.WKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute K: %w", err)
	}
	if m.QKVBias {
		K = addBias(K, m.BiasK)
	}

	V, err := tensor.Matmul(x, m.WValue)
	if err != nil {
		return nil, fmt.Errorf("failed to compute V: %w", err)
	}
	if m.QKVBias {
		V = addBias(V, m.BiasV)
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

	// Step 3: Compute attention scores
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

	// Step 4: Apply mask if provided
	if mask != nil {
		scores = tensor.ApplyMask(scores, mask)
	}

	// Step 5: Softmax (along last dimension)
	weights, err := tensor.Softmax(scores, len(scores.Shape)-1)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax: %w", err)
	}

	// Step 6: Apply dropout to attention weights (if training)
	if m.Dropout > 0 && training {
		weights = weights.Dropout(m.Dropout, training)
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

// addBias adds bias to a tensor by broadcasting the bias across all dimensions except the last.
// input shape: (..., d_out), bias shape: (d_out,)
// output shape: (..., d_out)
func addBias(input, bias *tensor.Tensor) *tensor.Tensor {
	// Simple broadcast addition
	// For now, manually add bias to each position
	result := tensor.NewTensor(input.Shape)
	copy(result.Data, input.Data)

	// Get dimensions
	dOut := bias.Shape[0]
	numElements := len(input.Data)
	numPositions := numElements / dOut

	// Add bias to each position
	for pos := 0; pos < numPositions; pos++ {
		offset := pos * dOut
		for i := 0; i < dOut; i++ {
			result.Data[offset+i] += bias.Data[i]
		}
	}

	return result
}
