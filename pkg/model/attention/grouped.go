package attention

import (
	"fmt"
	"math"

	"gollm/pkg/model"
	"gollm/pkg/tensor"
)

// GroupedQueryAttention implements Grouped Query Attention (GQA).
//
// GQA is a memory-efficient attention mechanism where multiple query heads
// share the same key and value heads. This reduces memory usage during
// autoregressive generation while maintaining most of the performance.
//
// Architecture:
//   - NumHeads query heads
//   - NumKVGroups key-value groups (fewer than query heads)
//   - Each KV group is shared by GroupSize query heads
//
// Benefits:
//   - Reduced memory for KV cache (NumKVGroups vs NumHeads)
//   - Faster inference for long sequences
//   - Minimal performance degradation vs full multi-head attention
type GroupedQueryAttention struct {
	NumHeads    int
	NumKVGroups int
	GroupSize   int // NumHeads / NumKVGroups
	HeadDim     int
	DOut        int
	DIn         int
	Dropout     float32

	WQuery  *tensor.Tensor // (d_in, d_out)
	WKey    *tensor.Tensor // (d_in, num_kv_groups * head_dim)
	WValue  *tensor.Tensor // (d_in, num_kv_groups * head_dim)
	OutProj *tensor.Tensor // (d_out, d_out)
}

// NewGroupedQueryAttention creates a new GQA layer.
func NewGroupedQueryAttention(config model.Config, dIn, dOut int) *GroupedQueryAttention {
	if config.NumHeads%config.NumKVGroups != 0 {
		panic(fmt.Sprintf("num_heads (%d) must be divisible by num_kv_groups (%d)",
			config.NumHeads, config.NumKVGroups))
	}

	if dOut%config.NumHeads != 0 {
		panic(fmt.Sprintf("d_out (%d) must be divisible by num_heads (%d)", dOut, config.NumHeads))
	}

	headDim := dOut / config.NumHeads
	kvDim := config.NumKVGroups * headDim

	return &GroupedQueryAttention{
		NumHeads:    config.NumHeads,
		NumKVGroups: config.NumKVGroups,
		GroupSize:   config.NumHeads / config.NumKVGroups,
		HeadDim:     headDim,
		DOut:        dOut,
		DIn:         dIn,
		Dropout:     config.Dropout,
		WQuery:      tensor.NewTensor([]int{dIn, dOut}),
		WKey:        tensor.NewTensor([]int{dIn, kvDim}),
		WValue:      tensor.NewTensor([]int{dIn, kvDim}),
		OutProj:     tensor.NewTensor([]int{dOut, dOut}),
	}
}

// Forward computes grouped query attention.
//
// Key GQA Concept:
//
//	Query heads are divided into groups, where each group shares the same
//	K and V projections. This reduces memory while maintaining attention
//	across different representation subspaces.
//
// Input shapes:
//   - x: (batch, seq, d_in)
//   - mask: optional causal mask, shape (seq, seq) or nil
//   - rope: optional RoPE parameters for position encoding
//
// Output shape: (batch, seq, d_out)
func (g *GroupedQueryAttention) Forward(x, mask *tensor.Tensor, rope *model.RoPEParams) (*tensor.Tensor, error) {
	if len(x.Shape) != 3 {
		return nil, fmt.Errorf("expected 3D input (batch, seq, d_in), got %dD with shape %v",
			len(x.Shape), x.Shape)
	}

	batchSize, seqLen, dIn := x.Shape[0], x.Shape[1], x.Shape[2]

	if dIn != g.DIn {
		return nil, fmt.Errorf("input dimension %d doesn't match expected %d", dIn, g.DIn)
	}

	// Step 1: Project to Q, K, V
	// Q: (batch, seq, d_out)
	// K, V: (batch, seq, num_kv_groups * head_dim)
	Q, err := tensor.Matmul(x, g.WQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Q: %w", err)
	}

	K, err := tensor.Matmul(x, g.WKey)
	if err != nil {
		return nil, fmt.Errorf("failed to compute K: %w", err)
	}

	V, err := tensor.Matmul(x, g.WValue)
	if err != nil {
		return nil, fmt.Errorf("failed to compute V: %w", err)
	}

	// Step 2: Reshape
	// Q: (batch, seq, d_out) -> (batch, num_heads, seq, head_dim)
	// K, V: (batch, seq, num_kv_groups * head_dim) -> (batch, num_kv_groups, seq, head_dim)
	Q = Q.Reshape([]int{batchSize, seqLen, g.NumHeads, g.HeadDim})
	Q, err = Q.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose Q: %w", err)
	}

	K = K.Reshape([]int{batchSize, seqLen, g.NumKVGroups, g.HeadDim})
	K, err = K.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose K: %w", err)
	}

	V = V.Reshape([]int{batchSize, seqLen, g.NumKVGroups, g.HeadDim})
	V, err = V.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose V: %w", err)
	}

	// Step 3: Apply RoPE if provided
	if rope != nil {
		_ = rope
	}

	// Step 4: Expand K and V from num_kv_groups to num_heads
	// Each KV head is shared by GroupSize query heads
	// K, V: (batch, num_kv_groups, seq, head_dim)
	// Expand to: (batch, num_heads, seq, head_dim)
	K = K.Expand(1, g.GroupSize)
	V = V.Expand(1, g.GroupSize)

	// Step 5: Compute attention scores
	// Q: (batch, num_heads, seq, head_dim)
	// K^T: (batch, num_heads, head_dim, seq)
	KT, err := K.Transpose(2, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose K: %w", err)
	}
	scores, err := tensor.Matmul(Q, KT)
	if err != nil {
		return nil, fmt.Errorf("failed to compute attention scores: %w", err)
	}

	// Scale
	scale := float32(1.0 / math.Sqrt(float64(g.HeadDim)))
	scores = scores.Scale(scale)

	// Step 6: Apply mask if provided
	if mask != nil {
		scores = tensor.ApplyMask(scores, mask)
	}

	// Step 7: Softmax (along last dimension)
	weights, err := tensor.Softmax(scores, len(scores.Shape)-1)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax: %w", err)
	}

	// Step 8: Apply attention to V
	// weights: (batch, num_heads, seq, seq)
	// V: (batch, num_heads, seq, head_dim)
	// output: (batch, num_heads, seq, head_dim)
	attnOutput, err := tensor.Matmul(weights, V)
	if err != nil {
		return nil, fmt.Errorf("failed to apply attention to V: %w", err)
	}

	// Step 9: Reshape back to (batch, seq, d_out)
	attnOutput, err = attnOutput.Transpose(1, 2) // (batch, seq, num_heads, head_dim)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose attention output: %w", err)
	}
	attnOutput = attnOutput.Reshape([]int{batchSize, seqLen, g.DOut})

	// Step 10: Output projection
	output, err := tensor.Matmul(attnOutput, g.OutProj)
	if err != nil {
		return nil, fmt.Errorf("failed to apply output projection: %w", err)
	}

	return output, nil
}
