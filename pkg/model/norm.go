package model

import (
	"fmt"
	"math"

	"gollm/pkg/tensor"
)

// LayerNorm implements layer normalization with learnable scale and shift.
//
// LayerNorm normalizes the input across the last dimension (feature dimension)
// and applies a learned scale (gamma) and shift (beta) transformation.
//
// Formula:
//
//	mean = mean(x, dim=-1, keepdim=True)
//	var = var(x, dim=-1, keepdim=True)
//	x_norm = (x - mean) / sqrt(var + eps)
//	output = x_norm * scale + shift
//
// This is used in GPT-2 (instead of RMSNorm used in LLaMA).
type LayerNorm struct {
	Scale *tensor.Tensor // (emb_dim,) - gamma parameter
	Shift *tensor.Tensor // (emb_dim,) - beta parameter
	Eps   float32        // Small constant for numerical stability
}

// NewLayerNorm creates a new LayerNorm layer.
//
// Parameters:
//   - embDim: embedding dimension
//   - eps: small constant for numerical stability (typically 1e-5)
//
// Returns:
//   - Initialized LayerNorm with scale=1 and shift=0
func NewLayerNorm(embDim int, eps float32) *LayerNorm {
	// Initialize scale (gamma) to 1
	scale := tensor.NewTensor([]int{embDim})
	for i := range scale.Data {
		scale.Data[i] = 1.0
	}

	// Initialize shift (beta) to 0
	shift := tensor.NewTensor([]int{embDim})

	return &LayerNorm{
		Scale: scale,
		Shift: shift,
		Eps:   eps,
	}
}

// Forward applies layer normalization to the input.
//
// Input shape: (batch, seq, emb_dim) or any shape where last dim is emb_dim
// Output shape: same as input
//
// The normalization is applied independently to each position in the sequence.
func (ln *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if len(x.Shape) == 0 {
		return nil, fmt.Errorf("cannot apply LayerNorm to 0D tensor")
	}

	// Get the last dimension size (embedding dimension)
	lastDim := x.Shape[len(x.Shape)-1]
	if lastDim != len(ln.Scale.Data) {
		return nil, fmt.Errorf("input last dimension %d doesn't match LayerNorm dimension %d",
			lastDim, len(ln.Scale.Data))
	}

	// Calculate number of independent normalization operations
	// This is the product of all dimensions except the last
	numSlices := 1
	for i := 0; i < len(x.Shape)-1; i++ {
		numSlices *= x.Shape[i]
	}
	sliceSize := lastDim

	// Create output tensor
	result := tensor.NewTensor(x.Shape)

	// Apply LayerNorm to each slice independently
	for sliceIdx := 0; sliceIdx < numSlices; sliceIdx++ {
		// Calculate offset for this slice
		offset := sliceIdx * sliceSize

		// Step 1: Compute mean
		mean := float32(0)
		for i := 0; i < sliceSize; i++ {
			mean += x.Data[offset+i]
		}
		mean /= float32(sliceSize)

		// Step 2: Compute variance
		variance := float32(0)
		for i := 0; i < sliceSize; i++ {
			diff := x.Data[offset+i] - mean
			variance += diff * diff
		}
		variance /= float32(sliceSize)

		// Step 3: Normalize
		invStd := float32(1.0 / math.Sqrt(float64(variance+ln.Eps)))

		// Step 4: Apply scale and shift
		for i := 0; i < sliceSize; i++ {
			xNorm := (x.Data[offset+i] - mean) * invStd
			result.Data[offset+i] = xNorm*ln.Scale.Data[i] + ln.Shift.Data[i]
		}
	}

	return result, nil
}
