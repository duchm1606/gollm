// Package model provides the transformer model components for LLaMA-3 style architectures.
//
// This file implements Rotary Position Embeddings (RoPE), a technique for encoding
// positional information in transformer models by rotating query and key vectors.
//
// RoPE applies rotations to pairs of dimensions within each head, allowing the model
// to understand relative positions through the rotation angles.
//
// We use the "split-halves" style (like Hugging Face Transformers), where the head
// dimension is split into two halves and rotated separately, rather than the
// "interleaved" style that alternates dimensions.
package model

import (
	"fmt"
	"math"

	"gollm/pkg/tensor"
)

// RoPEParams holds precomputed cosine and sine values for RoPE.
// These values are computed once during initialization and reused
// for all forward passes, improving efficiency.
//
// The shapes are (max_seq_len, head_dim), where:
//   - max_seq_len: maximum sequence length the model can process
//   - head_dim: dimension of each attention head
//
// Both Cos and Sin have the same shape and are precomputed for all positions
// up to max_seq_len. This allows efficient lookup during the forward pass.
type RoPEParams struct {
	Cos       []float32 // Precomputed cos values: (max_seq_len, head_dim)
	Sin       []float32 // Precomputed sin values: (max_seq_len, head_dim)
	MaxSeqLen int       // Maximum sequence length supported
	HeadDim   int       // Dimension per attention head
}

// ComputeRoPE precomputes RoPE cosine and sine values for all positions.
//
// This function computes the inverse frequency schedule and then calculates
// cos and sin values for each position and each dimension.
//
// The frequency schedule is computed as:
//
//	inv_freq[i] = 1.0 / (thetaBase ^ (2*i / head_dim)) for i in [0, head_dim/2)
//
// For each position m, we compute:
//
//	angle[m][i] = m * inv_freq[i]
//	angles[m][i+head_dim/2] = m * inv_freq[i]  (duplicated for split-halves style)
//
// Then:
//
//	cos[m] = cos(angles[m])
//	sin[m] = sin(angles[m])
//
// Parameters:
//   - headDim: dimension of each attention head (must be even)
//   - maxSeqLen: maximum sequence length to support
//   - thetaBase: base value for frequency calculation (typically 10000 or 500000)
//
// Returns:
//   - *RoPEParams containing precomputed cos and sin values
//   - error if headDim is not even
//
// Note on "split-halves" vs "interleaved":
//
//	We use the "split-halves" approach (also used by Hugging Face), where the
//	head dimension [0, head_dim/2) is paired with [head_dim/2, head_dim).
//	This is different from "interleaved" where dimensions are paired as (0,1), (2,3), etc.
//	Split-halves is more cache-friendly and is the standard in modern implementations.
func ComputeRoPE(headDim, maxSeqLen int, thetaBase float32) (*RoPEParams, error) {
	if headDim%2 != 0 {
		return nil, fmt.Errorf("head_dim must be even, got %d", headDim)
	}
	if maxSeqLen <= 0 {
		return nil, fmt.Errorf("max_seq_len must be positive, got %d", maxSeqLen)
	}
	if thetaBase <= 0 {
		return nil, fmt.Errorf("theta_base must be positive, got %f", thetaBase)
	}

	// Number of unique frequencies (half the head dimension)
	numFreqs := headDim / 2

	// Preallocate cos and sin arrays
	cosValues := make([]float32, maxSeqLen*headDim)
	sinValues := make([]float32, maxSeqLen*headDim)

	// Compute inverse frequencies: 1.0 / (thetaBase^(2*i/head_dim))
	invFreq := make([]float32, numFreqs)
	for i := 0; i < numFreqs; i++ {
		// Formula: 1.0 / (thetaBase ^ (2*i / head_dim))
		// In log space: exp(-ln(thetaBase) * 2*i / head_dim)
		exponent := -math.Log(float64(thetaBase)) * float64(2*i) / float64(headDim)
		invFreq[i] = float32(math.Exp(exponent))
	}

	// Compute cos and sin for each position
	for pos := 0; pos < maxSeqLen; pos++ {
		baseIdx := pos * headDim

		// For each frequency, compute the angle and duplicate it
		// (split-halves style: first half and second half get the same angles)
		for i := 0; i < numFreqs; i++ {
			// Angle for this position and frequency
			angle := float64(pos) * float64(invFreq[i])

			// Store in both halves of the head dimension
			cosValues[baseIdx+i] = float32(math.Cos(angle))
			sinValues[baseIdx+i] = float32(math.Sin(angle))
			cosValues[baseIdx+i+numFreqs] = float32(math.Cos(angle))
			sinValues[baseIdx+i+numFreqs] = float32(math.Sin(angle))
		}
	}

	return &RoPEParams{
		Cos:       cosValues,
		Sin:       sinValues,
		MaxSeqLen: maxSeqLen,
		HeadDim:   headDim,
	}, nil
}

// ApplyRoPE applies Rotary Position Embeddings to a tensor.
//
// This implements the RoPE rotation on query or key tensors. The rotation is
// applied independently to each head and each position in the sequence.
//
// The rotation formula (split-halves style):
//
//	Given x = [x1, x2] where x1 is first half, x2 is second half
//	rotated = [-x2, x1]  (negate second half, swap halves)
//	x_rotated = (x * cos) + (rotated * sin)
//
// In more detail:
//
//	For each position m and each pair of dimensions (i, i+head_dim/2):
//	  x1 = x[..., i]
//	  x2 = x[..., i+head_dim/2]
//	  x_rot[..., i]           = x1*cos[m][i] - x2*sin[m][i]
//	  x_rot[..., i+head_dim/2] = x2*cos[m][i] + x1*sin[m][i]
//
// Parameters:
//   - x: input tensor of shape (batch_size, num_heads, seq_len, head_dim)
//   - rope: precomputed RoPE parameters
//   - offset: starting position in the sequence (for KV cache support)
//
// Returns:
//   - rotated tensor with same shape as input
//   - error if shapes are incompatible
//
// Example with offset:
//
//	If offset=10 and seq_len=5, the function will use cos/sin values
//	from positions 10-14, allowing efficient KV cache implementation
//	where new tokens are computed relative to previous positions.
func ApplyRoPE(x *tensor.Tensor, rope *RoPEParams, offset int) (*tensor.Tensor, error) {
	// Validate input shape: (batch, heads, seq, head_dim)
	if len(x.Shape) != 4 {
		return nil, fmt.Errorf("expected 4D tensor (batch, heads, seq, head_dim), got shape %v", x.Shape)
	}

	batchSize := x.Shape[0]
	numHeads := x.Shape[1]
	seqLen := x.Shape[2]
	headDim := x.Shape[3]

	// Validate dimensions
	if headDim != rope.HeadDim {
		return nil, fmt.Errorf("head_dim mismatch: tensor has %d, RoPE expects %d", headDim, rope.HeadDim)
	}

	if offset+seqLen > rope.MaxSeqLen {
		return nil, fmt.Errorf("offset+seq_len (%d+%d=%d) exceeds max_seq_len (%d)",
			offset, seqLen, offset+seqLen, rope.MaxSeqLen)
	}

	// Create output tensor (copy input to preserve original for rotation)
	output := x.Clone()

	// Half dimension for split-halves rotation
	halfDim := headDim / 2

	// Strides for indexing into the 4D tensor
	// stride[0] = heads * seq_len * head_dim
	// stride[1] = seq_len * head_dim
	// stride[2] = head_dim
	// stride[3] = 1
	stride0 := numHeads * seqLen * headDim
	stride1 := seqLen * headDim
	stride2 := headDim

	// Apply rotation to each element
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				// Position in the precomputed cos/sin arrays
				// Use offset to support KV cache (start from specific position)
				pos := offset + s
				ropeBaseIdx := pos * headDim

				// Base index for this (batch, head, seq) position
				tensorBaseIdx := b*stride0 + h*stride1 + s*stride2

				// Process each pair of dimensions
				for i := 0; i < halfDim; i++ {
					// Indices into the tensor
					idx1 := tensorBaseIdx + i           // First half
					idx2 := tensorBaseIdx + i + halfDim // Second half

					// Get values
					x1 := x.Data[idx1]
					x2 := x.Data[idx2]

					// Get cos and sin for this position and dimension
					cosVal := rope.Cos[ropeBaseIdx+i]
					sinVal := rope.Sin[ropeBaseIdx+i]

					// Apply rotation:
					// x1' = x1*cos - x2*sin
					// x2' = x2*cos + x1*sin
					output.Data[idx1] = x1*cosVal - x2*sinVal
					output.Data[idx2] = x2*cosVal + x1*sinVal
				}
			}
		}
	}

	return output, nil
}

// GetCosSlice returns a slice of cosine values for a specific position range.
// This is useful for debugging and testing.
//
// Parameters:
//   - rope: the RoPE parameters
//   - pos: the position to retrieve
//
// Returns:
//   - slice of cos values for that position (length = head_dim)
func GetCosSlice(rope *RoPEParams, pos int) []float32 {
	start := pos * rope.HeadDim
	end := start + rope.HeadDim
	result := make([]float32, rope.HeadDim)
	copy(result, rope.Cos[start:end])
	return result
}

// GetSinSlice returns a slice of sine values for a specific position range.
// This is useful for debugging and testing.
//
// Parameters:
//   - rope: the RoPE parameters
//   - pos: the position to retrieve
//
// Returns:
//   - slice of sin values for that position (length = head_dim)
func GetSinSlice(rope *RoPEParams, pos int) []float32 {
	start := pos * rope.HeadDim
	end := start + rope.HeadDim
	result := make([]float32, rope.HeadDim)
	copy(result, rope.Sin[start:end])
	return result
}
