package tensor

import (
	"math/rand"
	"time"
)

// Dropout randomly zeros out elements with probability p during training.
// During inference (training=false), returns input unchanged.
//
// Parameters:
//   - t: input tensor
//   - p: dropout probability (0.0 to 1.0)
//   - training: if true, apply dropout; if false, return input unchanged
//
// Returns:
//   - Tensor with dropout applied (if training)
func (t *Tensor) Dropout(p float32, training bool) *Tensor {
	if !training || p == 0 {
		return t.Clone()
	}

	if p < 0 || p > 1 {
		panic("dropout probability must be between 0 and 1")
	}

	// Initialize random seed if needed
	if dropoutRand == nil {
		dropoutRand = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	result := NewTensor(t.Shape)
	scale := 1.0 / (1.0 - p) // Inverted dropout scaling

	for i := range t.Data {
		if dropoutRand.Float32() >= p {
			// Keep the value and scale it
			result.Data[i] = t.Data[i] * float32(scale)
		} else {
			// Drop the value (set to 0)
			result.Data[i] = 0
		}
	}

	return result
}

// dropoutRand is a package-level random number generator for dropout
var dropoutRand *rand.Rand

// SetDropoutSeed sets the random seed for dropout (useful for testing)
func SetDropoutSeed(seed int64) {
	dropoutRand = rand.New(rand.NewSource(seed))
}

// ApplyDropout applies dropout to a tensor using the given probability and training mode.
// This is a convenience function that calls the Dropout method.
func ApplyDropout(t *Tensor, p float32, training bool) *Tensor {
	return t.Dropout(p, training)
}
