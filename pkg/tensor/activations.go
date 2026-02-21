package tensor

import "math"

// GELU applies the Gaussian Error Linear Unit activation function.
//
// The GELU function is defined as:
//
//	GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// This is the approximation used in the original GPT-2 paper and is
// more efficient to compute than the exact GELU formulation.
//
// Reference: https://arxiv.org/abs/1606.08415
//
// Input: tensor of any shape
// Output: tensor of the same shape with GELU applied element-wise
func (t *Tensor) GELU() *Tensor {
	result := NewTensor(t.Shape)

	// GELU approximation constants
	const (
		sqrt2OverPi = 0.7978845608 // sqrt(2/π)
		coeff       = 0.044715
	)

	for i := range t.Data {
		x := t.Data[i]
		// Compute x + 0.044715 * x^3
		x3 := x * x * x
		inner := x + coeff*x3
		// Compute tanh(sqrt(2/π) * inner)
		tanhVal := float32(math.Tanh(float64(sqrt2OverPi * inner)))
		// GELU(x) = 0.5 * x * (1 + tanh(...))
		result.Data[i] = 0.5 * x * (1 + tanhVal)
	}

	return result
}

// GELU is a standalone function that applies GELU to a tensor.
// This is a convenience wrapper around the Tensor.GELU method.
func GELU(t *Tensor) *Tensor {
	return t.GELU()
}
