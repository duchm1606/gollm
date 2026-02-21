package tensor

import (
	"math"
	"testing"
)

// TestGELU_ZeroInput tests that GELU(0) is close to 0
func TestGELU_ZeroInput(t *testing.T) {
	input := NewTensor([]int{1})
	input.Data[0] = 0.0

	output := input.GELU()

	// GELU(0) should be very close to 0
	if math.Abs(float64(output.Data[0])) > 1e-6 {
		t.Errorf("GELU(0) = %f, expected close to 0", output.Data[0])
	}
}

// TestGELU_PositiveInput tests GELU with positive values
func TestGELU_PositiveInput(t *testing.T) {
	// Test values from PyTorch: torch.nn.functional.gelu([0.0, 1.0, 2.0])
	// Expected: [0.0000, 0.8413, 1.9545]
	testCases := []struct {
		input    float32
		expected float32
		tol      float32
	}{
		{1.0, 0.8413, 0.001},
		{2.0, 1.9545, 0.001},
		{0.5, 0.3457, 0.001},
	}

	for _, tc := range testCases {
		input := NewTensor([]int{1})
		input.Data[0] = tc.input

		output := input.GELU()

		diff := math.Abs(float64(output.Data[0] - tc.expected))
		if diff > float64(tc.tol) {
			t.Errorf("GELU(%f) = %f, expected %f (diff: %f)",
				tc.input, output.Data[0], tc.expected, diff)
		}
	}
}

// TestGELU_NegativeInput tests GELU with negative values
// For negative inputs, GELU should approach 0
func TestGELU_NegativeInput(t *testing.T) {
	// Test values from PyTorch: torch.nn.functional.gelu([-1.0, -2.0])
	// Expected: [-0.1587, -0.0455]
	testCases := []struct {
		input    float32
		expected float32
		tol      float32
	}{
		{-1.0, -0.1587, 0.001},
		{-2.0, -0.0455, 0.001},
		{-0.5, -0.1543, 0.001},
	}

	for _, tc := range testCases {
		input := NewTensor([]int{1})
		input.Data[0] = tc.input

		output := input.GELU()

		diff := math.Abs(float64(output.Data[0] - tc.expected))
		if diff > float64(tc.tol) {
			t.Errorf("GELU(%f) = %f, expected %f (diff: %f)",
				tc.input, output.Data[0], tc.expected, diff)
		}
	}
}

// TestGELU_ShapePreservation tests that output shape matches input shape
func TestGELU_ShapePreservation(t *testing.T) {
	testShapes := [][]int{
		{1},
		{10},
		{2, 3},
		{2, 3, 4},
		{1, 2, 3, 4},
	}

	for _, shape := range testShapes {
		input := NewTensor(shape)
		// Fill with some values
		for i := range input.Data {
			input.Data[i] = float32(i) * 0.1
		}

		output := input.GELU()

		// Check shape preservation
		if len(output.Shape) != len(shape) {
			t.Errorf("Shape rank mismatch: input %v, output %v", shape, output.Shape)
			continue
		}

		for i := range shape {
			if output.Shape[i] != shape[i] {
				t.Errorf("Shape dimension %d mismatch: input %d, output %d",
					i, shape[i], output.Shape[i])
			}
		}

		// Check data size
		expectedSize := 1
		for _, dim := range shape {
			expectedSize *= dim
		}
		if len(output.Data) != expectedSize {
			t.Errorf("Data size mismatch for shape %v: expected %d, got %d",
				shape, expectedSize, len(output.Data))
		}
	}
}

// TestGELU_LargePositive tests GELU with large positive values
// For large positive x, GELU(x) should approach x
func TestGELU_LargePositive(t *testing.T) {
	input := NewTensor([]int{1})
	input.Data[0] = 5.0

	output := input.GELU()

	// For large positive values, GELU(x) ≈ x
	// GELU(5.0) should be close to 5.0
	expected := float32(5.0)
	tol := float32(0.01)

	if math.Abs(float64(output.Data[0]-expected)) > float64(tol) {
		t.Errorf("GELU(%f) = %f, expected approximately %f",
			input.Data[0], output.Data[0], expected)
	}
}

// TestGELU_LargeNegative tests GELU with large negative values
// For large negative x, GELU(x) should approach 0
func TestGELU_LargeNegative(t *testing.T) {
	input := NewTensor([]int{1})
	input.Data[0] = -5.0

	output := input.GELU()

	// For large negative values, GELU(x) should be very close to 0
	if math.Abs(float64(output.Data[0])) > 0.01 {
		t.Errorf("GELU(%f) = %f, expected close to 0",
			input.Data[0], output.Data[0])
	}
}

// TestGELU_NonDestructive tests that GELU doesn't modify the input tensor
func TestGELU_NonDestructive(t *testing.T) {
	shape := []int{2, 3}
	input := NewTensor(shape)
	originalValues := make([]float32, len(input.Data))
	for i := range input.Data {
		input.Data[i] = float32(i) * 0.5
		originalValues[i] = input.Data[i]
	}

	_ = input.GELU()

	// Verify input wasn't modified
	for i := range input.Data {
		if input.Data[i] != originalValues[i] {
			t.Errorf("Input was modified at index %d: expected %f, got %f",
				i, originalValues[i], input.Data[i])
		}
	}
}

// BenchmarkGELU benchmarks the GELU function
func BenchmarkGELU(b *testing.B) {
	input := NewTensor([]int{1000})
	for i := range input.Data {
		input.Data[i] = float32(i%10) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = input.GELU()
	}
}
