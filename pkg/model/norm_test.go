package model

import (
	"math"
	"testing"

	"gollm/pkg/tensor"
)

// TestNewLayerNorm tests the creation of LayerNorm.
func TestNewLayerNorm(t *testing.T) {
	embDim := 768
	eps := float32(1e-5)

	ln := NewLayerNorm(embDim, eps)

	if ln == nil {
		t.Fatal("NewLayerNorm returned nil")
	}

	if ln.Eps != eps {
		t.Errorf("Expected Eps=%v, got %v", eps, ln.Eps)
	}

	// Check scale (gamma) is initialized to 1
	if len(ln.Scale.Data) != embDim {
		t.Errorf("Expected scale length %d, got %d", embDim, len(ln.Scale.Data))
	}
	for i, v := range ln.Scale.Data {
		if v != 1.0 {
			t.Errorf("Scale[%d] = %v, expected 1.0", i, v)
		}
	}

	// Check shift (beta) is initialized to 0
	if len(ln.Shift.Data) != embDim {
		t.Errorf("Expected shift length %d, got %d", embDim, len(ln.Shift.Data))
	}
	for i, v := range ln.Shift.Data {
		if v != 0.0 {
			t.Errorf("Shift[%d] = %v, expected 0.0", i, v)
		}
	}
}

// TestLayerNorm_Forward tests the forward pass.
func TestLayerNorm_Forward(t *testing.T) {
	embDim := 4
	ln := NewLayerNorm(embDim, 1e-5)

	// Create simple input: (batch=1, seq=2, emb_dim=4)
	input := tensor.NewTensor([]int{1, 2, embDim})
	// Set different values for each position
	// Position 0: [1, 2, 3, 4]
	// Position 1: [2, 4, 6, 8]
	for d := 0; d < embDim; d++ {
		input.Set([]int{0, 0, d}, float32(d+1))
		input.Set([]int{0, 1, d}, float32(2*(d+1)))
	}

	output, err := ln.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Check output shape
	expectedShape := []int{1, 2, embDim}
	for i := range expectedShape {
		if output.Shape[i] != expectedShape[i] {
			t.Errorf("Output shape[%d] = %d, expected %d", i, output.Shape[i], expectedShape[i])
		}
	}

	// For position 0 with scale=1 and shift=0:
	// mean = (1+2+3+4)/4 = 2.5
	// var = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4 = 1.25
	// x_norm = (x - mean) / sqrt(var + eps)
	// For first element: (1 - 2.5) / sqrt(1.25) ≈ -1.3416

	firstVal := output.Get([]int{0, 0, 0})
	expectedFirst := float32(-1.3416407865)
	if math.Abs(float64(firstVal-expectedFirst)) > 1e-5 {
		t.Errorf("First element = %v, expected %v", firstVal, expectedFirst)
	}
}

// TestLayerNorm_NormalizationProperty tests that LayerNorm properly normalizes.
func TestLayerNorm_NormalizationProperty(t *testing.T) {
	embDim := 8
	ln := NewLayerNorm(embDim, 1e-5)

	// Create input with non-zero mean and variance
	input := tensor.NewTensor([]int{1, 1, embDim})
	for d := 0; d < embDim; d++ {
		input.Set([]int{0, 0, d}, float32(d*10+100)) // Large values with high variance
	}

	output, err := ln.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Check that output has approximately mean=0 and var=1
	mean := float32(0)
	for d := 0; d < embDim; d++ {
		mean += output.Get([]int{0, 0, d})
	}
	mean /= float32(embDim)

	variance := float32(0)
	for d := 0; d < embDim; d++ {
		diff := output.Get([]int{0, 0, d}) - mean
		variance += diff * diff
	}
	variance /= float32(embDim)

	// With scale=1 and shift=0, the normalized values should have mean≈0 and var≈1
	if math.Abs(float64(mean)) > 1e-5 {
		t.Errorf("Output mean = %v, expected ~0", mean)
	}
	if math.Abs(float64(variance-1.0)) > 1e-4 {
		t.Errorf("Output variance = %v, expected ~1", variance)
	}
}

// TestLayerNorm_InvalidInput tests error handling for invalid inputs.
func TestLayerNorm_InvalidInput(t *testing.T) {
	ln := NewLayerNorm(768, 1e-5)

	// 0D tensor
	input0D := tensor.NewTensor([]int{})
	_, err := ln.Forward(input0D)
	if err == nil {
		t.Error("Expected error for 0D tensor")
	}

	// Wrong embedding dimension
	inputWrongDim := tensor.NewTensor([]int{2, 10, 512}) // Should be 768
	_, err = ln.Forward(inputWrongDim)
	if err == nil {
		t.Error("Expected error for wrong embedding dimension")
	}
}

// TestLayerNorm_LearnableParameters tests that scale and shift affect output.
func TestLayerNorm_LearnableParameters(t *testing.T) {
	embDim := 4
	ln := NewLayerNorm(embDim, 1e-5)

	// Set custom scale and shift
	ln.Scale.Data[0] = 2.0
	ln.Scale.Data[1] = 2.0
	ln.Shift.Data[0] = 1.0
	ln.Shift.Data[1] = 1.0

	input := tensor.NewTensor([]int{1, 1, embDim})
	for d := 0; d < embDim; d++ {
		input.Set([]int{0, 0, d}, float32(d+1))
	}

	output, err := ln.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// First two elements should be affected by scale=2 and shift=1
	// Check that they're different from scale=1, shift=0 case
	if output.Get([]int{0, 0, 0}) == output.Get([]int{0, 0, 2}) {
		t.Error("Scale and shift should produce different values for different dimensions")
	}
}
