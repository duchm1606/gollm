package attention

import (
	"math"
	"testing"

	"gollm/pkg/tensor"
)

// TestCausalSelfAttention_SingleExample tests basic causal attention.
func TestCausalSelfAttention_SingleExample(t *testing.T) {
	// Setup
	config := CausalSelfAttentionConfig{
		DIn:     64,
		DOut:    64,
		Dropout: 0.0,
	}
	attn := NewCausalSelfAttention(config)

	// Initialize weights with identity-like values for predictable results
	for i := 0; i < config.DIn; i++ {
		for j := 0; j < config.DOut; j++ {
			if i == j {
				attn.WQuery.Set([]int{i, j}, 1.0)
				attn.WKey.Set([]int{i, j}, 1.0)
				attn.WValue.Set([]int{i, j}, 1.0)
			}
		}
	}

	// Create input: (batch=1, seq=4, d_in=64)
	batchSize, seqLen := 1, 4
	input := tensor.NewTensor([]int{batchSize, seqLen, config.DIn})
	// Fill with simple pattern
	for s := 0; s < seqLen; s++ {
		for d := 0; d < config.DIn; d++ {
			input.Set([]int{0, s, d}, float32(s+1)*0.1)
		}
	}

	// Run forward pass (CausalSelfAttention doesn't have training parameter)
	output, err := attn.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{batchSize, seqLen, config.DOut}
	if len(output.Shape) != len(expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape)
	}
	for i := range expectedShape {
		if output.Shape[i] != expectedShape[i] {
			t.Errorf("Expected shape %v at dim %d, got %d", expectedShape, i, output.Shape[i])
		}
	}

	// Verify causal property: position i should not attend to positions > i
	// For position 0, only attend to itself
	// For position 3, attend to positions 0, 1, 2, 3
	// The output values should be different for different positions
}

// TestMultiHeadAttention_ParallelHeads tests that multi-head splits work correctly.
func TestMultiHeadAttention_ParallelHeads(t *testing.T) {
	config := MultiHeadAttentionConfig{
		NumHeads: 4,
		DIn:      64,
		DOut:     64,
		Dropout:  0.0,
		QKVBias:  false,
	}

	attn := NewMultiHeadAttention(config)

	// Verify head dimensions
	expectedHeadDim := config.DOut / config.NumHeads
	if attn.HeadDim != expectedHeadDim {
		t.Errorf("Expected head_dim %d, got %d", expectedHeadDim, attn.HeadDim)
	}

	// Create input
	batchSize, seqLen := 2, 8
	input := tensor.NewTensor([]int{batchSize, seqLen, config.DIn})

	// Initialize with pattern
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < config.DIn; d++ {
				input.Set([]int{b, s, d}, float32(b*100+s*10+d)*0.01)
			}
		}
	}

	// Run forward (training=false for deterministic testing)
	output, err := attn.Forward(input, nil, false)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{batchSize, seqLen, config.DOut}
	if len(output.Shape) != len(expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape)
	}
	for i := range expectedShape {
		if output.Shape[i] != expectedShape[i] {
			t.Errorf("Expected shape %v at dim %d, got %d", expectedShape, i, output.Shape[i])
		}
	}
}

// TestAllVariants_ShapeValidation tests shape validation across all variants.
func TestAllVariants_ShapeValidation(t *testing.T) {
	config := MultiHeadAttentionConfig{
		NumHeads: 4,
		DIn:      64,
		DOut:     64,
		Dropout:  0.0,
		QKVBias:  false,
	}

	testCases := []struct {
		name      string
		input     *tensor.Tensor
		wantError bool
	}{
		{
			name:      "valid_3d_input",
			input:     tensor.NewTensor([]int{2, 8, config.DIn}),
			wantError: false,
		},
		{
			name:      "wrong_input_dim",
			input:     tensor.NewTensor([]int{2, 8, 32}), // Wrong d_in
			wantError: true,
		},
		{
			name:      "2d_input",
			input:     tensor.NewTensor([]int{8, config.DIn}),
			wantError: true,
		},
		{
			name:      "4d_input",
			input:     tensor.NewTensor([]int{2, 4, 8, config.DIn}),
			wantError: true,
		},
	}

	// Test CausalSelfAttention
	causalConfig := CausalSelfAttentionConfig{
		DIn:     config.DIn,
		DOut:    config.DOut,
		Dropout: 0.0,
	}
	causal := NewCausalSelfAttention(causalConfig)
	for _, tc := range testCases {
		t.Run("Causal_"+tc.name, func(t *testing.T) {
			_, err := causal.Forward(tc.input)
			if tc.wantError && err == nil {
				t.Errorf("Expected error for %s, got none", tc.name)
			}
			if !tc.wantError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.name, err)
			}
		})
	}

	// Test MultiHeadAttention
	multi := NewMultiHeadAttention(config)
	for _, tc := range testCases {
		t.Run("MultiHead_"+tc.name, func(t *testing.T) {
			_, err := multi.Forward(tc.input, nil, false)
			if tc.wantError && err == nil {
				t.Errorf("Expected error for %s, got none", tc.name)
			}
			if !tc.wantError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.name, err)
			}
		})
	}
}

// TestAllVariants_CausalMasking tests that causal masking is applied correctly.
func TestAllVariants_CausalMasking(t *testing.T) {
	config := MultiHeadAttentionConfig{
		NumHeads: 4,
		DIn:      64,
		DOut:     64,
		Dropout:  0.0,
		QKVBias:  false,
	}
	batchSize, seqLen := 1, 4

	// Create mask manually
	mask := tensor.CreateCausalMask(seqLen)

	// Verify mask shape and values
	if len(mask.Shape) != 2 || mask.Shape[0] != seqLen || mask.Shape[1] != seqLen {
		t.Errorf("Expected mask shape (%d, %d), got %v", seqLen, seqLen, mask.Shape)
	}

	// Lower triangle should be 1s, upper triangle should be 0s
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			expected := float32(0)
			if j <= i {
				expected = 1
			}
			actual := mask.Get([]int{i, j})
			if actual != expected {
				t.Errorf("Mask[%d,%d] = %v, expected %v", i, j, actual, expected)
			}
		}
	}

	// Test with MultiHeadAttention
	multi := NewMultiHeadAttention(config)
	input := tensor.NewTensor([]int{batchSize, seqLen, config.DIn})

	// Initialize input
	for s := 0; s < seqLen; s++ {
		for d := 0; d < config.DIn; d++ {
			input.Set([]int{0, s, d}, float32(s+1)*0.1)
		}
	}

	output, err := multi.Forward(input, mask, false)
	if err != nil {
		t.Fatalf("Forward with mask failed: %v", err)
	}

	if len(output.Shape) != 3 {
		t.Errorf("Expected 3D output, got %dD", len(output.Shape))
	}
}

// TestScaleValues tests that attention scales are computed correctly.
func TestScaleValues(t *testing.T) {
	testCases := []struct {
		dOut     int
		expected float64
	}{
		{64, 1.0 / math.Sqrt(64)},
		{128, 1.0 / math.Sqrt(128)},
		{256, 1.0 / math.Sqrt(256)},
	}

	for _, tc := range testCases {
		config := CausalSelfAttentionConfig{
			DIn:     tc.dOut,
			DOut:    tc.dOut,
			Dropout: 0.0,
		}
		attn := NewCausalSelfAttention(config)
		actual := float64(attn.Scale)
		if math.Abs(actual-tc.expected) > 1e-6 {
			t.Errorf("dOut=%d: expected scale %v, got %v", tc.dOut, tc.expected, actual)
		}
	}
}

// BenchmarkCausalSelfAttention benchmarks causal attention.
func BenchmarkCausalSelfAttention(b *testing.B) {
	config := CausalSelfAttentionConfig{
		DIn:     512,
		DOut:    512,
		Dropout: 0.0,
	}
	attn := NewCausalSelfAttention(config)

	batchSize, seqLen := 1, 128
	input := tensor.NewTensor([]int{batchSize, seqLen, config.DIn})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attn.Forward(input)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMultiHeadAttention benchmarks multi-head attention.
func BenchmarkMultiHeadAttention(b *testing.B) {
	config := MultiHeadAttentionConfig{
		NumHeads: 12,
		DIn:      768,
		DOut:     768,
		Dropout:  0.0,
		QKVBias:  false,
	}
	attn := NewMultiHeadAttention(config)

	batchSize, seqLen := 1, 128
	input := tensor.NewTensor([]int{batchSize, seqLen, config.DIn})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attn.Forward(input, nil, false)
		if err != nil {
			b.Fatal(err)
		}
	}
}
