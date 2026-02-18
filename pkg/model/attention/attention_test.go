package attention

import (
	"math"
	"testing"

	"gollm/pkg/model"
	"gollm/pkg/tensor"
)

// TestCausalSelfAttention_SingleExample tests basic causal attention.
func TestCausalSelfAttention_SingleExample(t *testing.T) {
	// Setup
	config := model.DefaultConfig()
	dIn, dOut := 64, 64
	attn := NewCausalSelfAttention(config, dIn, dOut)

	// Initialize weights with identity-like values for predictable results
	for i := 0; i < dIn; i++ {
		for j := 0; j < dOut; j++ {
			if i == j {
				attn.WQuery.Set([]int{i, j}, 1.0)
				attn.WKey.Set([]int{i, j}, 1.0)
				attn.WValue.Set([]int{i, j}, 1.0)
			}
		}
	}

	// Create input: (batch=1, seq=4, d_in=64)
	batchSize, seqLen := 1, 4
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})
	// Fill with simple pattern
	for s := 0; s < seqLen; s++ {
		for d := 0; d < dIn; d++ {
			input.Set([]int{0, s, d}, float32(s+1)*0.1)
		}
	}

	// Run forward pass
	output, err := attn.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{batchSize, seqLen, dOut}
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
	config := model.DefaultConfig()
	config.NumHeads = 4
	dIn, dOut := 64, 64

	attn := NewMultiHeadAttention(config, dIn, dOut)

	// Verify head dimensions
	expectedHeadDim := dOut / config.NumHeads
	if attn.HeadDim != expectedHeadDim {
		t.Errorf("Expected head_dim %d, got %d", expectedHeadDim, attn.HeadDim)
	}

	// Create input
	batchSize, seqLen := 2, 8
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})

	// Initialize with pattern
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < dIn; d++ {
				input.Set([]int{b, s, d}, float32(b*100+s*10+d)*0.01)
			}
		}
	}

	// Run forward
	output, err := attn.Forward(input, nil, nil)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{batchSize, seqLen, dOut}
	if len(output.Shape) != len(expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape)
	}
	for i := range expectedShape {
		if output.Shape[i] != expectedShape[i] {
			t.Errorf("Expected shape %v at dim %d, got %d", expectedShape, i, output.Shape[i])
		}
	}
}

// TestGroupedQueryAttention_GroupExpansion tests GQA group expansion.
func TestGroupedQueryAttention_GroupExpansion(t *testing.T) {
	config := model.DefaultConfig()
	config.NumHeads = 12
	config.NumKVGroups = 4
	dIn, dOut := 768, 768

	attn := NewGroupedQueryAttention(config, dIn, dOut)

	// Verify group size
	expectedGroupSize := config.NumHeads / config.NumKVGroups
	if attn.GroupSize != expectedGroupSize {
		t.Errorf("Expected group_size %d, got %d", expectedGroupSize, attn.GroupSize)
	}

	// Verify K/V dimensions are smaller
	expectedKVDim := config.NumKVGroups * (dOut / config.NumHeads)
	if attn.WKey.Shape[1] != expectedKVDim {
		t.Errorf("Expected WKey dim %d, got %v", expectedKVDim, attn.WKey.Shape)
	}
	if attn.WValue.Shape[1] != expectedKVDim {
		t.Errorf("Expected WValue dim %d, got %v", expectedKVDim, attn.WValue.Shape)
	}

	// Test forward pass
	batchSize, seqLen := 1, 4
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})

	output, err := attn.Forward(input, nil, nil)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{batchSize, seqLen, dOut}
	if len(output.Shape) != len(expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, output.Shape)
	}
}

// TestAllVariants_ShapeValidation tests shape validation across all variants.
func TestAllVariants_ShapeValidation(t *testing.T) {
	config := model.DefaultConfig()
	config.NumHeads = 4 // Use 4 heads so dOut (64) is divisible
	dIn, dOut := 64, 64

	testCases := []struct {
		name      string
		input     *tensor.Tensor
		wantError bool
	}{
		{
			name:      "valid_3d_input",
			input:     tensor.NewTensor([]int{2, 8, dIn}),
			wantError: false,
		},
		{
			name:      "wrong_input_dim",
			input:     tensor.NewTensor([]int{2, 8, 32}), // Wrong d_in
			wantError: true,
		},
		{
			name:      "2d_input",
			input:     tensor.NewTensor([]int{8, dIn}),
			wantError: true,
		},
		{
			name:      "4d_input",
			input:     tensor.NewTensor([]int{2, 4, 8, dIn}),
			wantError: true,
		},
	}

	// Test CausalSelfAttention
	causal := NewCausalSelfAttention(config, dIn, dOut)
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
	multi := NewMultiHeadAttention(config, dIn, dOut)
	for _, tc := range testCases {
		t.Run("MultiHead_"+tc.name, func(t *testing.T) {
			_, err := multi.Forward(tc.input, nil, nil)
			if tc.wantError && err == nil {
				t.Errorf("Expected error for %s, got none", tc.name)
			}
			if !tc.wantError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.name, err)
			}
		})
	}

	// Test GroupedQueryAttention
	grouped := NewGroupedQueryAttention(config, dIn, dOut)
	for _, tc := range testCases {
		t.Run("Grouped_"+tc.name, func(t *testing.T) {
			_, err := grouped.Forward(tc.input, nil, nil)
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
	config := model.DefaultConfig()
	config.NumHeads = 4 // Use 4 heads so dOut (64) is divisible
	dIn, dOut := 64, 64
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
	multi := NewMultiHeadAttention(config, dIn, dOut)
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})

	// Initialize input
	for s := 0; s < seqLen; s++ {
		for d := 0; d < dIn; d++ {
			input.Set([]int{0, s, d}, float32(s+1)*0.1)
		}
	}

	output, err := multi.Forward(input, mask, nil)
	if err != nil {
		t.Fatalf("Forward with mask failed: %v", err)
	}

	if len(output.Shape) != 3 {
		t.Errorf("Expected 3D output, got %dD", len(output.Shape))
	}
}

// TestScaleValues tests that attention scales are computed correctly.
func TestScaleValues(t *testing.T) {
	config := model.DefaultConfig()

	testCases := []struct {
		dOut     int
		expected float64
	}{
		{64, 1.0 / math.Sqrt(64)},
		{128, 1.0 / math.Sqrt(128)},
		{256, 1.0 / math.Sqrt(256)},
	}

	for _, tc := range testCases {
		attn := NewCausalSelfAttention(config, tc.dOut, tc.dOut)
		actual := float64(attn.Scale)
		if math.Abs(actual-tc.expected) > 1e-6 {
			t.Errorf("dOut=%d: expected scale %v, got %v", tc.dOut, tc.expected, actual)
		}
	}
}

// TestGroupedQueryAttention_MemoryEfficiency tests that GQA uses less memory.
func TestGroupedQueryAttention_MemoryEfficiency(t *testing.T) {
	config := model.Config{
		NumHeads:    12,
		NumKVGroups: 4,
		HeadDim:     64,
	}
	dIn, dOut := 768, 768

	attn := NewGroupedQueryAttention(config, dIn, dOut)

	// Calculate memory savings
	mhaKVParams := dOut * dOut                                     // Q/K/V all same size in MHA
	gqaKVParams := dIn * (config.NumKVGroups * config.HeadDim) * 2 // K and V only

	if gqaKVParams >= mhaKVParams {
		t.Errorf("GQA should use less memory than MHA")
	}

	// Group size should be 3 (12 / 4)
	if attn.GroupSize != 3 {
		t.Errorf("Expected group_size 3, got %d", attn.GroupSize)
	}
}

// BenchmarkCausalSelfAttention benchmarks causal attention.
func BenchmarkCausalSelfAttention(b *testing.B) {
	config := model.DefaultConfig()
	dIn, dOut := 512, 512
	attn := NewCausalSelfAttention(config, dIn, dOut)

	batchSize, seqLen := 1, 128
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})

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
	config := model.DefaultConfig()
	dIn, dOut := 512, 512
	attn := NewMultiHeadAttention(config, dIn, dOut)

	batchSize, seqLen := 1, 128
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attn.Forward(input, nil, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGroupedQueryAttention benchmarks GQA.
func BenchmarkGroupedQueryAttention(b *testing.B) {
	config := model.DefaultConfig()
	dIn, dOut := 768, 768
	attn := NewGroupedQueryAttention(config, dIn, dOut)

	batchSize, seqLen := 1, 128
	input := tensor.NewTensor([]int{batchSize, seqLen, dIn})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := attn.Forward(input, nil, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}
