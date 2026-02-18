package model

import (
	"math"
	"testing"

	"gollm/pkg/tensor"
)

// TestComputeRoPE_Validation tests that ComputeRoPE validates inputs correctly
func TestComputeRoPE_Validation(t *testing.T) {
	tests := []struct {
		name      string
		headDim   int
		maxSeqLen int
		thetaBase float32
		wantErr   bool
	}{
		{
			name:      "valid parameters",
			headDim:   64,
			maxSeqLen: 2048,
			thetaBase: 10000.0,
			wantErr:   false,
		},
		{
			name:      "odd head_dim",
			headDim:   63,
			maxSeqLen: 2048,
			thetaBase: 10000.0,
			wantErr:   true,
		},
		{
			name:      "zero max_seq_len",
			headDim:   64,
			maxSeqLen: 0,
			thetaBase: 10000.0,
			wantErr:   true,
		},
		{
			name:      "negative max_seq_len",
			headDim:   64,
			maxSeqLen: -1,
			thetaBase: 10000.0,
			wantErr:   true,
		},
		{
			name:      "zero theta_base",
			headDim:   64,
			maxSeqLen: 2048,
			thetaBase: 0.0,
			wantErr:   true,
		},
		{
			name:      "negative theta_base",
			headDim:   64,
			maxSeqLen: 2048,
			thetaBase: -10000.0,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rope, err := ComputeRoPE(tt.headDim, tt.maxSeqLen, tt.thetaBase)
			if tt.wantErr {
				if err == nil {
					t.Errorf("ComputeRoPE() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ComputeRoPE() unexpected error: %v", err)
				return
			}
			if rope == nil {
				t.Errorf("ComputeRoPE() returned nil rope without error")
			}
		})
	}
}

// TestComputeRoPE_Shape tests that ComputeRoPE produces correct shapes
func TestComputeRoPE_Shape(t *testing.T) {
	headDim := 64
	maxSeqLen := 128
	thetaBase := float32(10000.0)

	rope, err := ComputeRoPE(headDim, maxSeqLen, thetaBase)
	if err != nil {
		t.Fatalf("ComputeRoPE() error: %v", err)
	}

	// Check that fields are set correctly
	if rope.MaxSeqLen != maxSeqLen {
		t.Errorf("MaxSeqLen = %d, want %d", rope.MaxSeqLen, maxSeqLen)
	}
	if rope.HeadDim != headDim {
		t.Errorf("HeadDim = %d, want %d", rope.HeadDim, headDim)
	}

	// Check array sizes
	expectedSize := maxSeqLen * headDim
	if len(rope.Cos) != expectedSize {
		t.Errorf("len(Cos) = %d, want %d", len(rope.Cos), expectedSize)
	}
	if len(rope.Sin) != expectedSize {
		t.Errorf("len(Sin) = %d, want %d", len(rope.Sin), expectedSize)
	}
}

// TestComputeRoPE_Values tests that ComputeRoPE produces correct values
// by comparing against manual calculation
func TestComputeRoPE_Values(t *testing.T) {
	headDim := 8
	maxSeqLen := 4
	thetaBase := float32(10000.0)

	rope, err := ComputeRoPE(headDim, maxSeqLen, thetaBase)
	if err != nil {
		t.Fatalf("ComputeRoPE() error: %v", err)
	}

	// For head_dim=8, num_freqs=4
	// inv_freq[i] = 1.0 / (thetaBase^(2*i/head_dim))
	// Position 1, frequency 0: angle = 1 * 1.0 / (10000^(0/8)) = 1 * 1.0 = 1.0
	// Position 1, frequency 1: angle = 1 * 1.0 / (10000^(2/8)) = 1 * 0.1 = 0.1

	// Check that position 0 has cos=1, sin=0 (no rotation at position 0)
	for i := 0; i < headDim; i++ {
		cosVal := rope.Cos[i]
		sinVal := rope.Sin[i]
		if math.Abs(float64(cosVal-1.0)) > 1e-6 {
			t.Errorf("position 0, dim %d: cos = %f, want 1.0", i, cosVal)
		}
		if math.Abs(float64(sinVal)) > 1e-6 {
			t.Errorf("position 0, dim %d: sin = %f, want 0.0", i, sinVal)
		}
	}

	// Check that first half equals second half (split-halves duplication)
	halfDim := headDim / 2
	for pos := 0; pos < maxSeqLen; pos++ {
		baseIdx := pos * headDim
		for i := 0; i < halfDim; i++ {
			cos1 := rope.Cos[baseIdx+i]
			cos2 := rope.Cos[baseIdx+i+halfDim]
			sin1 := rope.Sin[baseIdx+i]
			sin2 := rope.Sin[baseIdx+i+halfDim]

			if math.Abs(float64(cos1-cos2)) > 1e-6 {
				t.Errorf("position %d, dims %d and %d: cos values differ (%f vs %f)",
					pos, i, i+halfDim, cos1, cos2)
			}
			if math.Abs(float64(sin1-sin2)) > 1e-6 {
				t.Errorf("position %d, dims %d and %d: sin values differ (%f vs %f)",
					pos, i, i+halfDim, sin1, sin2)
			}
		}
	}

	// Check values monotonically change with position (for a fixed frequency)
	// Higher positions should have different angles
	freqIdx := 0
	prevAngle := 0.0
	for pos := 0; pos < maxSeqLen; pos++ {
		baseIdx := pos * headDim
		cosVal := float64(rope.Cos[baseIdx+freqIdx])
		sinVal := float64(rope.Sin[baseIdx+freqIdx])
		angle := math.Atan2(sinVal, cosVal)

		if pos > 0 && angle <= prevAngle {
			t.Errorf("position %d: angle %f not greater than previous %f",
				pos, angle, prevAngle)
		}
		prevAngle = angle
	}
}

// TestApplyRoPE_Shape tests that ApplyRoPE maintains tensor shape
func TestApplyRoPE_Shape(t *testing.T) {
	rope, err := ComputeRoPE(64, 128, 10000.0)
	if err != nil {
		t.Fatalf("ComputeRoPE() error: %v", err)
	}

	tests := []struct {
		name      string
		shape     []int
		wantError bool
	}{
		{
			name:      "standard 4D tensor",
			shape:     []int{2, 12, 10, 64},
			wantError: false,
		},
		{
			name:      "single batch single head",
			shape:     []int{1, 1, 5, 64},
			wantError: false,
		},
		{
			name:      "batch=1, heads=1, seq=1",
			shape:     []int{1, 1, 1, 64},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input tensor with some values
			data := make([]float32, 1)
			for _, dim := range tt.shape {
				data = make([]float32, len(data)*dim)
			}
			// Initialize with sequential values
			for i := range data {
				data[i] = float32(i) * 0.01
			}

			x := tensor.NewTensorFromData(data, tt.shape)

			result, err := ApplyRoPE(x, rope, 0)
			if tt.wantError {
				if err == nil {
					t.Errorf("ApplyRoPE() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("ApplyRoPE() unexpected error: %v", err)
				return
			}

			// Check shape is preserved
			if len(result.Shape) != len(tt.shape) {
				t.Errorf("output rank = %d, want %d", len(result.Shape), len(tt.shape))
				return
			}
			for i := range tt.shape {
				if result.Shape[i] != tt.shape[i] {
					t.Errorf("output shape[%d] = %d, want %d", i, result.Shape[i], tt.shape[i])
				}
			}
		})
	}
}

// TestApplyRoPE_InvalidShape tests error handling for invalid input shapes
func TestApplyRoPE_InvalidShape(t *testing.T) {
	rope, _ := ComputeRoPE(64, 128, 10000.0)

	tests := []struct {
		name   string
		shape  []int
		offset int
	}{
		{
			name:   "3D tensor",
			shape:  []int{2, 10, 64},
			offset: 0,
		},
		{
			name:   "wrong head_dim",
			shape:  []int{1, 1, 5, 32},
			offset: 0,
		},
		{
			name:   "offset too large",
			shape:  []int{1, 1, 10, 64},
			offset: 120, // max_seq_len=128, offset+seq_len=130 > 128
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := tensor.NewTensor(tt.shape)
			_, err := ApplyRoPE(x, rope, tt.offset)
			if err == nil {
				t.Errorf("ApplyRoPE() expected error for shape %v, offset %d", tt.shape, tt.offset)
			}
		})
	}
}

// TestApplyRoPE_Rotation tests that rotation actually changes values
func TestApplyRoPE_Rotation(t *testing.T) {
	headDim := 8
	rope, _ := ComputeRoPE(headDim, 16, 10000.0)

	// Create a simple input tensor
	// Shape: (1, 1, 2, 8) - batch=1, heads=1, seq=2, head_dim=8
	data := []float32{
		// Position 0
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
		// Position 1
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
	}
	x := tensor.NewTensorFromData(data, []int{1, 1, 2, headDim})

	result, err := ApplyRoPE(x, rope, 0)
	if err != nil {
		t.Fatalf("ApplyRoPE() error: %v", err)
	}

	// Position 0: cos=1, sin=0, so values should be unchanged
	for i := 0; i < headDim; i++ {
		inputVal := x.Data[i]
		outputVal := result.Data[i]
		if math.Abs(float64(inputVal-outputVal)) > 1e-6 {
			t.Errorf("position 0, dim %d: value changed from %f to %f (should be unchanged)",
				i, inputVal, outputVal)
		}
	}

	// Position 1: should have some rotation
	hasChange := false
	for i := headDim; i < 2*headDim; i++ {
		inputVal := x.Data[i]
		outputVal := result.Data[i]
		if math.Abs(float64(inputVal-outputVal)) > 1e-6 {
			hasChange = true
			break
		}
	}
	if !hasChange {
		t.Errorf("position 1: values unchanged after rotation")
	}
}

// TestApplyRoPE_Offset tests the offset parameter for KV cache support
func TestApplyRoPE_Offset(t *testing.T) {
	headDim := 8
	rope, _ := ComputeRoPE(headDim, 16, 10000.0)

	// Create input with 2 positions
	data := []float32{
		// Position 0
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
		// Position 1
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
	}
	x := tensor.NewTensorFromData(data, []int{1, 1, 2, headDim})

	// Apply RoPE with offset=0
	result1, err := ApplyRoPE(x, rope, 0)
	if err != nil {
		t.Fatalf("ApplyRoPE() with offset=0 error: %v", err)
	}

	// Apply RoPE with offset=1 to a tensor with just position 1
	data2 := []float32{
		// Position 1 (same values as before)
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
	}
	x2 := tensor.NewTensorFromData(data2, []int{1, 1, 1, headDim})
	result2, err := ApplyRoPE(x2, rope, 1)
	if err != nil {
		t.Fatalf("ApplyRoPE() with offset=1 error: %v", err)
	}

	// The second position in result1 should match the first position in result2
	// (both use position 1's cos/sin values)
	for i := 0; i < headDim; i++ {
		val1 := result1.Data[headDim+i] // Position 1 in first result
		val2 := result2.Data[i]         // Position 0 (offset=1) in second result
		if math.Abs(float64(val1-val2)) > 1e-6 {
			t.Errorf("offset mismatch at dim %d: offset=0 gives %f, offset=1 gives %f",
				i, val1, val2)
		}
	}
}

// TestApplyRoPE_ManualCalculation tests against a manual calculation
func TestApplyRoPE_ManualCalculation(t *testing.T) {
	headDim := 4
	rope, _ := ComputeRoPE(headDim, 8, 10000.0)

	// Create a simple input: [1, 2, 3, 4] at position 1
	// Shape: (1, 1, 1, 4)
	data := []float32{1.0, 2.0, 3.0, 4.0}
	x := tensor.NewTensorFromData(data, []int{1, 1, 1, headDim})

	result, err := ApplyRoPE(x, rope, 1) // Apply at position 1
	if err != nil {
		t.Fatalf("ApplyRoPE() error: %v", err)
	}

	// Get cos and sin for position 1
	cosVals := GetCosSlice(rope, 1)
	sinVals := GetSinSlice(rope, 1)

	// Manual calculation for split-halves style
	// x = [1, 2, 3, 4]
	// x1 = [1, 2], x2 = [3, 4]
	// rotated = [-x2, x1] = [-3, -4, 1, 2]
	// For dim 0: cos[0]*1 - sin[0]*3 = 1*cos[0] - 3*sin[0]
	// For dim 2: cos[0]*3 + sin[0]*1 = 3*cos[0] + 1*sin[0]

	// Check first pair (dims 0 and 2)
	x1, x2 := 1.0, 3.0
	cos0, sin0 := cosVals[0], sinVals[0]
	expected0 := x1*float64(cos0) - x2*float64(sin0)
	expected2 := x2*float64(cos0) + x1*float64(sin0)

	actual0 := float64(result.Data[0])
	actual2 := float64(result.Data[2])

	tolerance := 1e-6
	if math.Abs(actual0-expected0) > tolerance {
		t.Errorf("dim 0: expected %f, got %f", expected0, actual0)
	}
	if math.Abs(actual2-expected2) > tolerance {
		t.Errorf("dim 2: expected %f, got %f", expected2, actual2)
	}

	// Check second pair (dims 1 and 3)
	x1, x2 = 2.0, 4.0
	cos1, sin1 := cosVals[1], sinVals[1]
	expected1 := x1*float64(cos1) - x2*float64(sin1)
	expected3 := x2*float64(cos1) + x1*float64(sin1)

	actual1 := float64(result.Data[1])
	actual3 := float64(result.Data[3])

	if math.Abs(actual1-expected1) > tolerance {
		t.Errorf("dim 1: expected %f, got %f", expected1, actual1)
	}
	if math.Abs(actual3-expected3) > tolerance {
		t.Errorf("dim 3: expected %f, got %f", expected3, actual3)
	}
}

// TestApplyRoPE_MultipleBatchesAndHeads tests with multiple batches and heads
func TestApplyRoPE_MultipleBatchesAndHeads(t *testing.T) {
	headDim := 8
	rope, _ := ComputeRoPE(headDim, 16, 10000.0)

	// Shape: (batch=2, heads=3, seq=4, head_dim=8)
	batchSize, numHeads, seqLen := 2, 3, 4
	totalSize := batchSize * numHeads * seqLen * headDim
	data := make([]float32, totalSize)
	for i := range data {
		data[i] = float32(i%10) * 0.1 // Some test values
	}
	x := tensor.NewTensorFromData(data, []int{batchSize, numHeads, seqLen, headDim})

	result, err := ApplyRoPE(x, rope, 0)
	if err != nil {
		t.Fatalf("ApplyRoPE() error: %v", err)
	}

	// Verify shape
	if len(result.Shape) != 4 {
		t.Errorf("result rank = %d, want 4", len(result.Shape))
	}
	expectedShape := []int{batchSize, numHeads, seqLen, headDim}
	for i := range expectedShape {
		if result.Shape[i] != expectedShape[i] {
			t.Errorf("result.Shape[%d] = %d, want %d", i, result.Shape[i], expectedShape[i])
		}
	}

	// Verify that different positions have different rotations
	// Sample position 0 and position 2 from the first head of the first batch
	getPosition := func(pos int) []float32 {
		baseIdx := pos * headDim
		return result.Data[baseIdx : baseIdx+headDim]
	}

	pos0 := getPosition(0)
	pos2 := getPosition(2)

	// They should be different (pos0 has no rotation, pos2 has some)
	allSame := true
	for i := 0; i < headDim; i++ {
		if math.Abs(float64(pos0[i]-pos2[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("positions 0 and 2 have identical values after rotation")
	}
}

// TestApplyRoPE_PositionZeroUnchanged tests that position 0 is unchanged
func TestApplyRoPE_PositionZeroUnchanged(t *testing.T) {
	headDim := 8
	rope, _ := ComputeRoPE(headDim, 16, 10000.0)

	// Create a tensor with multiple sequences, all starting from position 0
	data := []float32{
		// Seq 1
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
		// Seq 2
		0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
	}
	x := tensor.NewTensorFromData(data, []int{1, 1, 2, headDim})

	result, err := ApplyRoPE(x, rope, 0)
	if err != nil {
		t.Fatalf("ApplyRoPE() error: %v", err)
	}

	// Both sequences should have their first position unchanged
	// because cos[0]=1 and sin[0]=0 for all dimensions
	tolerance := 1e-6
	for seq := 0; seq < 2; seq++ {
		baseIdx := seq * headDim
		for i := 0; i < headDim; i++ {
			inputVal := x.Data[baseIdx+i]
			outputVal := result.Data[baseIdx+i]
			if math.Abs(float64(inputVal-outputVal)) > tolerance {
				t.Errorf("seq %d, dim %d: position 0 changed from %f to %f",
					seq, i, inputVal, outputVal)
			}
		}
	}
}

// TestComputeRoPE_DifferentTheta tests that different theta_base values
// produce different frequency schedules
func TestComputeRoPE_DifferentTheta(t *testing.T) {
	headDim := 8
	maxSeqLen := 8

	rope1, _ := ComputeRoPE(headDim, maxSeqLen, 10000.0)
	rope2, _ := ComputeRoPE(headDim, maxSeqLen, 500000.0)

	// The cos/sin values should be different
	allSame := true
	for i := range rope1.Cos {
		if math.Abs(float64(rope1.Cos[i]-rope2.Cos[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("different theta_base values produced identical cos values")
	}
}

// TestGetCosSlice and TestGetSinSlice test the helper functions
func TestGetCosSlice(t *testing.T) {
	headDim := 4
	rope, _ := ComputeRoPE(headDim, 8, 10000.0)

	slice := GetCosSlice(rope, 3)
	if len(slice) != headDim {
		t.Errorf("GetCosSlice length = %d, want %d", len(slice), headDim)
	}

	// Verify it matches the internal data
	baseIdx := 3 * headDim
	for i := 0; i < headDim; i++ {
		if slice[i] != rope.Cos[baseIdx+i] {
			t.Errorf("GetCosSlice[%d] = %f, want %f", i, slice[i], rope.Cos[baseIdx+i])
		}
	}
}

func TestGetSinSlice(t *testing.T) {
	headDim := 4
	rope, _ := ComputeRoPE(headDim, 8, 10000.0)

	slice := GetSinSlice(rope, 5)
	if len(slice) != headDim {
		t.Errorf("GetSinSlice length = %d, want %d", len(slice), headDim)
	}

	// Verify it matches the internal data
	baseIdx := 5 * headDim
	for i := 0; i < headDim; i++ {
		if slice[i] != rope.Sin[baseIdx+i] {
			t.Errorf("GetSinSlice[%d] = %f, want %f", i, slice[i], rope.Sin[baseIdx+i])
		}
	}
}

// BenchmarkComputeRoPE benchmarks the precomputation
func BenchmarkComputeRoPE(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := ComputeRoPE(128, 2048, 500000.0)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkApplyRoPE benchmarks the application of RoPE
func BenchmarkApplyRoPE(b *testing.B) {
	rope, _ := ComputeRoPE(128, 2048, 500000.0)
	x := tensor.NewTensor([]int{4, 32, 512, 128}) // batch=4, heads=32, seq=512, head_dim=128

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ApplyRoPE(x, rope, 0)
		if err != nil {
			b.Fatal(err)
		}
	}
}
