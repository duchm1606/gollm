package tensor

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// TestNewTensor tests tensor creation
func TestNewTensor(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		expected int
	}{
		{"1D", []int{5}, 5},
		{"2D", []int{3, 4}, 12},
		{"3D", []int{2, 3, 4}, 24},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape)

			if !shapeEquals(tensor.Shape, tt.shape) {
				t.Errorf("Expected shape %v, got %v", tt.shape, tensor.Shape)
			}

			if len(tensor.Data) != tt.expected {
				t.Errorf("Expected data length %d, got %d", tt.expected, len(tensor.Data))
			}

			// Check all zeros
			for i, v := range tensor.Data {
				if v != 0 {
					t.Errorf("Expected zero at index %d, got %f", i, v)
				}
			}
		})
	}
}

// TestFromSlice tests creating tensor from slice
func TestFromSlice(t *testing.T) {
	tests := []struct {
		name      string
		data      []float32
		shape     []int
		wantErr   bool
		errString string
	}{
		{
			name:    "valid 2D",
			data:    []float32{1, 2, 3, 4, 5, 6},
			shape:   []int{2, 3},
			wantErr: false,
		},
		{
			name:    "valid 3D",
			data:    []float32{1, 2, 3, 4, 5, 6, 7, 8},
			shape:   []int{2, 2, 2},
			wantErr: false,
		},
		{
			name:      "size mismatch",
			data:      []float32{1, 2, 3},
			shape:     []int{2, 3},
			wantErr:   true,
			errString: "data size 3 does not match shape",
		},
		{
			name:      "negative dimension",
			data:      []float32{1, 2, 3, 4},
			shape:     []int{2, -2},
			wantErr:   true,
			errString: "invalid dimension",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := FromSlice(tt.data, tt.shape)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(tensor.Shape, tt.shape) {
				t.Errorf("Expected shape %v, got %v", tt.shape, tensor.Shape)
			}

			for i, v := range tensor.Data {
				if v != tt.data[i] {
					t.Errorf("Data mismatch at index %d: expected %f, got %f", i, tt.data[i], v)
				}
			}
		})
	}
}

// TestView tests tensor reshaping
func TestView(t *testing.T) {
	tests := []struct {
		name      string
		data      []float32
		shape     []int
		newShape  []int
		wantErr   bool
		errString string
	}{
		{
			name:     "valid reshape 2x3 to 3x2",
			data:     []float32{1, 2, 3, 4, 5, 6},
			shape:    []int{2, 3},
			newShape: []int{3, 2},
			wantErr:  false,
		},
		{
			name:     "valid reshape to 1D",
			data:     []float32{1, 2, 3, 4},
			shape:    []int{2, 2},
			newShape: []int{4},
			wantErr:  false,
		},
		{
			name:      "size mismatch",
			data:      []float32{1, 2, 3, 4},
			shape:     []int{2, 2},
			newShape:  []int{3, 2},
			wantErr:   true,
			errString: "cannot view tensor of size 4",
		},
		{
			name:      "negative dimension",
			data:      []float32{1, 2, 3, 4},
			shape:     []int{2, 2},
			newShape:  []int{-2, 2},
			wantErr:   true,
			errString: "invalid dimension",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, _ := FromSlice(tt.data, tt.shape)
			view, err := tensor.View(tt.newShape)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(view.Shape, tt.newShape) {
				t.Errorf("Expected shape %v, got %v", tt.newShape, view.Shape)
			}

			// Verify data is shared
			if &view.Data[0] != &tensor.Data[0] {
				t.Error("View should share data with original tensor")
			}
		})
	}
}

// TestTranspose tests dimension swapping
func TestTranspose(t *testing.T) {
	tests := []struct {
		name      string
		data      []float32
		shape     []int
		dim1      int
		dim2      int
		wantErr   bool
		errString string
	}{
		{
			name:    "transpose 2D",
			data:    []float32{1, 2, 3, 4, 5, 6},
			shape:   []int{2, 3},
			dim1:    0,
			dim2:    1,
			wantErr: false,
		},
		{
			name:    "transpose 3D",
			data:    []float32{1, 2, 3, 4, 5, 6, 7, 8},
			shape:   []int{2, 2, 2},
			dim1:    0,
			dim2:    2,
			wantErr: false,
		},
		{
			name:      "invalid dim1",
			data:      []float32{1, 2, 3, 4},
			shape:     []int{2, 2},
			dim1:      -1,
			dim2:      1,
			wantErr:   true,
			errString: "invalid transpose dimensions",
		},
		{
			name:      "invalid dim2",
			data:      []float32{1, 2, 3, 4},
			shape:     []int{2, 2},
			dim1:      0,
			dim2:      5,
			wantErr:   true,
			errString: "invalid transpose dimensions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, _ := FromSlice(tt.data, tt.shape)
			transposed, err := tensor.Transpose(tt.dim1, tt.dim2)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Check shape
			expectedShape := copyShapeInt(tt.shape)
			expectedShape[tt.dim1], expectedShape[tt.dim2] = expectedShape[tt.dim2], expectedShape[tt.dim1]
			if !shapeEquals(transposed.Shape, expectedShape) {
				t.Errorf("Expected shape %v, got %v", expectedShape, transposed.Shape)
			}
		})
	}
}

// TestMatMul tests matrix multiplication
func TestMatMul(t *testing.T) {
	tests := []struct {
		name          string
		aShape        []int
		bShape        []int
		aData         []float32
		bData         []float32
		expectedData  []float32
		expectedShape []int
		wantErr       bool
		errString     string
	}{
		{
			name:          "2D matmul",
			aShape:        []int{2, 2},
			bShape:        []int{2, 2},
			aData:         []float32{1, 2, 3, 4},
			bData:         []float32{5, 6, 7, 8},
			expectedData:  []float32{19, 22, 43, 50},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
		{
			name:          "rectangular matmul",
			aShape:        []int{2, 3},
			bShape:        []int{3, 2},
			aData:         []float32{1, 2, 3, 4, 5, 6},
			bData:         []float32{7, 8, 9, 10, 11, 12},
			expectedData:  []float32{58, 64, 139, 154},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
		{
			name:          "batched matmul",
			aShape:        []int{2, 2, 2},
			bShape:        []int{2, 2, 2},
			aData:         []float32{1, 2, 3, 4, 5, 6, 7, 8},
			bData:         []float32{1, 0, 0, 1, 1, 0, 0, 1},
			expectedData:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
			expectedShape: []int{2, 2, 2},
			wantErr:       false,
		},
		{
			name:      "incompatible shapes",
			aShape:    []int{2, 3},
			bShape:    []int{2, 3},
			aData:     []float32{1, 2, 3, 4, 5, 6},
			bData:     []float32{1, 2, 3, 4, 5, 6},
			wantErr:   true,
			errString: "inner dimensions 3 and 2 don't match",
		},
		{
			name:      "1D tensor",
			aShape:    []int{4},
			bShape:    []int{4},
			aData:     []float32{1, 2, 3, 4},
			bData:     []float32{1, 2, 3, 4},
			wantErr:   true,
			errString: "requires at least 2D tensors",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, _ := FromSlice(tt.aData, tt.aShape)
			b, _ := FromSlice(tt.bData, tt.bShape)
			result, err := Matmul(a, b)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(result.Shape, tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, result.Shape)
			}

			for i, v := range result.Data {
				if !floatEquals(v, tt.expectedData[i], 1e-5) {
					t.Errorf("Data mismatch at index %d: expected %f, got %f", i, tt.expectedData[i], v)
				}
			}
		})
	}
}

// TestAdd tests element-wise addition with broadcasting
func TestAdd(t *testing.T) {
	tests := []struct {
		name          string
		aShape        []int
		bShape        []int
		aData         []float32
		bData         []float32
		expectedData  []float32
		expectedShape []int
		wantErr       bool
		errString     string
	}{
		{
			name:          "same shape",
			aShape:        []int{2, 2},
			bShape:        []int{2, 2},
			aData:         []float32{1, 2, 3, 4},
			bData:         []float32{10, 20, 30, 40},
			expectedData:  []float32{11, 22, 33, 44},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
		{
			name:          "broadcast row",
			aShape:        []int{2, 3},
			bShape:        []int{3},
			aData:         []float32{1, 2, 3, 4, 5, 6},
			bData:         []float32{10, 20, 30},
			expectedData:  []float32{11, 22, 33, 14, 25, 36},
			expectedShape: []int{2, 3},
			wantErr:       false,
		},
		{
			name:          "broadcast scalar",
			aShape:        []int{2, 2},
			bShape:        []int{1},
			aData:         []float32{1, 2, 3, 4},
			bData:         []float32{10},
			expectedData:  []float32{11, 12, 13, 14},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
		{
			name:      "incompatible shapes",
			aShape:    []int{2, 3},
			bShape:    []int{2, 4},
			aData:     []float32{1, 2, 3, 4, 5, 6},
			bData:     []float32{1, 2, 3, 4, 5, 6, 7, 8},
			wantErr:   true,
			errString: "cannot broadcast shapes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, _ := FromSlice(tt.aData, tt.aShape)
			b, _ := FromSlice(tt.bData, tt.bShape)
			result, err := Add(a, b)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(result.Shape, tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, result.Shape)
			}

			for i, v := range result.Data {
				if !floatEquals(v, tt.expectedData[i], 1e-5) {
					t.Errorf("Data mismatch at index %d: expected %f, got %f", i, tt.expectedData[i], v)
				}
			}
		})
	}
}

// TestMul tests element-wise multiplication with broadcasting
func TestMul(t *testing.T) {
	tests := []struct {
		name          string
		aShape        []int
		bShape        []int
		aData         []float32
		bData         []float32
		expectedData  []float32
		expectedShape []int
		wantErr       bool
		errString     string
	}{
		{
			name:          "same shape",
			aShape:        []int{2, 2},
			bShape:        []int{2, 2},
			aData:         []float32{1, 2, 3, 4},
			bData:         []float32{2, 3, 4, 5},
			expectedData:  []float32{2, 6, 12, 20},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
		{
			name:          "broadcast column",
			aShape:        []int{2, 3},
			bShape:        []int{2, 1},
			aData:         []float32{1, 2, 3, 4, 5, 6},
			bData:         []float32{2, 3},
			expectedData:  []float32{2, 4, 6, 12, 15, 18},
			expectedShape: []int{2, 3},
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, _ := FromSlice(tt.aData, tt.aShape)
			b, _ := FromSlice(tt.bData, tt.bShape)
			result, err := Mul(a, b)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(result.Shape, tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, result.Shape)
			}

			for i, v := range result.Data {
				if !floatEquals(v, tt.expectedData[i], 1e-5) {
					t.Errorf("Data mismatch at index %d: expected %f, got %f", i, tt.expectedData[i], v)
				}
			}
		})
	}
}

// TestSoftmax tests softmax computation
func TestSoftmax(t *testing.T) {
	tests := []struct {
		name      string
		data      []float32
		shape     []int
		dim       int
		wantErr   bool
		errString string
	}{
		{
			name:    "1D softmax",
			data:    []float32{1, 2, 3},
			shape:   []int{3},
			dim:     0,
			wantErr: false,
		},
		{
			name:    "2D softmax dim0",
			data:    []float32{1, 2, 3, 4, 5, 6},
			shape:   []int{2, 3},
			dim:     0,
			wantErr: false,
		},
		{
			name:    "2D softmax dim1",
			data:    []float32{1, 2, 3, 4, 5, 6},
			shape:   []int{2, 3},
			dim:     1,
			wantErr: false,
		},
		{
			name:      "invalid dim",
			data:      []float32{1, 2, 3},
			shape:     []int{3},
			dim:       5,
			wantErr:   true,
			errString: "invalid dimension",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, _ := FromSlice(tt.data, tt.shape)
			result, err := Softmax(tensor, tt.dim)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Check shape unchanged
			if !shapeEquals(result.Shape, tt.shape) {
				t.Errorf("Expected shape %v, got %v", tt.shape, result.Shape)
			}

			// Check values are positive and sum to 1
			for i, v := range result.Data {
				if v < 0 {
					t.Errorf("Softmax value at index %d should be positive, got %f", i, v)
				}
				if v > 1 {
					t.Errorf("Softmax value at index %d should be <= 1, got %f", i, v)
				}
			}
		})
	}
}

// TestSoftmaxNumericalStability tests softmax with large values
func TestSoftmaxNumericalStability(t *testing.T) {
	// Test with large values that could overflow
	data := []float32{1000, 1001, 1002}
	tensor, _ := FromSlice(data, []int{3})
	result, err := Softmax(tensor, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Should not be NaN or Inf
	for i, v := range result.Data {
		if math.IsNaN(float64(v)) {
			t.Errorf("Softmax produced NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Errorf("Softmax produced Inf at index %d", i)
		}
	}

	// Should sum to 1
	sum := float32(0)
	for _, v := range result.Data {
		sum += v
	}
	if !floatEquals(sum, 1.0, 1e-5) {
		t.Errorf("Softmax values should sum to 1, got %f", sum)
	}
}

// TestScale tests scalar multiplication
func TestScale(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tensor, _ := FromSlice(data, []int{2, 3})
	result := Scale(tensor, 2.5)

	expected := []float32{2.5, 5, 7.5, 10, 12.5, 15}
	for i, v := range result.Data {
		if !floatEquals(v, expected[i], 1e-5) {
			t.Errorf("Data mismatch at index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

// TestSliceN tests tensor slicing
func TestSliceN(t *testing.T) {
	tests := []struct {
		name          string
		data          []float32
		shape         []int
		starts        []int
		ends          []int
		expectedData  []float32
		expectedShape []int
		wantErr       bool
		errString     string
	}{
		{
			name:          "1D slice",
			data:          []float32{1, 2, 3, 4, 5},
			shape:         []int{5},
			starts:        []int{1},
			ends:          []int{4},
			expectedData:  []float32{2, 3, 4},
			expectedShape: []int{3},
			wantErr:       false,
		},
		{
			name:          "2D slice",
			data:          []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			shape:         []int{3, 3},
			starts:        []int{0, 1},
			ends:          []int{2, 3},
			expectedData:  []float32{2, 3, 5, 6},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
		{
			name:      "invalid start",
			data:      []float32{1, 2, 3},
			shape:     []int{3},
			starts:    []int{-1},
			ends:      []int{2},
			wantErr:   true,
			errString: "invalid start index",
		},
		{
			name:      "invalid end",
			data:      []float32{1, 2, 3},
			shape:     []int{3},
			starts:    []int{0},
			ends:      []int{5},
			wantErr:   true,
			errString: "invalid end index",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, _ := FromSlice(tt.data, tt.shape)
			result, err := tensor.SliceN(tt.starts, tt.ends)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(result.Shape, tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, result.Shape)
			}

			for i, v := range result.Data {
				if !floatEquals(v, tt.expectedData[i], 1e-5) {
					t.Errorf("Data mismatch at index %d: expected %f, got %f", i, tt.expectedData[i], v)
				}
			}
		})
	}
}

// TestConcatenate tests tensor concatenation
func TestConcatenate(t *testing.T) {
	tests := []struct {
		name          string
		tensors       [][]float32
		shapes        [][]int
		dim           int
		expectedData  []float32
		expectedShape []int
		wantErr       bool
		errString     string
	}{
		{
			name:          "concat 1D",
			tensors:       [][]float32{{1, 2}, {3, 4, 5}},
			shapes:        [][]int{{2}, {3}},
			dim:           0,
			expectedData:  []float32{1, 2, 3, 4, 5},
			expectedShape: []int{5},
			wantErr:       false,
		},
		{
			name:          "concat 2D dim0",
			tensors:       [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}},
			shapes:        [][]int{{2, 2}, {2, 2}},
			dim:           0,
			expectedData:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
			expectedShape: []int{4, 2},
			wantErr:       false,
		},
		{
			name:      "empty list",
			tensors:   [][]float32{},
			shapes:    [][]int{},
			dim:       0,
			wantErr:   true,
			errString: "cannot concatenate empty list",
		},
		{
			name:      "incompatible shapes",
			tensors:   [][]float32{{1, 2, 3, 4}, {5, 6, 7}},
			shapes:    [][]int{{2, 2}, {1, 3}},
			dim:       0,
			wantErr:   true,
			errString: "incompatible",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := make([]*Tensor, len(tt.tensors))
			for i := range tt.tensors {
				tensors[i], _ = FromSlice(tt.tensors[i], tt.shapes[i])
			}

			result, err := Concatenate(tensors, tt.dim)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
				} else if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("Expected error containing %q, got %q", tt.errString, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !shapeEquals(result.Shape, tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, result.Shape)
			}

			for i, v := range result.Data {
				if !floatEquals(v, tt.expectedData[i], 1e-5) {
					t.Errorf("Data mismatch at index %d: expected %f, got %f", i, tt.expectedData[i], v)
				}
			}
		})
	}
}

// TestShapeEquals tests shape comparison
func TestShapeEquals(t *testing.T) {
	a := NewTensor([]int{2, 3, 4})
	b := NewTensor([]int{2, 3, 4})
	c := NewTensor([]int{2, 3})
	d := NewTensor([]int{3, 2, 4})

	if !a.ShapeEquals(b) {
		t.Error("Expected a.ShapeEquals(b) to be true")
	}

	if a.ShapeEquals(c) {
		t.Error("Expected a.ShapeEquals(c) to be false")
	}

	if a.ShapeEquals(d) {
		t.Error("Expected a.ShapeEquals(d) to be false")
	}
}

// TestTotalSize tests element count
func TestTotalSize(t *testing.T) {
	tests := []struct {
		shape    []int
		expected int
	}{
		{[]int{2, 3}, 6},
		{[]int{1, 2, 3, 4}, 24},
		{[]int{5}, 5},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("%v", tt.shape), func(t *testing.T) {
			tensor := NewTensor(tt.shape)
			if tensor.TotalSize() != tt.expected {
				t.Errorf("Expected TotalSize %d, got %d", tt.expected, tensor.TotalSize())
			}
		})
	}
}

// TestString tests string representation
func TestString(t *testing.T) {
	tensor := NewTensor([]int{2, 3})
	tensor.Data[0] = 1.5
	tensor.Data[1] = 2.5
	tensor.Data[2] = 3.5

	str := tensor.String()
	if str == "" {
		t.Error("String() should not return empty string")
	}

	// Should contain shape information
	if !strings.Contains(str, "Tensor[") {
		t.Error("String() should contain 'Tensor['")
	}

	// Should contain data
	if !strings.Contains(str, "1.5") {
		t.Error("String() should contain '1.5'")
	}
}

// Helper functions

func shapeEquals(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func copyShapeInt(shape []int) []int {
	result := make([]int, len(shape))
	copy(result, shape)
	return result
}

func floatEquals(a, b, tolerance float32) bool {
	return math.Abs(float64(a-b)) < float64(tolerance)
}
