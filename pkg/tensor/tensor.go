// Package tensor provides basic tensor operations for the LLM implementation.
// This is a simplified implementation focused on the needs of transformer models.
package tensor

import (
	"fmt"
	"math"
	"strings"
)

// Tensor represents a multi-dimensional array of float32 values.
// It stores data in a flat slice with shape information for indexing.
type Tensor struct {
	Data    []float32 // Flattened data storage
	Shape   []int     // Dimensions (e.g., [batch, heads, seq, dim])
	Strides []int     // Precomputed strides for indexing
}

// NewTensor creates a new tensor with the given shape, initialized to zeros.
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	// Precompute strides
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor{
		Data:    make([]float32, size),
		Shape:   copyShape(shape),
		Strides: strides,
	}
}

// FromSlice creates a tensor from existing data with the given shape.
// Returns an error if data size doesn't match the shape.
func FromSlice(data []float32, shape []int) (*Tensor, error) {
	expectedSize := 1
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid dimension %d in shape %v", dim, shape)
		}
		expectedSize *= dim
	}
	if len(data) != expectedSize {
		return nil, fmt.Errorf("data size %d does not match shape %v (expected %d elements)",
			len(data), shape, expectedSize)
	}

	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)

	// Precompute strides
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor{
		Data:    dataCopy,
		Shape:   copyShape(shape),
		Strides: strides,
	}, nil
}

// View returns a new tensor with a different shape but sharing the same underlying data.
// Returns an error if total size doesn't match.
func (t *Tensor) View(newShape []int) (*Tensor, error) {
	newSize := 1
	for _, dim := range newShape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid dimension %d in shape %v", dim, newShape)
		}
		newSize *= dim
	}

	if newSize != len(t.Data) {
		return nil, fmt.Errorf("cannot view tensor of size %d as shape %v (total size %d)",
			len(t.Data), newShape, newSize)
	}

	// Recompute strides for new shape
	newStrides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	return &Tensor{
		Data:    t.Data,
		Shape:   copyShape(newShape),
		Strides: newStrides,
	}, nil
}

// Transpose exchanges two dimensions of the tensor.
func (t *Tensor) Transpose(dim1, dim2 int) (*Tensor, error) {
	if dim1 < 0 || dim1 >= len(t.Shape) || dim2 < 0 || dim2 >= len(t.Shape) {
		return nil, fmt.Errorf("invalid transpose dimensions %d and %d for tensor with %d dimensions",
			dim1, dim2, len(t.Shape))
	}

	if dim1 == dim2 {
		// No change needed, return a copy
		return t.Clone(), nil
	}

	// Create new shape
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[dim1], newShape[dim2] = newShape[dim2], newShape[dim1]

	result := NewTensor(newShape)

	// Compute old strides
	oldStrides := t.Strides

	// Compute new strides
	newStrides := make([]int, len(t.Shape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	// Generic transpose using recursion
	// Iterate over source tensor indices (using original shape)
	srcIndices := make([]int, len(t.Shape))
	var transposeRec func(pos int)
	transposeRec = func(pos int) {
		if pos == len(t.Shape) {
			// Compute source index
			srcIdx := 0
			for i := 0; i < len(t.Shape); i++ {
				srcIdx += srcIndices[i] * oldStrides[i]
			}

			// Compute destination indices by swapping dim1 and dim2
			dstIndices := make([]int, len(t.Shape))
			copy(dstIndices, srcIndices)
			dstIndices[dim1], dstIndices[dim2] = dstIndices[dim2], dstIndices[dim1]

			// Compute destination index
			dstIdx := 0
			for i := 0; i < len(t.Shape); i++ {
				dstIdx += dstIndices[i] * newStrides[i]
			}

			result.Data[dstIdx] = t.Data[srcIdx]
			return
		}
		for i := 0; i < t.Shape[pos]; i++ {
			srcIndices[pos] = i
			transposeRec(pos + 1)
		}
	}
	transposeRec(0)

	return result, nil
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// TotalSize returns the total number of elements (alias for Size).
func (t *Tensor) TotalSize() int {
	return t.Size()
}

// NumDims returns the number of dimensions (rank) of the tensor.
func (t *Tensor) NumDims() int {
	return len(t.Shape)
}

// FlatIndex converts multi-dimensional indices to a flat index.
func (t *Tensor) FlatIndex(indices []int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("indices length %d does not match shape dimensions %d",
			len(indices), len(t.Shape)))
	}

	idx := 0
	for i := 0; i < len(t.Shape); i++ {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			panic(fmt.Sprintf("index %d out of bounds for dimension %d with size %d",
				indices[i], i, t.Shape[i]))
		}
		idx += indices[i] * t.Strides[i]
	}
	return idx
}

// Get retrieves a value at the specified indices.
func (t *Tensor) Get(indices []int) float32 {
	return t.Data[t.FlatIndex(indices)]
}

// Set sets a value at the specified indices.
func (t *Tensor) Set(indices []int, value float32) {
	t.Data[t.FlatIndex(indices)] = value
}

// GetFlat retrieves a value at a flat index.
func (t *Tensor) GetFlat(idx int) float32 {
	if idx < 0 || idx >= len(t.Data) {
		panic(fmt.Sprintf("flat index %d out of bounds [0, %d)", idx, len(t.Data)))
	}
	return t.Data[idx]
}

// SetFlat sets a value at a flat index.
func (t *Tensor) SetFlat(idx int, value float32) {
	if idx < 0 || idx >= len(t.Data) {
		panic(fmt.Sprintf("flat index %d out of bounds [0, %d)", idx, len(t.Data)))
	}
	t.Data[idx] = value
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	return NewTensorFromData(t.Data, t.Shape)
}

// NewTensorFromData creates a tensor from existing data with the given shape.
// It copies the data to ensure the tensor owns its memory.
func NewTensorFromData(data []float32, shape []int) *Tensor {
	expectedSize := 1
	for _, dim := range shape {
		expectedSize *= dim
	}
	if len(data) != expectedSize {
		panic(fmt.Sprintf("data size %d does not match shape %v (expected %d)",
			len(data), shape, expectedSize))
	}

	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)

	// Precompute strides
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor{
		Data:    dataCopy,
		Shape:   copyShape(shape),
		Strides: strides,
	}
}

// ShapeString returns a string representation of the shape.
func (t *Tensor) ShapeString() string {
	return fmt.Sprintf("%v", t.Shape)
}

// Equals checks if two tensors have the same shape and approximately equal values.
func (t *Tensor) Equals(other *Tensor, tolerance float32) bool {
	if len(t.Shape) != len(other.Shape) {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}
	for i := range t.Data {
		if math.Abs(float64(t.Data[i]-other.Data[i])) > float64(tolerance) {
			return false
		}
	}
	return true
}

// ShapeEquals checks if two tensors have the same shape.
func (t *Tensor) ShapeEquals(other *Tensor) bool {
	if len(t.Shape) != len(other.Shape) {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}
	return true
}

// Reshape returns a view with a different shape (same underlying data).
func (t *Tensor) Reshape(newShape []int) *Tensor {
	result, err := t.View(newShape)
	if err != nil {
		panic(err)
	}
	return result
}

// Slice returns a view into a portion of the tensor along the last dimension.
// This creates a new tensor sharing the underlying data (no copy).
func (t *Tensor) Slice(start, end int) (*Tensor, error) {
	if len(t.Shape) == 0 {
		return nil, fmt.Errorf("cannot slice a scalar tensor")
	}

	lastDim := len(t.Shape) - 1
	if start < 0 || end > t.Shape[lastDim] || start >= end {
		return nil, fmt.Errorf("invalid slice range [%d, %d) for dimension with size %d",
			start, end, t.Shape[lastDim])
	}

	// Calculate the size of one slice
	outerSize := 1
	for i := 0; i < lastDim; i++ {
		outerSize *= t.Shape[i]
	}
	sliceSize := end - start

	newShape := append([]int{}, t.Shape...)
	newShape[lastDim] = sliceSize

	// Calculate offset into data
	offset := start

	// Create new tensor sharing data
	newSize := outerSize * sliceSize

	// Compute new strides
	newStrides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	return &Tensor{
		Data:    t.Data[offset : offset+newSize],
		Shape:   newShape,
		Strides: newStrides,
	}, nil
}

// SliceN extracts a sub-tensor from the given ranges for all dimensions.
func (t *Tensor) SliceN(starts, ends []int) (*Tensor, error) {
	if len(starts) != len(t.Shape) || len(ends) != len(t.Shape) {
		return nil, fmt.Errorf("starts and ends must have same length as tensor dimensions (%d), got %d and %d",
			len(t.Shape), len(starts), len(ends))
	}

	newShape := make([]int, len(t.Shape))
	for i := 0; i < len(t.Shape); i++ {
		if starts[i] < 0 || starts[i] > t.Shape[i] {
			return nil, fmt.Errorf("invalid start index %d for dimension %d with size %d", starts[i], i, t.Shape[i])
		}
		if ends[i] < starts[i] || ends[i] > t.Shape[i] {
			return nil, fmt.Errorf("invalid end index %d for dimension %d (start=%d, size=%d)", ends[i], i, starts[i], t.Shape[i])
		}
		newShape[i] = ends[i] - starts[i]
	}

	result := NewTensor(newShape)

	// Copy data
	srcIndices := make([]int, len(t.Shape))
	dstIndices := make([]int, len(t.Shape))

	var copyData func(dim int)
	copyData = func(dim int) {
		if dim == len(t.Shape) {
			srcOffset := t.FlatIndex(srcIndices)
			dstOffset := result.FlatIndex(dstIndices)
			result.Data[dstOffset] = t.Data[srcOffset]
			return
		}

		for i := 0; i < newShape[dim]; i++ {
			srcIndices[dim] = starts[dim] + i
			dstIndices[dim] = i
			copyData(dim + 1)
		}
	}

	copyData(0)
	return result, nil
}

// FlatSlice returns a flat slice of the data from start to end index.
func (t *Tensor) FlatSlice(start, end int) []float32 {
	if start < 0 || end > len(t.Data) || start >= end {
		panic(fmt.Sprintf("invalid flat slice range [%d, %d) for data of size %d",
			start, end, len(t.Data)))
	}
	return t.Data[start:end]
}

// Matmul performs matrix multiplication on the last two dimensions.
// For tensors of shape (..., m, n) and (..., n, p), returns (..., m, p).
// Supports broadcasting: if one operand is 2D and the other is 3D, the 2D is broadcast.
func Matmul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		return nil, fmt.Errorf("matmul requires at least 2D tensors, got %dD and %dD",
			len(a.Shape), len(b.Shape))
	}

	// Get matrix dimensions
	k_a := a.Shape[len(a.Shape)-1]
	k_b := b.Shape[len(b.Shape)-2]

	if k_a != k_b {
		return nil, fmt.Errorf("incompatible shapes for matmul: %v and %v (inner dimensions %d and %d don't match)",
			a.Shape, b.Shape, k_a, k_b)
	}

	// Handle broadcasting: if one is 2D and other is 3D
	if len(a.Shape) == 2 && len(b.Shape) == 3 {
		return matmul2D3D(a, b)
	}
	if len(a.Shape) == 3 && len(b.Shape) == 2 {
		return matmul3D2D(a, b)
	}

	// Standard batched matmul
	return matmulBatched(a, b)
}

// matmul3D2D handles (batch, m, n) @ (n, p) -> (batch, m, p)
func matmul3D2D(a, b *Tensor) (*Tensor, error) {
	batch, m, n := a.Shape[0], a.Shape[1], a.Shape[2]
	n2, p := b.Shape[0], b.Shape[1]

	if n != n2 {
		return nil, fmt.Errorf("incompatible shapes: %v @ %v", a.Shape, b.Shape)
	}

	result := NewTensor([]int{batch, m, p})

	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for k := 0; k < p; k++ {
				sum := float32(0)
				for j := 0; j < n; j++ {
					aVal := a.Data[(bi*m+i)*n+j]
					bVal := b.Data[j*p+k]
					sum += aVal * bVal
				}
				result.Data[(bi*m+i)*p+k] = sum
			}
		}
	}

	return result, nil
}

// matmul2D3D handles (m, n) @ (batch, n, p) -> (batch, m, p)
func matmul2D3D(a, b *Tensor) (*Tensor, error) {
	m, n := a.Shape[0], a.Shape[1]
	batch, n2, p := b.Shape[0], b.Shape[1], b.Shape[2]

	if n != n2 {
		return nil, fmt.Errorf("incompatible shapes: %v @ %v", a.Shape, b.Shape)
	}

	result := NewTensor([]int{batch, m, p})

	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for k := 0; k < p; k++ {
				sum := float32(0)
				for j := 0; j < n; j++ {
					aVal := a.Data[i*n+j]
					bVal := b.Data[(bi*n2+j)*p+k]
					sum += aVal * bVal
				}
				result.Data[(bi*m+i)*p+k] = sum
			}
		}
	}

	return result, nil
}

// matmulBatched handles batched matrix multiplication.
func matmulBatched(a, b *Tensor) (*Tensor, error) {
	// Get the matrix dimensions
	m := a.Shape[len(a.Shape)-2]
	n := a.Shape[len(a.Shape)-1]
	p := b.Shape[len(b.Shape)-1]

	if b.Shape[len(b.Shape)-2] != n {
		return nil, fmt.Errorf("incompatible shapes for matmul: %v and %v", a.Shape, b.Shape)
	}

	// Get batch dimensions (all but last 2)
	batchDims := a.Shape[:len(a.Shape)-2]
	batchSize := 1
	for _, dim := range batchDims {
		batchSize *= dim
	}

	// Compute result shape
	resultShape := append([]int{}, batchDims...)
	resultShape = append(resultShape, m, p)
	result := NewTensor(resultShape)

	// Perform batched matrix multiplication
	for batch := 0; batch < batchSize; batch++ {
		aOffset := batch * m * n
		bOffset := batch * n * p
		rOffset := batch * m * p

		for i := 0; i < m; i++ {
			for k := 0; k < p; k++ {
				sum := float32(0)
				for j := 0; j < n; j++ {
					sum += a.Data[aOffset+i*n+j] * b.Data[bOffset+j*p+k]
				}
				result.Data[rOffset+i*p+k] = sum
			}
		}
	}

	return result, nil
}

// Scale multiplies all elements by a scalar.
func Scale(t *Tensor, scalar float32) *Tensor {
	result := NewTensor(t.Shape)
	for i := range t.Data {
		result.Data[i] = t.Data[i] * scalar
	}
	return result
}

// ScaleT multiplies all elements by a scalar (tensor method version).
func (t *Tensor) Scale(s float32) *Tensor {
	return Scale(t, s)
}

// Softmax applies softmax along the specified dimension.
func Softmax(t *Tensor, dim int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.Shape) {
		return nil, fmt.Errorf("invalid dimension %d for tensor with %d dimensions", dim, len(t.Shape))
	}

	result := NewTensor(t.Shape)

	// Number of slices along the softmax dimension
	sliceSize := t.Shape[dim]

	// Number of independent softmax operations
	numSlices := len(t.Data) / sliceSize

	for sliceIdx := 0; sliceIdx < numSlices; sliceIdx++ {
		// Compute starting offset for this slice
		offsets := make([]int, len(t.Shape))
		remaining := sliceIdx
		for i := len(t.Shape) - 1; i >= 0; i-- {
			if i == dim {
				offsets[i] = 0
			} else {
				dimSize := t.Shape[i]
				if i > dim {
					offsets[i] = remaining % dimSize
					remaining /= dimSize
				} else {
					offsets[i] = remaining % dimSize
					remaining /= dimSize
				}
			}
		}

		// Find max for numerical stability
		maxVal := float32(math.Inf(-1))
		for i := 0; i < sliceSize; i++ {
			offsets[dim] = i
			idx := t.FlatIndex(offsets)
			if t.Data[idx] > maxVal {
				maxVal = t.Data[idx]
			}
		}

		// Compute exp(x - max) and sum
		expSum := float32(0.0)
		expVals := make([]float32, sliceSize)
		for i := 0; i < sliceSize; i++ {
			offsets[dim] = i
			idx := t.FlatIndex(offsets)
			expVals[i] = float32(math.Exp(float64(t.Data[idx] - maxVal)))
			expSum += expVals[i]
		}

		// Normalize
		for i := 0; i < sliceSize; i++ {
			offsets[dim] = i
			dstIdx := result.FlatIndex(offsets)
			result.Data[dstIdx] = expVals[i] / expSum
		}
	}

	return result, nil
}

// SoftmaxLast applies softmax along the last dimension (convenience function).
func SoftmaxLast(t *Tensor) *Tensor {
	result, err := Softmax(t, len(t.Shape)-1)
	if err != nil {
		panic(err)
	}
	return result
}

// Add performs element-wise addition with broadcasting.
func Add(a, b *Tensor) (*Tensor, error) {
	return elementWiseOp(a, b, func(x, y float32) float32 { return x + y })
}

// Mul performs element-wise multiplication with broadcasting.
func Mul(a, b *Tensor) (*Tensor, error) {
	return elementWiseOp(a, b, func(x, y float32) float32 { return x * y })
}

// elementWiseOp performs an element-wise operation with broadcasting
func elementWiseOp(a, b *Tensor, op func(float32, float32) float32) (*Tensor, error) {
	outShape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, fmt.Errorf("cannot broadcast shapes %v and %v: %w", a.Shape, b.Shape, err)
	}

	result := NewTensor(outShape)

	// Iterate over all output positions
	indices := make([]int, len(outShape))
	var iterate func(dim int)
	iterate = func(dim int) {
		if dim == len(outShape) {
			// Get values from a and b (with broadcasting)
			aIdx := broadcastIndex(indices, outShape, a.Shape)
			bIdx := broadcastIndex(indices, outShape, b.Shape)

			aVal := float32(0)
			if aIdx >= 0 && aIdx < len(a.Data) {
				aVal = a.Data[aIdx]
			}

			bVal := float32(0)
			if bIdx >= 0 && bIdx < len(b.Data) {
				bVal = b.Data[bIdx]
			}

			outIdx := result.FlatIndex(indices)
			result.Data[outIdx] = op(aVal, bVal)
			return
		}

		for i := 0; i < outShape[dim]; i++ {
			indices[dim] = i
			iterate(dim + 1)
		}
	}

	iterate(0)
	return result, nil
}

// broadcastShapes computes the broadcasted shape of two shapes
func broadcastShapes(a, b []int) ([]int, error) {
	maxLen := len(a)
	if len(b) > maxLen {
		maxLen = len(b)
	}

	result := make([]int, maxLen)

	for i := 0; i < maxLen; i++ {
		dimA := 1
		if i < len(a) {
			dimA = a[len(a)-1-i]
		}
		dimB := 1
		if i < len(b) {
			dimB = b[len(b)-1-i]
		}

		if dimA != dimB && dimA != 1 && dimB != 1 {
			return nil, fmt.Errorf("incompatible dimensions %d and %d", dimA, dimB)
		}

		if dimA > dimB {
			result[maxLen-1-i] = dimA
		} else {
			result[maxLen-1-i] = dimB
		}
	}

	return result, nil
}

// broadcastIndex computes the index in a broadcasted tensor
func broadcastIndex(outIndices []int, outShape, inShape []int) int {
	if len(inShape) == 0 {
		return 0
	}

	// Pad inShape with leading 1s to match outShape length
	diff := len(outShape) - len(inShape)
	inIndices := make([]int, len(inShape))

	for i := 0; i < len(inShape); i++ {
		outDimIdx := i + diff
		if inShape[i] == outShape[outDimIdx] {
			inIndices[i] = outIndices[outDimIdx]
		} else if inShape[i] == 1 {
			inIndices[i] = 0
		} else {
			// Should not happen if broadcastShapes passed
			return -1
		}
	}

	// Compute flat index using strides
	inTensor := &Tensor{Shape: inShape}
	inTensor.Strides = make([]int, len(inShape))
	stride := 1
	for i := len(inShape) - 1; i >= 0; i-- {
		inTensor.Strides[i] = stride
		stride *= inShape[i]
	}

	return inTensor.FlatIndex(inIndices)
}

// Concatenate concatenates tensors along a dimension.
func Concatenate(tensors []*Tensor, dim int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("cannot concatenate empty list of tensors")
	}

	if dim < 0 || dim >= len(tensors[0].Shape) {
		return nil, fmt.Errorf("invalid dimension %d for tensor with %d dimensions", dim, len(tensors[0].Shape))
	}

	// Validate shapes and compute output shape
	outShape := copyShape(tensors[0].Shape)
	concatSize := tensors[0].Shape[dim]

	for i := 1; i < len(tensors); i++ {
		t := tensors[i]
		if len(t.Shape) != len(outShape) {
			return nil, fmt.Errorf("tensor %d has %d dimensions, expected %d", i, len(t.Shape), len(outShape))
		}
		for j := 0; j < len(outShape); j++ {
			if j == dim {
				concatSize += t.Shape[j]
			} else if t.Shape[j] != outShape[j] {
				return nil, fmt.Errorf("tensor %d has shape %v, incompatible with %v at dimension %d", i, t.Shape, outShape, j)
			}
		}
	}
	outShape[dim] = concatSize

	result := NewTensor(outShape)

	// Copy data from each tensor
	// For simplicity, this assumes contiguous memory along the concat dimension
	dstOffset := 0
	for _, t := range tensors {
		copy(result.Data[dstOffset:dstOffset+len(t.Data)], t.Data)
		dstOffset += len(t.Data)
	}

	return result, nil
}

// ApplyMask sets elements to -inf where mask is 0 (for causal masking).
func ApplyMask(t, mask *Tensor) *Tensor {
	result := NewTensor(t.Shape)
	copy(result.Data, t.Data)

	for i := range t.Data {
		if i < len(mask.Data) && mask.Data[i] == 0 {
			result.Data[i] = float32(math.Inf(-1))
		}
	}

	return result
}

// CreateCausalMask creates an upper triangular causal mask for attention.
// Shape: (seq_len, seq_len), with 1s in lower triangle and 0s in upper triangle.
func CreateCausalMask(seqLen int) *Tensor {
	mask := NewTensor([]int{seqLen, seqLen})
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				mask.Data[i*seqLen+j] = 1
			}
		}
	}
	return mask
}

// Expand repeats a tensor along a dimension.
// For attention: expand K/V from (batch, kv_heads, seq, dim) to (batch, num_heads, seq, dim).
func (t *Tensor) Expand(dim, times int) *Tensor {
	if dim < 0 || dim >= len(t.Shape) {
		panic(fmt.Sprintf("invalid dimension %d for tensor with %d dimensions", dim, len(t.Shape)))
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[dim] *= times

	result := NewTensor(newShape)

	// Optimized for 4D tensors expanding dim 1 (heads)
	if len(t.Shape) == 4 && dim == 1 {
		b, h, s, d := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
		for i := 0; i < b; i++ {
			for j := 0; j < h; j++ {
				for r := 0; r < times; r++ {
					for k := 0; k < s; k++ {
						for l := 0; l < d; l++ {
							srcIdx := ((i*h+j)*s+k)*d + l
							dstIdx := ((i*(h*times)+(j*times+r))*s+k)*d + l
							result.Data[dstIdx] = t.Data[srcIdx]
						}
					}
				}
			}
		}
		return result
	}

	// Generic implementation
	// Not implemented for other cases
	panic("Expand only implemented for 4D tensors expanding dim 1")
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	var sb strings.Builder
	sb.WriteString("Tensor[")
	for i, dim := range t.Shape {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("%d", dim))
	}
	sb.WriteString("]")

	// Show data
	sb.WriteString(": ")
	sb.WriteString(t.formatData(t.Shape, t.Data, 0, 0))

	return sb.String()
}

// formatData recursively formats tensor data
func (t *Tensor) formatData(shape []int, data []float32, dim, offset int) string {
	if len(shape) == 0 {
		return fmt.Sprintf("%g", data[offset])
	}

	if len(shape) == 1 {
		var sb strings.Builder
		sb.WriteString("[")
		for i := 0; i < shape[0] && i < 6; i++ {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%g", data[offset+i]))
		}
		if shape[0] > 6 {
			sb.WriteString(", ...")
		}
		sb.WriteString("]")
		return sb.String()
	}

	var sb strings.Builder
	sb.WriteString("[")
	subSize := 1
	for i := 1; i < len(shape); i++ {
		subSize *= shape[i]
	}

	for i := 0; i < shape[0] && i < 3; i++ {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(t.formatData(shape[1:], data, dim+1, offset+i*subSize))
	}
	if shape[0] > 3 {
		sb.WriteString(", ...")
	}
	sb.WriteString("]")
	return sb.String()
}

// copyShape creates a copy of a shape slice
func copyShape(shape []int) []int {
	result := make([]int, len(shape))
	copy(result, shape)
	return result
}
