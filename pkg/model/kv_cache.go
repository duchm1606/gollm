// Package model provides KV caching for efficient autoregressive generation.
//
// During autoregressive generation, we generate one token at a time. Without caching,
// we recompute K and V for all previous tokens every step. With caching, we store K/V
// and only compute for the new token, significantly reducing computation.
//
// Memory Complexity:
//   - Without cache: O(n²) per token (recomputing all positions)
//   - With cache: O(n) per token (only new position)
//
// Where n is the sequence length.
//
// This implementation supports standard Multi-Head Attention (MHA) as used in GPT-2.
package model

import (
	"fmt"

	"gollm/pkg/tensor"
)

// KVCache stores key and value tensors for efficient autoregressive generation.
// Instead of recomputing K and V for all previous tokens at each step,
// we cache them and only compute for new tokens.
//
// Shapes (for standard MHA):
//   - K: (batch, num_heads, cached_len, head_dim)
//   - V: (batch, num_heads, cached_len, head_dim)
type KVCache struct {
	K          *tensor.Tensor // Shape: (batch, num_heads, max_length, head_dim)
	V          *tensor.Tensor // Shape: (batch, num_heads, max_length, head_dim)
	CurrentPos int            // Next position to write (0 = empty)
	MaxLength  int            // Maximum cache capacity
	batchSize  int            // Batch size
	numHeads   int            // Number of attention heads
	headDim    int            // Head dimension
}

// NewKVCache creates a new KV cache with the specified capacity.
//
// Parameters:
//   - batchSize: batch size
//   - numHeads: number of attention heads (for standard MHA)
//   - maxLength: maximum sequence length to cache
//   - headDim: dimension per attention head
//
// Returns:
//   - Initialized KVCache with pre-allocated tensors
func NewKVCache(batchSize, numHeads, maxLength, headDim int) *KVCache {
	cacheShape := []int{batchSize, numHeads, maxLength, headDim}

	return &KVCache{
		K:          tensor.NewTensor(cacheShape),
		V:          tensor.NewTensor(cacheShape),
		CurrentPos: 0,
		MaxLength:  maxLength,
		batchSize:  batchSize,
		numHeads:   numHeads,
		headDim:    headDim,
	}
}

// Update appends new K and V tensors to the cache.
//
// Parameters:
//   - newK: new key tensor, shape: (batch, num_heads, new_tokens, head_dim)
//   - newV: new value tensor, shape: (batch, num_heads, new_tokens, head_dim)
//
// Returns:
//   - Full cached K tensor, shape: (batch, num_heads, CurrentPos, head_dim)
//   - Full cached V tensor, shape: (batch, num_heads, CurrentPos, head_dim)
//   - Error if cache overflow or shape mismatch
func (c *KVCache) Update(newK, newV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	// Validate input shapes
	if len(newK.Shape) != 4 || len(newV.Shape) != 4 {
		return nil, nil, fmt.Errorf("expected 4D tensors, got K=%dD, V=%dD",
			len(newK.Shape), len(newV.Shape))
	}

	// Extract dimensions
	batchSize := newK.Shape[0]
	numHeads := newK.Shape[1]
	newTokens := newK.Shape[2]
	headDim := newK.Shape[3]

	// Validate consistency with cache
	if batchSize != c.batchSize {
		return nil, nil, fmt.Errorf("batch size mismatch: expected %d, got %d",
			c.batchSize, batchSize)
	}
	if numHeads != c.numHeads {
		return nil, nil, fmt.Errorf("num_heads mismatch: expected %d, got %d",
			c.numHeads, numHeads)
	}
	if headDim != c.headDim {
		return nil, nil, fmt.Errorf("head_dim mismatch: expected %d, got %d",
			c.headDim, headDim)
	}

	// Check for cache overflow
	if c.CurrentPos+newTokens > c.MaxLength {
		return nil, nil, fmt.Errorf("cache overflow: cannot add %d tokens at position %d (max %d)",
			newTokens, c.CurrentPos, c.MaxLength)
	}

	// Validate newV matches newK shape
	if newV.Shape[0] != batchSize || newV.Shape[1] != numHeads ||
		newV.Shape[2] != newTokens || newV.Shape[3] != headDim {
		return nil, nil, fmt.Errorf("newK and newV must have same shape, got K=%v, V=%v",
			newK.Shape, newV.Shape)
	}

	// Copy new K data into cache at CurrentPos
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for t := 0; t < newTokens; t++ {
				for d := 0; d < headDim; d++ {
					// Source index in newK
					srcIdx := []int{b, h, t, d}
					val := newK.Get(srcIdx)

					// Destination index in cache
					dstPos := c.CurrentPos + t
					dstIdx := []int{b, h, dstPos, d}
					c.K.Set(dstIdx, val)
				}
			}
		}
	}

	// Copy new V data into cache at CurrentPos
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for t := 0; t < newTokens; t++ {
				for d := 0; d < headDim; d++ {
					// Source index in newV
					srcIdx := []int{b, h, t, d}
					val := newV.Get(srcIdx)

					// Destination index in cache
					dstPos := c.CurrentPos + t
					dstIdx := []int{b, h, dstPos, d}
					c.V.Set(dstIdx, val)
				}
			}
		}
	}

	// Update current position
	c.CurrentPos += newTokens

	// Return views of the cached data up to CurrentPos
	cachedK := sliceAlongSeqDim(c.K, c.CurrentPos)
	cachedV := sliceAlongSeqDim(c.V, c.CurrentPos)

	return cachedK, cachedV, nil
}

// sliceAlongSeqDim creates a view of the tensor up to seqLen along dimension 2.
// The tensor has shape (batch, num_heads, max_length, head_dim).
// We want to return a view with shape (batch, num_heads, seqLen, head_dim).
func sliceAlongSeqDim(t *tensor.Tensor, seqLen int) *tensor.Tensor {
	batchSize := t.Shape[0]
	numHeads := t.Shape[1]
	maxLength := t.Shape[2]
	headDim := t.Shape[3]

	// Create new shape with the correct sequence length
	newShape := []int{batchSize, numHeads, seqLen, headDim}

	// Allocate new data slice for the compacted view
	newSize := batchSize * numHeads * seqLen * headDim
	newData := make([]float32, newSize)

	// Copy data: for each (batch, head), copy only the first seqLen positions
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			// Source offset in original tensor for this (batch, head) pair
			srcOffset := ((b*numHeads + h) * maxLength) * headDim
			// Destination offset in new tensor
			dstOffset := ((b*numHeads + h) * seqLen) * headDim
			// Copy seqLen * headDim elements
			copy(newData[dstOffset:dstOffset+seqLen*headDim], t.Data[srcOffset:srcOffset+seqLen*headDim])
		}
	}

	// Compute strides for the new shape
	strides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= newShape[i]
	}

	// Return a new tensor with its own copy of the data
	return &tensor.Tensor{
		Data:    newData,
		Shape:   newShape,
		Strides: strides,
	}
}

// GetKV returns the cached K and V tensors up to CurrentPos.
//
// Returns:
//   - K tensor, shape: (batch, num_heads, CurrentPos, head_dim)
//   - V tensor, shape: (batch, num_heads, CurrentPos, head_dim)
//   - seqLen: CurrentPos (length of cached sequence)
func (c *KVCache) GetKV() (k, v *tensor.Tensor, seqLen int) {
	if c.CurrentPos == 0 {
		// Return empty tensors with correct batch and head dimensions
		emptyShape := []int{c.batchSize, c.numHeads, 0, c.headDim}
		return tensor.NewTensor(emptyShape), tensor.NewTensor(emptyShape), 0
	}

	cachedK := sliceAlongSeqDim(c.K, c.CurrentPos)
	cachedV := sliceAlongSeqDim(c.V, c.CurrentPos)

	return cachedK, cachedV, c.CurrentPos
}

// Clear resets the cache to empty state.
// Optionally zeros out the underlying tensors to prevent data leakage.
func (c *KVCache) Clear() {
	c.CurrentPos = 0
	// Zero out the cache to prevent data leakage
	for i := range c.K.Data {
		c.K.Data[i] = 0
	}
	for i := range c.V.Data {
		c.V.Data[i] = 0
	}
}

// GetPosition returns the current cache position (for positional embeddings offset).
// This is the number of tokens currently cached.
func (c *KVCache) GetPosition() int {
	return c.CurrentPos
}

// GetMaxLength returns the maximum capacity of the cache.
func (c *KVCache) GetMaxLength() int {
	return c.MaxLength
}

// GetCacheSize returns the memory size of the cache in bytes.
// Useful for monitoring memory usage.
func (c *KVCache) GetCacheSize() int {
	// Each float32 is 4 bytes
	elements := c.batchSize * c.numHeads * c.MaxLength * c.headDim
	return elements * 4 * 2 // K and V
}
