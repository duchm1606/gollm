package model

import (
	"testing"

	"gollm/pkg/tensor"
)

// TestNewKVCache_Preallocation verifies that KVCache is properly initialized
// with pre-allocated tensors of the correct shape.
func TestNewKVCache_Preallocation(t *testing.T) {
	batchSize := 2
	numKVGroups := 4
	maxLength := 100
	headDim := 64

	cache := NewKVCache(batchSize, numKVGroups, maxLength, headDim)

	// Verify cache is initialized correctly
	if cache == nil {
		t.Fatal("NewKVCache returned nil")
	}

	if cache.CurrentPos != 0 {
		t.Errorf("Expected CurrentPos=0, got %d", cache.CurrentPos)
	}

	if cache.MaxLength != maxLength {
		t.Errorf("Expected MaxLength=%d, got %d", maxLength, cache.MaxLength)
	}

	// Verify K tensor shape
	expectedShape := []int{batchSize, numKVGroups, maxLength, headDim}
	if len(cache.K.Shape) != len(expectedShape) {
		t.Errorf("K shape length mismatch: expected %v, got %v", expectedShape, cache.K.Shape)
	}
	for i, dim := range expectedShape {
		if cache.K.Shape[i] != dim {
			t.Errorf("K shape[%d] mismatch: expected %d, got %d", i, dim, cache.K.Shape[i])
		}
	}

	// Verify V tensor shape
	if len(cache.V.Shape) != len(expectedShape) {
		t.Errorf("V shape length mismatch: expected %v, got %v", expectedShape, cache.V.Shape)
	}
	for i, dim := range expectedShape {
		if cache.V.Shape[i] != dim {
			t.Errorf("V shape[%d] mismatch: expected %d, got %d", i, dim, cache.V.Shape[i])
		}
	}

	// Verify GetKV returns empty tensors initially
	k, v, seqLen := cache.GetKV()
	if seqLen != 0 {
		t.Errorf("Expected seqLen=0, got %d", seqLen)
	}
	if k.Shape[2] != 0 {
		t.Errorf("Expected empty K cache, got shape %v", k.Shape)
	}
	if v.Shape[2] != 0 {
		t.Errorf("Expected empty V cache, got shape %v", v.Shape)
	}
}

// TestUpdate_AppendSingleToken verifies that a single token can be appended
// to the cache correctly.
func TestUpdate_AppendSingleToken(t *testing.T) {
	cache := NewKVCache(1, 2, 10, 8)

	// Create new K and V for a single token
	newK := tensor.NewTensor([]int{1, 2, 1, 8})
	newV := tensor.NewTensor([]int{1, 2, 1, 8})

	// Fill with test values
	for g := 0; g < 2; g++ {
		for d := 0; d < 8; d++ {
			newK.Set([]int{0, g, 0, d}, float32(g*10+d))
			newV.Set([]int{0, g, 0, d}, float32(g*100+d))
		}
	}

	// Update cache
	cachedK, cachedV, err := cache.Update(newK, newV)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Verify position updated
	if cache.CurrentPos != 1 {
		t.Errorf("Expected CurrentPos=1, got %d", cache.CurrentPos)
	}

	// Verify returned tensors have correct shape
	expectedShape := []int{1, 2, 1, 8}
	for i, dim := range expectedShape {
		if cachedK.Shape[i] != dim {
			t.Errorf("cachedK shape[%d] mismatch: expected %d, got %d", i, dim, cachedK.Shape[i])
		}
		if cachedV.Shape[i] != dim {
			t.Errorf("cachedV shape[%d] mismatch: expected %d, got %d", i, dim, cachedV.Shape[i])
		}
	}

	// Verify values were copied correctly
	for g := 0; g < 2; g++ {
		for d := 0; d < 8; d++ {
			expectedK := float32(g*10 + d)
			expectedV := float32(g*100 + d)

			if cachedK.Get([]int{0, g, 0, d}) != expectedK {
				t.Errorf("K[%d,%d,0,%d] = %f, expected %f", 0, g, d, cachedK.Get([]int{0, g, 0, d}), expectedK)
			}
			if cachedV.Get([]int{0, g, 0, d}) != expectedV {
				t.Errorf("V[%d,%d,0,%d] = %f, expected %f", 0, g, d, cachedV.Get([]int{0, g, 0, d}), expectedV)
			}
		}
	}
}

// TestUpdate_AppendMultipleTokens verifies that multiple tokens can be
// appended to the cache in sequence.
func TestUpdate_AppendMultipleTokens(t *testing.T) {
	cache := NewKVCache(1, 2, 10, 8)

	// Append 3 tokens one at a time
	for tokenIdx := 0; tokenIdx < 3; tokenIdx++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})

		// Fill with unique values per token
		for g := 0; g < 2; g++ {
			for d := 0; d < 8; d++ {
				newK.Set([]int{0, g, 0, d}, float32(tokenIdx*100+g*10+d))
				newV.Set([]int{0, g, 0, d}, float32(tokenIdx*1000+g*100+d))
			}
		}

		cachedK, cachedV, err := cache.Update(newK, newV)
		if err != nil {
			t.Fatalf("Update at token %d failed: %v", tokenIdx, err)
		}

		// Verify position
		expectedPos := tokenIdx + 1
		if cache.CurrentPos != expectedPos {
			t.Errorf("After token %d, expected CurrentPos=%d, got %d", tokenIdx, expectedPos, cache.CurrentPos)
		}

		// Verify cached length
		if cachedK.Shape[2] != expectedPos {
			t.Errorf("After token %d, expected cachedK seq_len=%d, got %d", tokenIdx, expectedPos, cachedK.Shape[2])
		}

		// Verify all previous tokens are still in cache
		for prevIdx := 0; prevIdx <= tokenIdx; prevIdx++ {
			for g := 0; g < 2; g++ {
				for d := 0; d < 8; d++ {
					expectedK := float32(prevIdx*100 + g*10 + d)
					expectedV := float32(prevIdx*1000 + g*100 + d)

					if cachedK.Get([]int{0, g, prevIdx, d}) != expectedK {
						t.Errorf("Token %d: K[%d,%d,%d,%d] = %f, expected %f",
							tokenIdx, 0, g, prevIdx, d, cachedK.Get([]int{0, g, prevIdx, d}), expectedK)
					}
					if cachedV.Get([]int{0, g, prevIdx, d}) != expectedV {
						t.Errorf("Token %d: V[%d,%d,%d,%d] = %f, expected %f",
							tokenIdx, 0, g, prevIdx, d, cachedV.Get([]int{0, g, prevIdx, d}), expectedV)
					}
				}
			}
		}
	}
}

// TestUpdate_CacheOverflow verifies that attempting to exceed MaxLength
// returns an error.
func TestUpdate_CacheOverflow(t *testing.T) {
	cache := NewKVCache(1, 2, 5, 8)

	// Fill cache to capacity
	for i := 0; i < 5; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		_, _, err := cache.Update(newK, newV)
		if err != nil {
			t.Fatalf("Update at position %d failed unexpectedly: %v", i, err)
		}
	}

	// Try to add one more token - should fail
	newK := tensor.NewTensor([]int{1, 2, 1, 8})
	newV := tensor.NewTensor([]int{1, 2, 1, 8})
	_, _, err := cache.Update(newK, newV)

	if err == nil {
		t.Error("Expected error for cache overflow, got nil")
	}

	// Verify position hasn't changed
	if cache.CurrentPos != 5 {
		t.Errorf("Expected CurrentPos to remain 5, got %d", cache.CurrentPos)
	}
}

// TestUpdate_BatchAndMultipleTokens verifies updating with multiple tokens
// at once and with batch size > 1.
func TestUpdate_BatchAndMultipleTokens(t *testing.T) {
	batchSize := 2
	numKVGroups := 4
	maxLength := 20
	headDim := 32
	cache := NewKVCache(batchSize, numKVGroups, maxLength, headDim)

	// Append 3 tokens at once
	newTokens := 3
	newK := tensor.NewTensor([]int{batchSize, numKVGroups, newTokens, headDim})
	newV := tensor.NewTensor([]int{batchSize, numKVGroups, newTokens, headDim})

	// Fill with test values
	for b := 0; b < batchSize; b++ {
		for g := 0; g < numKVGroups; g++ {
			for tok := 0; tok < newTokens; tok++ {
				for d := 0; d < headDim; d++ {
					val := float32(b*10000 + g*1000 + tok*100 + d)
					newK.Set([]int{b, g, tok, d}, val)
					newV.Set([]int{b, g, tok, d}, val+50000)
				}
			}
		}
	}

	cachedK, cachedV, err := cache.Update(newK, newV)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Verify position
	if cache.CurrentPos != newTokens {
		t.Errorf("Expected CurrentPos=%d, got %d", newTokens, cache.CurrentPos)
	}

	// Verify shapes
	if cachedK.Shape[2] != newTokens {
		t.Errorf("Expected cachedK seq_len=%d, got %d", newTokens, cachedK.Shape[2])
	}

	// Verify all values were copied
	for b := 0; b < batchSize; b++ {
		for g := 0; g < numKVGroups; g++ {
			for tok := 0; tok < newTokens; tok++ {
				for d := 0; d < headDim; d++ {
					expectedK := float32(b*10000 + g*1000 + tok*100 + d)
					expectedV := expectedK + 50000

					if cachedK.Get([]int{b, g, tok, d}) != expectedK {
						t.Errorf("K[%d,%d,%d,%d] = %f, expected %f",
							b, g, tok, d, cachedK.Get([]int{b, g, tok, d}), expectedK)
					}
					if cachedV.Get([]int{b, g, tok, d}) != expectedV {
						t.Errorf("V[%d,%d,%d,%d] = %f, expected %f",
							b, g, tok, d, cachedV.Get([]int{b, g, tok, d}), expectedV)
					}
				}
			}
		}
	}

	// Append 2 more tokens
	newK2 := tensor.NewTensor([]int{batchSize, numKVGroups, 2, headDim})
	newV2 := tensor.NewTensor([]int{batchSize, numKVGroups, 2, headDim})

	cachedK2, _, err := cache.Update(newK2, newV2)
	if err != nil {
		t.Fatalf("Second update failed: %v", err)
	}

	// Verify total length
	expectedLen := 5
	if cache.CurrentPos != expectedLen {
		t.Errorf("Expected CurrentPos=%d, got %d", expectedLen, cache.CurrentPos)
	}
	if cachedK2.Shape[2] != expectedLen {
		t.Errorf("Expected cachedK seq_len=%d, got %d", expectedLen, cachedK2.Shape[2])
	}
}

// TestClear_Reset verifies that Clear properly resets the cache.
func TestClear_Reset(t *testing.T) {
	cache := NewKVCache(1, 2, 10, 8)

	// Add some data
	for i := 0; i < 3; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		cache.Update(newK, newV)
	}

	if cache.CurrentPos != 3 {
		t.Errorf("Before clear, expected CurrentPos=3, got %d", cache.CurrentPos)
	}

	// Clear the cache
	cache.Clear()

	// Verify reset
	if cache.CurrentPos != 0 {
		t.Errorf("After clear, expected CurrentPos=0, got %d", cache.CurrentPos)
	}

	// Verify GetKV returns empty tensors
	k, v, seqLen := cache.GetKV()
	if seqLen != 0 {
		t.Errorf("After clear, expected seqLen=0, got %d", seqLen)
	}
	if k.Shape[2] != 0 || v.Shape[2] != 0 {
		t.Errorf("After clear, expected empty tensors, got K=%v, V=%v", k.Shape, v.Shape)
	}
}

// TestGetPosition verifies GetPosition returns the correct offset.
func TestGetPosition(t *testing.T) {
	cache := NewKVCache(1, 2, 10, 8)

	if cache.GetPosition() != 0 {
		t.Errorf("Initial position should be 0, got %d", cache.GetPosition())
	}

	// Add tokens
	for i := 0; i < 5; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		cache.Update(newK, newV)

		expectedPos := i + 1
		if cache.GetPosition() != expectedPos {
			t.Errorf("After %d updates, expected position %d, got %d", i+1, expectedPos, cache.GetPosition())
		}
	}
}

// TestGetMaxLength verifies GetMaxLength returns the capacity.
func TestGetMaxLength(t *testing.T) {
	maxLength := 512
	cache := NewKVCache(1, 2, maxLength, 8)

	if cache.GetMaxLength() != maxLength {
		t.Errorf("Expected MaxLength=%d, got %d", maxLength, cache.GetMaxLength())
	}
}

// TestGetCacheSize verifies cache size calculation.
func TestGetCacheSize(t *testing.T) {
	batchSize := 2
	numKVGroups := 4
	maxLength := 100
	headDim := 64
	cache := NewKVCache(batchSize, numKVGroups, maxLength, headDim)

	// Calculate expected size: batch * groups * length * head_dim * 2 (K+V) * 4 bytes
	elements := batchSize * numKVGroups * maxLength * headDim
	expectedBytes := elements * 2 * 4

	if cache.GetCacheSize() != expectedBytes {
		t.Errorf("Expected cache size %d bytes, got %d", expectedBytes, cache.GetCacheSize())
	}
}

// TestUpdate_ShapeMismatch verifies proper error handling for shape mismatches.
func TestUpdate_ShapeMismatch(t *testing.T) {
	cache := NewKVCache(2, 4, 100, 64)

	// Test wrong batch size
	wrongBatchK := tensor.NewTensor([]int{1, 4, 1, 64})
	wrongBatchV := tensor.NewTensor([]int{1, 4, 1, 64})
	_, _, err := cache.Update(wrongBatchK, wrongBatchV)
	if err == nil {
		t.Error("Expected error for batch size mismatch")
	}

	// Test wrong num_kv_groups
	wrongGroupsK := tensor.NewTensor([]int{2, 2, 1, 64})
	wrongGroupsV := tensor.NewTensor([]int{2, 2, 1, 64})
	_, _, err = cache.Update(wrongGroupsK, wrongGroupsV)
	if err == nil {
		t.Error("Expected error for num_kv_groups mismatch")
	}

	// Test wrong head_dim
	wrongHeadK := tensor.NewTensor([]int{2, 4, 1, 32})
	wrongHeadV := tensor.NewTensor([]int{2, 4, 1, 32})
	_, _, err = cache.Update(wrongHeadK, wrongHeadV)
	if err == nil {
		t.Error("Expected error for head_dim mismatch")
	}

	// Test K and V shape mismatch
	k := tensor.NewTensor([]int{2, 4, 1, 64})
	v := tensor.NewTensor([]int{2, 4, 2, 64})
	_, _, err = cache.Update(k, v)
	if err == nil {
		t.Error("Expected error for K/V shape mismatch")
	}
}

// TestUpdate_InvalidDimensions verifies proper error handling for invalid tensor dimensions.
func TestUpdate_InvalidDimensions(t *testing.T) {
	cache := NewKVCache(2, 4, 100, 64)

	// Test 3D tensor (should be 4D)
	k3D := tensor.NewTensor([]int{2, 4, 64})
	v3D := tensor.NewTensor([]int{2, 4, 64})
	_, _, err := cache.Update(k3D, v3D)
	if err == nil {
		t.Error("Expected error for 3D tensor")
	}

	// Test 5D tensor (should be 4D)
	k5D := tensor.NewTensor([]int{2, 4, 1, 1, 64})
	v5D := tensor.NewTensor([]int{2, 4, 1, 1, 64})
	_, _, err = cache.Update(k5D, v5D)
	if err == nil {
		t.Error("Expected error for 5D tensor")
	}
}

// TestMultipleCaches verifies multiple independent caches work correctly.
func TestMultipleCaches(t *testing.T) {
	cache1 := NewKVCache(1, 2, 10, 8)
	cache2 := NewKVCache(1, 2, 10, 8)

	// Add different data to each
	for i := 0; i < 3; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		newK.Set([]int{0, 0, 0, 0}, float32(i+1))
		cache1.Update(newK, newV)
	}

	for i := 0; i < 5; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		newK.Set([]int{0, 0, 0, 0}, float32((i+1)*10))
		cache2.Update(newK, newV)
	}

	// Verify they're independent
	if cache1.CurrentPos != 3 {
		t.Errorf("cache1 position should be 3, got %d", cache1.CurrentPos)
	}
	if cache2.CurrentPos != 5 {
		t.Errorf("cache2 position should be 5, got %d", cache2.CurrentPos)
	}

	// Verify values
	k1, _, _ := cache1.GetKV()
	k2, _, _ := cache2.GetKV()

	if k1.Get([]int{0, 0, 0, 0}) != 1 {
		t.Errorf("cache1 first value should be 1, got %f", k1.Get([]int{0, 0, 0, 0}))
	}
	if k2.Get([]int{0, 0, 0, 0}) != 10 {
		t.Errorf("cache2 first value should be 10, got %f", k2.Get([]int{0, 0, 0, 0}))
	}
}

// TestCacheReuse verifies a cache can be cleared and reused.
func TestCacheReuse(t *testing.T) {
	cache := NewKVCache(1, 2, 10, 8)

	// First use
	for i := 0; i < 5; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		newK.Set([]int{0, 0, 0, 0}, float32(i+1))
		cache.Update(newK, newV)
	}

	k1, _, _ := cache.GetKV()
	firstValue := k1.Get([]int{0, 0, 0, 0})
	if firstValue != 1 {
		t.Errorf("First use first value should be 1, got %f", firstValue)
	}

	// Clear and reuse
	cache.Clear()

	for i := 0; i < 3; i++ {
		newK := tensor.NewTensor([]int{1, 2, 1, 8})
		newV := tensor.NewTensor([]int{1, 2, 1, 8})
		newK.Set([]int{0, 0, 0, 0}, float32((i+1)*100))
		cache.Update(newK, newV)
	}

	k2, _, _ := cache.GetKV()
	firstValue = k2.Get([]int{0, 0, 0, 0})
	if firstValue != 100 {
		t.Errorf("Second use first value should be 100, got %f", firstValue)
	}

	if cache.CurrentPos != 3 {
		t.Errorf("Position should be 3 after reuse, got %d", cache.CurrentPos)
	}
}

// TestCacheMemoryLayout verifies the internal memory layout is correct.
func TestCacheMemoryLayout(t *testing.T) {
	batchSize := 1
	numKVGroups := 2
	maxLength := 4
	headDim := 3
	cache := NewKVCache(batchSize, numKVGroups, maxLength, headDim)

	// Add tokens with predictable values
	// Token 0: values are 100 + group*10 + dim
	// Token 1: values are 200 + group*10 + dim
	for tokenIdx := 0; tokenIdx < 2; tokenIdx++ {
		newK := tensor.NewTensor([]int{batchSize, numKVGroups, 1, headDim})
		newV := tensor.NewTensor([]int{batchSize, numKVGroups, 1, headDim})

		for g := 0; g < numKVGroups; g++ {
			for d := 0; d < headDim; d++ {
				val := float32((tokenIdx+1)*100 + g*10 + d)
				newK.Set([]int{0, g, 0, d}, val)
				newV.Set([]int{0, g, 0, d}, val+1000)
			}
		}

		cache.Update(newK, newV)
	}

	// Verify internal cache layout by checking the full cache tensor
	// Token 0 should be at position 0
	for g := 0; g < numKVGroups; g++ {
		for d := 0; d < headDim; d++ {
			expectedK := float32(100 + g*10 + d)
			expectedV := expectedK + 1000

			if cache.K.Get([]int{0, g, 0, d}) != expectedK {
				t.Errorf("K cache[0,%d,0,%d] = %f, expected %f",
					g, d, cache.K.Get([]int{0, g, 0, d}), expectedK)
			}
			if cache.V.Get([]int{0, g, 0, d}) != expectedV {
				t.Errorf("V cache[0,%d,0,%d] = %f, expected %f",
					g, d, cache.V.Get([]int{0, g, 0, d}), expectedV)
			}
		}
	}

	// Token 1 should be at position 1
	for g := 0; g < numKVGroups; g++ {
		for d := 0; d < headDim; d++ {
			expectedK := float32(200 + g*10 + d)
			expectedV := expectedK + 1000

			if cache.K.Get([]int{0, g, 1, d}) != expectedK {
				t.Errorf("K cache[0,%d,1,%d] = %f, expected %f",
					g, d, cache.K.Get([]int{0, g, 1, d}), expectedK)
			}
			if cache.V.Get([]int{0, g, 1, d}) != expectedV {
				t.Errorf("V cache[0,%d,1,%d] = %f, expected %f",
					g, d, cache.V.Get([]int{0, g, 1, d}), expectedV)
			}
		}
	}
}
