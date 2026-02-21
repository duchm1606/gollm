package model

import (
	"testing"

	"gollm/pkg/tensor"
)

func TestArgmax(t *testing.T) {
	// Create a simple 2D tensor: (batch=2, vocab_size=5)
	// Batch 0: [0.1, 0.3, 0.5, 0.2, 0.0] -> argmax = 2
	// Batch 1: [0.8, 0.1, 0.05, 0.02, 0.03] -> argmax = 0
	data := []float32{
		0.1, 0.3, 0.5, 0.2, 0.0,
		0.8, 0.1, 0.05, 0.02, 0.03,
	}
	logits, err := tensor.FromSlice(data, []int{2, 5})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	result, err := argmax(logits)
	if err != nil {
		t.Fatalf("argmax failed: %v", err)
	}

	// Check shape
	if len(result.Shape) != 2 || result.Shape[0] != 2 || result.Shape[1] != 1 {
		t.Errorf("Expected shape [2, 1], got %v", result.Shape)
	}

	// Check values
	if result.Get([]int{0, 0}) != 2 {
		t.Errorf("Expected batch 0 argmax = 2, got %f", result.Get([]int{0, 0}))
	}
	if result.Get([]int{1, 0}) != 0 {
		t.Errorf("Expected batch 1 argmax = 0, got %f", result.Get([]int{1, 0}))
	}
}

func TestExtractLastToken(t *testing.T) {
	// Create a 3D tensor: (batch=2, seq=3, vocab_size=4)
	data := []float32{
		// Batch 0
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		// Batch 1
		13, 14, 15, 16,
		17, 18, 19, 20,
		21, 22, 23, 24,
	}
	logits, err := tensor.FromSlice(data, []int{2, 3, 4})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	result, err := extractLastToken(logits)
	if err != nil {
		t.Fatalf("extractLastToken failed: %v", err)
	}

	// Check shape
	if len(result.Shape) != 2 || result.Shape[0] != 2 || result.Shape[1] != 4 {
		t.Errorf("Expected shape [2, 4], got %v", result.Shape)
	}

	// Check values - should extract the last token (index 2)
	expectedBatch0 := []float32{9, 10, 11, 12}
	expectedBatch1 := []float32{21, 22, 23, 24}

	for i := 0; i < 4; i++ {
		if result.Get([]int{0, i}) != expectedBatch0[i] {
			t.Errorf("Batch 0 index %d: expected %f, got %f", i, expectedBatch0[i], result.Get([]int{0, i}))
		}
		if result.Get([]int{1, i}) != expectedBatch1[i] {
			t.Errorf("Batch 1 index %d: expected %f, got %f", i, expectedBatch1[i], result.Get([]int{1, i}))
		}
	}
}

func TestGenerateTextSimple(t *testing.T) {
	// Create a small model for testing
	config := GPT2Config{
		VocabSize:     10,
		ContextLength: 8,
		EmbeddingDim:  16,
		NumHeads:      4,
		NumLayers:     2,
		HeadDim:       4,
		HiddenDim:     32,
		Dropout:       0.0, // No dropout for deterministic testing
		QKVBias:       false,
	}

	model := NewGPT2Model(config)

	// Set deterministic weights for testing (override random initialization)
	// For simplicity, set all weights to small positive values
	for i := range model.TokEmb.Data {
		model.TokEmb.Data[i] = float32(i%10) * 0.01
	}
	for i := range model.PosEmb.Data {
		model.PosEmb.Data[i] = float32(i%10) * 0.01
	}

	// Create initial input: batch=1, seq=2
	inputData := []float32{1, 2}
	idx, err := tensor.FromSlice(inputData, []int{1, 2})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Generate 3 tokens
	maxNewTokens := 3
	contextSize := 8

	result, err := GenerateTextSimple(model, idx, maxNewTokens, contextSize)
	if err != nil {
		t.Fatalf("GenerateTextSimple failed: %v", err)
	}

	// Check output shape: (batch=1, seq=2+3=5)
	if len(result.Shape) != 2 || result.Shape[0] != 1 || result.Shape[1] != 5 {
		t.Errorf("Expected shape [1, 5], got %v", result.Shape)
	}

	// Verify that we generated tokens (they should be non-negative integers within vocab range)
	for i := 0; i < result.Shape[1]; i++ {
		tokenID := result.Get([]int{0, i})
		if tokenID < 0 || tokenID >= float32(config.VocabSize) {
			t.Errorf("Token at position %d is out of range: %f", i, tokenID)
		}
	}

	t.Logf("Generated sequence: %v", result.Data)
}

func TestGenerateTextSimple_ContextCropping(t *testing.T) {
	// Test that context cropping works correctly
	config := GPT2Config{
		VocabSize:     10,
		ContextLength: 4, // Small context for testing
		EmbeddingDim:  8,
		NumHeads:      2,
		NumLayers:     1,
		HeadDim:       4,
		HiddenDim:     16,
		Dropout:       0.0,
		QKVBias:       false,
	}

	model := NewGPT2Model(config)

	// Create initial input that's longer than context length
	// (batch=1, seq=6) but context_length is only 4
	inputData := []float32{1, 2, 3, 4, 5, 6}
	idx, err := tensor.FromSlice(inputData, []int{1, 6})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Generate 2 tokens
	result, err := GenerateTextSimple(model, idx, 2, config.ContextLength)
	if err != nil {
		t.Fatalf("GenerateTextSimple failed: %v", err)
	}

	// Check output shape: (batch=1, seq=6+2=8)
	if len(result.Shape) != 2 || result.Shape[0] != 1 || result.Shape[1] != 8 {
		t.Errorf("Expected shape [1, 8], got %v", result.Shape)
	}
}

func TestGenerateTextSimple_InvalidInput(t *testing.T) {
	config := DefaultGPT2Config()
	model := NewGPT2Model(config)

	// Test with 1D input (should fail)
	inputData := []float32{1, 2, 3}
	idx, _ := tensor.FromSlice(inputData, []int{3})

	_, err := GenerateTextSimple(model, idx, 1, config.ContextLength)
	if err == nil {
		t.Error("Expected error for 1D input, got nil")
	}
}
