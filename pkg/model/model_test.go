package model

import (
	"testing"

	"gollm/pkg/tensor"
)

// TestNewGPT2Model tests the creation of GPT2Model.
func TestNewGPT2Model(t *testing.T) {
	config := DefaultGPT2Config()

	model := NewGPT2Model(config)

	if model == nil {
		t.Fatal("NewGPT2Model returned nil")
	}

	// Check token embeddings shape
	expectedTokEmbShape := []int{config.VocabSize, config.EmbeddingDim}
	for i := range expectedTokEmbShape {
		if model.TokEmb.Shape[i] != expectedTokEmbShape[i] {
			t.Errorf("TokEmb shape[%d] = %d, expected %d", i, model.TokEmb.Shape[i], expectedTokEmbShape[i])
		}
	}

	// Check positional embeddings shape
	expectedPosEmbShape := []int{config.ContextLength, config.EmbeddingDim}
	for i := range expectedPosEmbShape {
		if model.PosEmb.Shape[i] != expectedPosEmbShape[i] {
			t.Errorf("PosEmb shape[%d] = %d, expected %d", i, model.PosEmb.Shape[i], expectedPosEmbShape[i])
		}
	}

	// Check number of transformer blocks
	if len(model.Blocks) != config.NumLayers {
		t.Errorf("Expected %d blocks, got %d", config.NumLayers, len(model.Blocks))
	}

	// Check output head shape
	expectedOutHeadShape := []int{config.EmbeddingDim, config.VocabSize}
	for i := range expectedOutHeadShape {
		if model.OutHead.Shape[i] != expectedOutHeadShape[i] {
			t.Errorf("OutHead shape[%d] = %d, expected %d", i, model.OutHead.Shape[i], expectedOutHeadShape[i])
		}
	}
}

// TestGPT2Model_ForwardShape tests the forward pass output shape.
func TestGPT2Model_ForwardShape(t *testing.T) {
	config := GPT2Config{
		VocabSize:     100,
		ContextLength: 32,
		EmbeddingDim:  64,
		NumHeads:      4,
		NumLayers:     2,
		HeadDim:       16,
		HiddenDim:     256,
		Dropout:       0.0,
		QKVBias:       false,
	}

	model := NewGPT2Model(config)

	// Create input: (batch=2, seq=10)
	batchSize, seqLen := 2, 10
	input := tensor.NewTensor([]int{batchSize, seqLen})
	// Fill with token IDs (must be < vocabSize)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			input.Set([]int{b, s}, float32((b*seqLen+s)%config.VocabSize))
		}
	}

	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Check output shape: (batch, seq, vocab_size)
	expectedShape := []int{batchSize, seqLen, config.VocabSize}
	if len(output.Shape) != len(expectedShape) {
		t.Fatalf("Expected shape %v, got %v", expectedShape, output.Shape)
	}
	for i := range expectedShape {
		if output.Shape[i] != expectedShape[i] {
			t.Errorf("Output shape[%d] = %d, expected %d", i, output.Shape[i], expectedShape[i])
		}
	}
}

// TestGPT2Model_ForwardInvalidToken tests error handling for invalid token IDs.
func TestGPT2Model_ForwardInvalidToken(t *testing.T) {
	config := GPT2Config{
		VocabSize:     100,
		ContextLength: 32,
		EmbeddingDim:  64,
		NumHeads:      4,
		NumLayers:     1,
		HeadDim:       16,
		HiddenDim:     256,
		Dropout:       0.0,
		QKVBias:       false,
	}

	model := NewGPT2Model(config)

	// Create input with invalid token ID
	input := tensor.NewTensor([]int{1, 5})
	input.Set([]int{0, 0}, 150) // > vocabSize

	_, err := model.Forward(input)
	if err == nil {
		t.Error("Expected error for invalid token ID")
	}
}

// TestGPT2Model_ForwardTooLongSequence tests error handling for too long sequences.
func TestGPT2Model_ForwardTooLongSequence(t *testing.T) {
	config := GPT2Config{
		VocabSize:     100,
		ContextLength: 32,
		EmbeddingDim:  64,
		NumHeads:      4,
		NumLayers:     1,
		HeadDim:       16,
		HiddenDim:     256,
		Dropout:       0.0,
		QKVBias:       false,
	}

	model := NewGPT2Model(config)

	// Create input with sequence longer than context length
	input := tensor.NewTensor([]int{1, 50}) // 50 > ContextLength=32

	_, err := model.Forward(input)
	if err == nil {
		t.Error("Expected error for sequence longer than context length")
	}
}

// TestGPT2Model_DefaultConfig validates the default GPT-2 124M config.
func TestGPT2Model_DefaultConfig(t *testing.T) {
	config := DefaultGPT2Config()

	// Verify GPT-2 124M specifications
	if config.VocabSize != 50257 {
		t.Errorf("VocabSize = %d, expected 50257", config.VocabSize)
	}
	if config.ContextLength != 1024 {
		t.Errorf("ContextLength = %d, expected 1024", config.ContextLength)
	}
	if config.EmbeddingDim != 768 {
		t.Errorf("EmbeddingDim = %d, expected 768", config.EmbeddingDim)
	}
	if config.NumHeads != 12 {
		t.Errorf("NumHeads = %d, expected 12", config.NumHeads)
	}
	if config.NumLayers != 12 {
		t.Errorf("NumLayers = %d, expected 12", config.NumLayers)
	}
	if config.HiddenDim != 3072 {
		t.Errorf("HiddenDim = %d, expected 3072", config.HiddenDim)
	}
	if config.HeadDim != 64 {
		t.Errorf("HeadDim = %d, expected 64", config.HeadDim)
	}
	if config.Dropout != 0.1 {
		t.Errorf("Dropout = %v, expected 0.1", config.Dropout)
	}
	if config.QKVBias != false {
		t.Errorf("QKVBias = %v, expected false", config.QKVBias)
	}

	// Validate the config
	err := config.Validate()
	if err != nil {
		t.Errorf("Default config failed validation: %v", err)
	}
}

// TestGPT2Model_ConfigValidation tests config validation.
func TestGPT2Model_ConfigValidation(t *testing.T) {
	testCases := []struct {
		name    string
		config  GPT2Config
		wantErr bool
	}{
		{
			name: "valid_config",
			config: GPT2Config{
				VocabSize:     100,
				ContextLength: 64,
				EmbeddingDim:  64,
				NumHeads:      4,
				NumLayers:     2,
				HeadDim:       16,
				HiddenDim:     256,
				Dropout:       0.1,
				QKVBias:       false,
			},
			wantErr: false,
		},
		{
			name: "invalid_heads_division",
			config: GPT2Config{
				VocabSize:     100,
				ContextLength: 64,
				EmbeddingDim:  63, // Not divisible by 4
				NumHeads:      4,
				NumLayers:     2,
				HeadDim:       15,
				HiddenDim:     256,
				Dropout:       0.1,
				QKVBias:       false,
			},
			wantErr: true,
		},
		{
			name: "zero_vocab_size",
			config: GPT2Config{
				VocabSize:     0,
				ContextLength: 64,
				EmbeddingDim:  64,
				NumHeads:      4,
				NumLayers:     2,
				HeadDim:       16,
				HiddenDim:     256,
				Dropout:       0.1,
				QKVBias:       false,
			},
			wantErr: true,
		},
		{
			name: "zero_context_length",
			config: GPT2Config{
				VocabSize:     100,
				ContextLength: 0,
				EmbeddingDim:  64,
				NumHeads:      4,
				NumLayers:     2,
				HeadDim:       16,
				HiddenDim:     256,
				Dropout:       0.1,
				QKVBias:       false,
			},
			wantErr: true,
		},
		{
			name: "zero_num_layers",
			config: GPT2Config{
				VocabSize:     100,
				ContextLength: 64,
				EmbeddingDim:  64,
				NumHeads:      4,
				NumLayers:     0,
				HeadDim:       16,
				HiddenDim:     256,
				Dropout:       0.1,
				QKVBias:       false,
			},
			wantErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.config.Validate()
			if tc.wantErr && err == nil {
				t.Errorf("Expected error for %s, got none", tc.name)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.name, err)
			}
		})
	}
}

// TestGPT2Model_HeadDimension tests the HeadDimension method.
func TestGPT2Model_HeadDimension(t *testing.T) {
	config := GPT2Config{
		EmbeddingDim: 768,
		NumHeads:     12,
	}

	headDim := config.HeadDimension()
	expected := 64

	if headDim != expected {
		t.Errorf("HeadDimension() = %d, expected %d", headDim, expected)
	}
}

// BenchmarkGPT2Model_Forward benchmarks the forward pass.
func BenchmarkGPT2Model_Forward(b *testing.B) {
	config := GPT2Config{
		VocabSize:     1000,
		ContextLength: 128,
		EmbeddingDim:  256,
		NumHeads:      8,
		NumLayers:     2,
		HeadDim:       32,
		HiddenDim:     1024,
		Dropout:       0.0,
		QKVBias:       false,
	}

	model := NewGPT2Model(config)
	input := tensor.NewTensor([]int{1, 32}) // batch=1, seq=32

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := model.Forward(input)
		if err != nil {
			b.Fatal(err)
		}
	}
}
