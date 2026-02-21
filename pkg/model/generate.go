package model

import (
	"fmt"
	"math"

	"gollm/pkg/tensor"
)

// GenerateTextSimple generates text using greedy decoding.
//
// This implements the generate_text_simple function from Chapter 4 of LLMs from Scratch.
// It uses greedy decoding (argmax) to select the next token at each step.
//
// Parameters:
//   - model: GPT2Model instance
//   - idx: initial token indices, shape (batch, seq)
//   - maxNewTokens: number of tokens to generate
//   - contextSize: maximum context window size
//
// Returns:
//   - Generated token indices, shape (batch, seq + maxNewTokens)
func GenerateTextSimple(model *GPT2Model, idx *tensor.Tensor, maxNewTokens, contextSize int) (*tensor.Tensor, error) {
	if len(idx.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D input (batch, seq), got %dD with shape %v", len(idx.Shape), idx.Shape)
	}

	batchSize := idx.Shape[0]

	// Ensure we're in inference mode
	wasTraining := model.Training
	model.SetTraining(false)
	defer model.SetTraining(wasTraining)

	// Loop for maxNewTokens iterations
	for i := 0; i < maxNewTokens; i++ {
		// Step 1: Crop current context if it exceeds the supported context size
		// Take the last contextSize tokens: idx[:, -contextSize:]
		var idxCond *tensor.Tensor
		if idx.Shape[1] > contextSize {
			var err error
			idxCond, err = idx.SliceN(
				[]int{0, idx.Shape[1] - contextSize},
				[]int{batchSize, idx.Shape[1]},
			)
			if err != nil {
				return nil, fmt.Errorf("failed to crop context at step %d: %w", i, err)
			}
		} else {
			idxCond = idx
		}

		// Step 2: Get predictions from model
		// logits shape: (batch, seq, vocab_size)
		logits, err := model.Forward(idxCond)
		if err != nil {
			return nil, fmt.Errorf("model forward pass failed at step %d: %w", i, err)
		}

		// Step 3: Focus only on the last time step
		// logits shape: (batch, vocab_size)
		logitsLast, err := extractLastToken(logits)
		if err != nil {
			return nil, fmt.Errorf("failed to extract last token at step %d: %w", i, err)
		}

		// Step 4: Get the index of the vocab entry with the highest logits value
		// idxNext shape: (batch, 1)
		idxNext, err := argmax(logitsLast)
		if err != nil {
			return nil, fmt.Errorf("failed to compute argmax at step %d: %w", i, err)
		}

		// Step 5: Append sampled index to the running sequence
		// idx shape: (batch, seq + 1)
		idx, err = tensor.Concatenate([]*tensor.Tensor{idx, idxNext}, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to concatenate at step %d: %w", i, err)
		}
	}

	return idx, nil
}

// extractLastToken extracts the logits for the last token position.
//
// Input shape: (batch, seq, vocab_size)
// Output shape: (batch, vocab_size)
func extractLastToken(logits *tensor.Tensor) (*tensor.Tensor, error) {
	if len(logits.Shape) != 3 {
		return nil, fmt.Errorf("expected 3D input (batch, seq, vocab_size), got %dD", len(logits.Shape))
	}

	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	// Extract the last token: logits[:, -1, :]
	startIdx := []int{0, seqLen - 1, 0}
	endIdx := []int{batchSize, seqLen, vocabSize}

	result, err := logits.SliceN(startIdx, endIdx)
	if err != nil {
		return nil, err
	}

	// SliceN returns [batch, 1, vocab_size], we need to squeeze to [batch, vocab_size]
	return result.View([]int{batchSize, vocabSize})
}

// argmax returns the index of the maximum value along the last dimension.
//
// Input shape: (batch, vocab_size)
// Output shape: (batch, 1)
func argmax(logits *tensor.Tensor) (*tensor.Tensor, error) {
	if len(logits.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D input (batch, vocab_size), got %dD", len(logits.Shape))
	}

	batchSize := logits.Shape[0]
	vocabSize := logits.Shape[1]

	// Create output tensor with shape (batch, 1)
	result := tensor.NewTensor([]int{batchSize, 1})

	// Find argmax for each batch
	for b := 0; b < batchSize; b++ {
		maxIdx := 0
		maxVal := float32(math.Inf(-1))

		for v := 0; v < vocabSize; v++ {
			val := logits.Get([]int{b, v})
			if val > maxVal {
				maxVal = val
				maxIdx = v
			}
		}

		result.Set([]int{b, 0}, float32(maxIdx))
	}

	return result, nil
}
