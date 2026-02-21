package model

import (
	"fmt"
	"math"
	"math/rand"

	"gollm/pkg/model/attention"
	"gollm/pkg/tensor"
)

// GPT2Model implements the complete GPT-2 transformer model.
//
// Architecture:
//  1. Token embeddings: lookup table (vocab_size, emb_dim)
//  2. Positional embeddings: learned (context_length, emb_dim)
//  3. Transformer blocks: stack of NumLayers blocks
//  4. Final layer norm
//  5. Output projection: linear layer (emb_dim, vocab_size)
//
// Unlike LLaMA, GPT-2 uses:
//   - Learned positional embeddings (not RoPE)
//   - LayerNorm with scale AND shift (not RMSNorm)
//   - Standard Multi-Head Attention (not GQA)
//   - GELU activation (not SwiGLU)
type GPT2Model struct {
	Config    GPT2Config
	TokEmb    *tensor.Tensor // (vocab_size, emb_dim)
	PosEmb    *tensor.Tensor // (context_length, emb_dim) - LEARNED
	Blocks    []*attention.TransformerBlock
	FinalNorm *LayerNorm
	OutHead   *tensor.Tensor // (emb_dim, vocab_size)
	Training  bool           // If false, dropout is disabled
}

// NewGPT2Model creates a new GPT-2 model.
//
// Parameters:
//   - config: GPT2Config containing all architecture parameters
//
// Returns:
//   - Initialized GPT2Model with all weights allocated
func NewGPT2Model(config GPT2Config) *GPT2Model {
	// Validate config
	if err := config.Validate(); err != nil {
		panic(fmt.Sprintf("invalid config: %v", err))
	}

	model := &GPT2Model{
		Config:    config,
		TokEmb:    tensor.NewTensor([]int{config.VocabSize, config.EmbeddingDim}),
		PosEmb:    tensor.NewTensor([]int{config.ContextLength, config.EmbeddingDim}),
		Blocks:    make([]*attention.TransformerBlock, config.NumLayers),
		FinalNorm: NewLayerNorm(config.EmbeddingDim, 1e-5),
		OutHead:   tensor.NewTensor([]int{config.EmbeddingDim, config.VocabSize}),
		Training:  true, // Default to training mode
	}

	// Initialize transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		// Create attention with the new config format
		attnConfig := attention.MultiHeadAttentionConfig{
			NumHeads: config.NumHeads,
			DIn:      config.EmbeddingDim,
			DOut:     config.EmbeddingDim,
			Dropout:  config.Dropout,
			QKVBias:  config.QKVBias,
		}
		attn := attention.NewMultiHeadAttention(attnConfig)

		// Create feed-forward
		ff := NewFeedForward(config)

		// Create layer norms
		norm1 := NewLayerNorm(config.EmbeddingDim, 1e-5)
		norm2 := NewLayerNorm(config.EmbeddingDim, 1e-5)

		// Create block
		model.Blocks[i] = attention.NewTransformerBlock(attn, ff, norm1, norm2, config.Dropout)
	}

	// Initialize weights
	initializeWeights(model)

	return model
}

// SetTraining sets the training mode for the model.
// When training=false, dropout is disabled.
func (m *GPT2Model) SetTraining(training bool) {
	m.Training = training
}

// Forward computes the forward pass through the entire model.
//
// Input shape: (batch, seq) - token indices
// Output shape: (batch, seq, vocab_size) - logits
//
// Steps:
//  1. Get token embeddings: tok_embeds = TokEmb[input_idx]
//  2. Get positional embeddings: pos_embeds = PosEmb[0:seq_len]
//  3. Broadcast pos_embeds to (batch, seq, emb_dim)
//  4. x = tok_embeds + pos_embeds
//  5. x = Dropout(x, dropout_rate)
//  6. Create causal mask
//  7. For each block: x = block.Forward(x, mask)
//  8. x = FinalNorm(x)
//  9. logits = x @ OutHead^T
func (m *GPT2Model) Forward(inputIdx *tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputIdx.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D input (batch, seq), got %dD", len(inputIdx.Shape))
	}

	batchSize, seqLen := inputIdx.Shape[0], inputIdx.Shape[1]

	if seqLen > m.Config.ContextLength {
		return nil, fmt.Errorf("sequence length %d exceeds context length %d", seqLen, m.Config.ContextLength)
	}

	// Step 1: Get token embeddings via lookup
	tokEmbeds, err := lookupEmbeddings(m.TokEmb, inputIdx)
	if err != nil {
		return nil, fmt.Errorf("failed to lookup token embeddings: %w", err)
	}

	// Step 2: Get positional embeddings for positions 0 to seqLen-1
	posEmbedsSlice, err := m.PosEmb.SliceN(
		[]int{0, 0},
		[]int{seqLen, m.Config.EmbeddingDim},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to slice positional embeddings: %w", err)
	}

	// Step 3: Broadcast positional embeddings to (batch, seq, emb_dim)
	posEmbeds := broadcastPosEmbeddings(posEmbedsSlice, batchSize)

	// Step 4: Add token and positional embeddings
	x, err := tensor.Add(tokEmbeds, posEmbeds)
	if err != nil {
		return nil, fmt.Errorf("failed to add embeddings: %w", err)
	}

	// Step 5: Apply dropout (if training)
	if m.Config.Dropout > 0 {
		x = x.Dropout(m.Config.Dropout, m.Training)
	}

	// Step 6: Create causal mask
	mask := tensor.CreateCausalMask(seqLen)

	// Step 7: Pass through transformer blocks
	for i, block := range m.Blocks {
		x, err = block.Forward(x, mask, m.Training)
		if err != nil {
			return nil, fmt.Errorf("failed in transformer block %d: %w", i, err)
		}
	}

	// Step 8: Apply final layer norm
	x, err = m.FinalNorm.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("failed to apply final layer norm: %w", err)
	}

	// Step 9: Output projection to vocab size
	// x: (batch, seq, emb_dim) @ OutHead: (emb_dim, vocab_size) -> (batch, seq, vocab_size)
	logits, err := tensor.Matmul(x, m.OutHead)
	if err != nil {
		return nil, fmt.Errorf("failed to compute output logits: %w", err)
	}

	return logits, nil
}

// lookupEmbeddings performs embedding lookup for token indices.
//
// embTable: (vocab_size, emb_dim)
// indices: (batch, seq)
// output: (batch, seq, emb_dim)
func lookupEmbeddings(embTable *tensor.Tensor, indices *tensor.Tensor) (*tensor.Tensor, error) {
	batchSize := indices.Shape[0]
	seqLen := indices.Shape[1]
	vocabSize := embTable.Shape[0]
	embDim := embTable.Shape[1]

	output := tensor.NewTensor([]int{batchSize, seqLen, embDim})

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			tokenID := int(indices.Get([]int{b, s}))

			// Validate token ID
			if tokenID < 0 || tokenID >= vocabSize {
				return nil, fmt.Errorf("invalid token ID %d at position (%d, %d), vocab size is %d",
					tokenID, b, s, vocabSize)
			}

			// Copy embedding for this token
			srcOffset := tokenID * embDim
			dstOffset := (b*seqLen + s) * embDim
			copy(output.Data[dstOffset:dstOffset+embDim], embTable.Data[srcOffset:srcOffset+embDim])
		}
	}

	return output, nil
}

// broadcastPosEmbeddings broadcasts positional embeddings to batch dimension.
//
// posEmb: (seq, emb_dim) or (context_length, emb_dim)
// batchSize: target batch size
// output: (batch, seq, emb_dim)
func broadcastPosEmbeddings(posEmb *tensor.Tensor, batchSize int) *tensor.Tensor {
	seqLen := posEmb.Shape[0]
	embDim := posEmb.Shape[1]

	output := tensor.NewTensor([]int{batchSize, seqLen, embDim})

	// Copy the same positional embeddings for each batch
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			srcOffset := s * embDim
			dstOffset := (b*seqLen + s) * embDim
			copy(output.Data[dstOffset:dstOffset+embDim], posEmb.Data[srcOffset:srcOffset+embDim])
		}
	}

	return output
}

// initializeWeights initializes model weights using Xavier/Glorot initialization.
//
// Following GPT-2 initialization:
//   - Token embeddings: small normal distribution ~ N(0, 0.02)
//   - Positional embeddings: small normal distribution ~ N(0, 0.02)
//   - Linear layer weights: Xavier uniform
//   - LayerNorm scale: ones, shift: zeros (already done in NewLayerNorm)
func initializeWeights(model *GPT2Model) {
	// Initialize token embeddings with small normal distribution
	normalInit(model.TokEmb, 0.02)

	// Initialize positional embeddings with small normal distribution
	normalInit(model.PosEmb, 0.02)

	// Initialize output head with Xavier uniform
	xavierUniformInit(model.OutHead)

	// Initialize attention weights in each block
	for _, block := range model.Blocks {
		attn := block.Attn
		xavierUniformInit(attn.WQuery)
		xavierUniformInit(attn.WKey)
		xavierUniformInit(attn.WValue)
		xavierUniformInit(attn.OutProj)
	}
}

// normalInit initializes a tensor with values from a normal distribution N(0, std^2).
func normalInit(t *tensor.Tensor, std float32) {
	for i := range t.Data {
		t.Data[i] = float32(rand.NormFloat64()) * std
	}
}

// xavierUniformInit initializes a tensor with Xavier/Glorot uniform initialization.
// This is commonly used for linear layer weights.
// The variance is scaled by 2 / (fan_in + fan_out).
func xavierUniformInit(t *tensor.Tensor) {
	if len(t.Shape) < 2 {
		// For 1D tensors, use a simple uniform initialization
		for i := range t.Data {
			t.Data[i] = float32(rand.Float64()*2 - 1)
		}
		return
	}

	// Calculate fan_in and fan_out for the last two dimensions
	fanIn := t.Shape[len(t.Shape)-2]
	fanOut := t.Shape[len(t.Shape)-1]

	// Xavier uniform: U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))

	for i := range t.Data {
		t.Data[i] = float32(rand.Float64()*2*limit - limit)
	}
}
