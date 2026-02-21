package attention

import (
	"fmt"

	"gollm/pkg/tensor"
)

// FeedForward is an interface for feed-forward layers
type FeedForward interface {
	Forward(x *tensor.Tensor) (*tensor.Tensor, error)
}

// LayerNorm is an interface for layer normalization
type LayerNorm interface {
	Forward(x *tensor.Tensor) (*tensor.Tensor, error)
}

// TransformerBlock implements a single transformer block for GPT-2.
//
// Architecture (per block):
//  1. shortcut = x
//  2. x = Norm1(x)           # Pre-attention layer norm
//  3. x = Attn(x, mask)      # Multi-head attention
//  4. x = Dropout(x)         # Apply dropout
//  5. x = x + shortcut       # Residual connection
//  6. shortcut = x
//  7. x = Norm2(x)           # Pre-FFN layer norm
//  8. x = FF(x)              # Feed-forward network
//  9. x = Dropout(x)         # Apply dropout
//  10. x = x + shortcut      # Residual connection
//
// GPT-2 uses LayerNorm before attention and FFN (pre-norm).
type TransformerBlock struct {
	Attn    *MultiHeadAttention
	FF      FeedForward
	Norm1   LayerNorm // Pre-attention
	Norm2   LayerNorm // Pre-FFN
	Dropout float32
}

// NewTransformerBlock creates a new transformer block.
//
// Parameters:
//   - attn: MultiHeadAttention instance
//   - ff: FeedForward instance
//   - norm1: LayerNorm for pre-attention
//   - norm2: LayerNorm for pre-FFN
//   - dropout: dropout rate
//
// Returns:
//   - Initialized TransformerBlock
func NewTransformerBlock(attn *MultiHeadAttention, ff FeedForward, norm1, norm2 LayerNorm, dropout float32) *TransformerBlock {
	return &TransformerBlock{
		Attn:    attn,
		FF:      ff,
		Norm1:   norm1,
		Norm2:   norm2,
		Dropout: dropout,
	}
}

// Forward computes one transformer block.
//
// Input shapes:
//   - x: (batch, seq, emb_dim)
//   - mask: optional causal mask, shape (seq, seq) or nil
//   - training: if true, apply dropout; if false, skip dropout
//
// Output shape: (batch, seq, emb_dim)
func (b *TransformerBlock) Forward(x, mask *tensor.Tensor, training bool) (*tensor.Tensor, error) {
	// Step 1: Attention block with residual
	// shortcut = x
	// x = Norm1(x)
	// x = Attn(x, mask)
	// x = Dropout(x)
	// x = x + shortcut

	shortcut := x

	// LayerNorm before attention
	normed, err := b.Norm1.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("failed to apply Norm1: %w", err)
	}

	// Multi-head attention
	attnOut, err := b.Attn.Forward(normed, mask, training)
	if err != nil {
		return nil, fmt.Errorf("failed to compute attention: %w", err)
	}

	// Apply dropout (if training)
	if b.Dropout > 0 && training {
		attnOut = attnOut.Dropout(b.Dropout, training)
	}

	// Residual connection
	x, err = tensor.Add(attnOut, shortcut)
	if err != nil {
		return nil, fmt.Errorf("failed to add attention residual: %w", err)
	}

	// Step 2: Feed-forward block with residual
	// shortcut = x
	// x = Norm2(x)
	// x = FF(x)
	// x = Dropout(x)
	// x = x + shortcut

	shortcut = x

	// LayerNorm before feed-forward
	normed, err = b.Norm2.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("failed to apply Norm2: %w", err)
	}

	// Feed-forward network
	ffOut, err := b.FF.Forward(normed)
	if err != nil {
		return nil, fmt.Errorf("failed to compute feed-forward: %w", err)
	}

	// Apply dropout (if training)
	if b.Dropout > 0 && training {
		ffOut = ffOut.Dropout(b.Dropout, training)
	}

	// Residual connection
	output, err := tensor.Add(ffOut, shortcut)
	if err != nil {
		return nil, fmt.Errorf("failed to add feed-forward residual: %w", err)
	}

	return output, nil
}
