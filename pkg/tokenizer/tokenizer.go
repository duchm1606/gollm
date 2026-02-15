// Package tokenizer implements Byte-Pair Encoding (BPE) tokenization
// compatible with LLaMA 3.
//
// This is an educational implementation focusing on clarity and understanding
// of the BPE algorithm. It supports training from scratch and loading
// LLaMA 3 format tokenizer.model files.
//
// Key features:
//   - Train BPE vocabulary from text corpus
//   - Encode/decode text with special token handling
//   - LLaMA 3 chat format support
//   - Load/save LLaMA 3 tokenizer.model format
package tokenizer

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Special token constants for LLaMA 3
const (
	SpecialBOS         = "<|begin_of_text|>"
	SpecialEOS         = "<|end_of_text|>"
	SpecialStartHeader = "<|start_header_id|>"
	SpecialEndHeader   = "<|end_header_id|>"
	SpecialEOT         = "<|eot_id|>" // end of turn
	SpecialEOM         = "<|eom_id|>" // end of message
	SpecialPythonTag   = "<|python_tag|>"
	SpecialImage       = "<|image|>"
	SpecialFinetunePad = "<|finetune_right_pad_id|>"
	SpecialStepID      = "<|step_id|>"

	NumReservedSpecialTokens = 256
	// Simplified pattern compatible with Go's regexp
	// Original: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
	// Simplified: Handles contractions, words, numbers, punctuation, whitespace
	// Note: Removed unsupported features: negative lookahead, possessive quantifiers
	DefaultPattern = `'s|'t|'re|'ve|'m|'ll|'d|[^\r\n0-9A-Za-z]*[0-9A-Za-z]+|[0-9]{1,3}|[^\s0-9A-Za-z]+[\r\n]*|\s*[\r\n]+|\s+`
)

// Tokenizer implements BPE tokenization
type Tokenizer struct {
	// vocab maps token ID to token bytes
	// Token 0-255: single bytes
	// Token 256+: merged tokens
	// Token 128000+: special tokens
	vocab map[int][]byte

	// inverseVocab maps token string to ID for O(1) lookup
	inverseVocab map[string]int

	// bpeRanks maps (id1, id2) to merge priority
	// Lower rank = merge first (learned earlier)
	bpeRanks map[[2]int]int

	// specialTokens maps special token name to ID
	specialTokens map[string]int

	// Configuration
	vocabSize int
	pattern   *regexp.Regexp
	bosID     int
	eosID     int
	eotID     int
	eomID     int
}

// Pair represents two adjacent token IDs
type Pair [2]int

// NewTokenizer creates a new uninitialized tokenizer
func NewTokenizer() *Tokenizer {
	t := &Tokenizer{
		vocab:         make(map[int][]byte),
		inverseVocab:  make(map[string]int),
		bpeRanks:      make(map[[2]int]int),
		specialTokens: make(map[string]int),
	}

	// Compile regex pattern
	var err error
	t.pattern, err = regexp.Compile(DefaultPattern)
	if err != nil {
		panic(fmt.Sprintf("failed to compile regex pattern: %v", err))
	}

	return t
}

// InitializeVocab initializes the base vocabulary with 256 byte tokens
// This is the starting point before any merges
func (t *Tokenizer) InitializeVocab() {
	// Clear existing vocab
	t.vocab = make(map[int][]byte)
	t.inverseVocab = make(map[string]int)
	t.bpeRanks = make(map[[2]int]int)

	// Add 256 byte tokens (0-255)
	for i := 0; i < 256; i++ {
		t.vocab[i] = []byte{byte(i)}
		t.inverseVocab[string([]byte{byte(i)})] = i
	}

	t.vocabSize = 256
}

// SetSpecialTokens configures special tokens
// In LLaMA 3, special tokens start at baseVocabSize (128000)
func (t *Tokenizer) SetSpecialTokens(baseVocabSize int) {
	// Define special tokens in order
	specials := []string{
		SpecialBOS,
		SpecialEOS,
		SpecialFinetunePad,
		SpecialStepID,
		SpecialStartHeader,
		SpecialEndHeader,
		SpecialEOM,
		SpecialEOT,
		SpecialPythonTag,
		SpecialImage,
	}

	// Add reserved tokens
	for i := 0; i < NumReservedSpecialTokens-len(specials); i++ {
		specials = append(specials, fmt.Sprintf("<|reserved_special_token_%d|>", i))
	}

	// Assign IDs
	for i, token := range specials {
		id := baseVocabSize + i
		t.specialTokens[token] = id
		t.vocab[id] = []byte(token)
		t.inverseVocab[token] = id
	}

	// Store common special token IDs for quick access
	t.bosID = t.specialTokens[SpecialBOS]
	t.eosID = t.specialTokens[SpecialEOS]
	t.eotID = t.specialTokens[SpecialEOT]
	t.eomID = t.specialTokens[SpecialEOM]
}

// VocabSize returns the current vocabulary size
func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}

// GetBOSID returns the BOS token ID
func (t *Tokenizer) GetBOSID() int {
	return t.bosID
}

// GetEOSID returns the EOS token ID
func (t *Tokenizer) GetEOSID() int {
	return t.eosID
}

// GetVocab returns the vocabulary map (for inspection/testing)
func (t *Tokenizer) GetVocab() map[int][]byte {
	return t.vocab
}

// GetSpecialTokens returns the special tokens map
func (t *Tokenizer) GetSpecialTokens() map[string]int {
	return t.specialTokens
}

// preprocessText handles special characters for BPE training
// Replaces spaces with 'Ġ' to mark word boundaries
func preprocessText(text string) string {
	var result strings.Builder
	for i, char := range text {
		if char == ' ' && i != 0 {
			// Use 'Ġ' (U+0120) to represent leading spaces
			result.WriteString("Ġ")
		} else if char != ' ' {
			result.WriteRune(char)
		}
	}
	return result.String()
}

// postProcess reverses preprocessing for decoding
// Converts 'Ġ' back to spaces
func postProcess(text string) string {
	// Replace 'Ġ' with space
	text = strings.ReplaceAll(text, "Ġ", " ")
	return text
}

// textToTokenIDs converts preprocessed text to initial byte IDs
func (t *Tokenizer) textToTokenIDs(text string) []int {
	bytes := []byte(text)
	ids := make([]int, len(bytes))
	for i, b := range bytes {
		ids[i] = int(b)
	}
	return ids
}

// countPairs counts all adjacent pairs in token sequences
func countPairs(sequences [][]int) map[Pair]int {
	pairs := make(map[Pair]int)
	for _, seq := range sequences {
		for i := 0; i < len(seq)-1; i++ {
			pair := Pair{seq[i], seq[i+1]}
			pairs[pair]++
		}
	}
	return pairs
}

// findMostFrequentPair returns the pair with highest count
func findMostFrequentPair(pairs map[Pair]int) (Pair, int) {
	var bestPair Pair
	maxCount := 0

	for pair, count := range pairs {
		if count > maxCount {
			maxCount = count
			bestPair = pair
		}
	}

	return bestPair, maxCount
}

// applyMerge replaces all occurrences of a pair with new token ID
// Uses greedy left-to-right replacement
func applyMerge(tokenIDs []int, pair Pair, newID int) []int {
	result := make([]int, 0, len(tokenIDs))
	i := 0

	for i < len(tokenIDs) {
		// Check if we can merge at current position
		if i < len(tokenIDs)-1 && tokenIDs[i] == pair[0] && tokenIDs[i+1] == pair[1] {
			result = append(result, newID)
			i += 2 // Skip both tokens
		} else {
			result = append(result, tokenIDs[i])
			i++
		}
	}

	return result
}

// Train builds a BPE vocabulary from a text corpus
//
// Algorithm:
// 1. Initialize with 256 byte tokens
// 2. Preprocess corpus (replace spaces with 'Ġ')
// 3. Convert corpus to initial token IDs
// 4. Iteratively merge most frequent pairs until vocabSize reached
// 5. Set special tokens
func (t *Tokenizer) Train(corpus []string, vocabSize int) error {
	if vocabSize < 256+NumReservedSpecialTokens {
		return fmt.Errorf("vocabSize must be at least %d", 256+NumReservedSpecialTokens)
	}

	// Step 1: Initialize vocabulary
	t.InitializeVocab()

	// Step 2 & 3: Preprocess and convert to token IDs
	sequences := make([][]int, len(corpus))
	for i, text := range corpus {
		processed := preprocessText(text)
		sequences[i] = t.textToTokenIDs(processed)
	}

	// Step 4: Iteratively merge most frequent pairs
	targetSize := vocabSize - NumReservedSpecialTokens // Reserve space for special tokens

	for t.vocabSize < targetSize {
		// Count all pairs
		pairs := countPairs(sequences)

		if len(pairs) == 0 {
			break // No more pairs to merge
		}

		// Find most frequent pair
		bestPair, count := findMostFrequentPair(pairs)

		if count < 2 {
			// No pair appears more than once, stop early
			fmt.Printf("Stopping early: no frequent pairs (max count: %d)\n", count)
			break
		}

		// Create new token
		newID := t.vocabSize
		mergedBytes := append(t.vocab[bestPair[0]], t.vocab[bestPair[1]]...)
		t.vocab[newID] = mergedBytes
		t.inverseVocab[string(mergedBytes)] = newID

		// Record merge with rank (order learned)
		t.bpeRanks[bestPair] = newID - 256

		// Apply merge to all sequences
		for i, seq := range sequences {
			sequences[i] = applyMerge(seq, bestPair, newID)
		}

		t.vocabSize++

		// Progress indicator every 1000 merges
		if (t.vocabSize-256)%1000 == 0 {
			fmt.Printf("Trained %d/%d tokens\n", t.vocabSize, targetSize)
		}
	}

	// Step 5: Set special tokens
	t.SetSpecialTokens(t.vocabSize)
	t.vocabSize += NumReservedSpecialTokens

	fmt.Printf("Training complete. Vocabulary size: %d\n", t.vocabSize)
	return nil
}

// preTokenize splits text into chunks using regex pattern
func (t *Tokenizer) preTokenize(text string) []string {
	// Find all matches
	matches := t.pattern.FindAllString(text, -1)

	// Filter out empty strings
	var result []string
	for _, match := range matches {
		if match != "" {
			result = append(result, match)
		}
	}

	return result
}

// encodeChunk applies BPE to a single text chunk
func (t *Tokenizer) encodeChunk(chunk string) []int {
	// Convert to initial byte IDs
	bytes := []byte(chunk)
	tokenIDs := make([]int, len(bytes))
	for i, b := range bytes {
		tokenIDs[i] = int(b)
	}

	// Apply BPE merges greedily by rank
	for len(tokenIDs) > 1 {
		// Find all pairs and their ranks
		pairs := make(map[Pair]int)
		for i := 0; i < len(tokenIDs)-1; i++ {
			pair := Pair{tokenIDs[i], tokenIDs[i+1]}
			if rank, ok := t.bpeRanks[pair]; ok {
				pairs[pair] = rank
			}
		}

		if len(pairs) == 0 {
			break // No more merges possible
		}

		// Find pair with best (lowest) rank
		var bestPair Pair
		bestRank := int(^uint(0) >> 1) // Max int

		for pair, rank := range pairs {
			if rank < bestRank {
				bestRank = rank
				bestPair = pair
			}
		}

		// Apply the merge
		newID := t.bpeRanks[bestPair] + 256
		tokenIDs = applyMerge(tokenIDs, bestPair, newID)
	}

	return tokenIDs
}

// EncodeOptions contains options for encoding
type EncodeOptions struct {
	BOS            bool
	EOS            bool
	AllowedSpecial []string
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string, opts EncodeOptions) ([]int, error) {
	var result []int

	// Add BOS if requested
	if opts.BOS {
		result = append(result, t.bosID)
	}

	// Pre-tokenize
	chunks := t.preTokenize(text)

	// Check for special tokens
	if len(opts.AllowedSpecial) > 0 {
		// Handle special tokens in text
		// For simplicity, we'll just check if any allowed special appears
		for _, special := range opts.AllowedSpecial {
			if strings.Contains(text, special) {
				// Split text around special token
				parts := strings.Split(text, special)
				for i, part := range parts {
					if i > 0 {
						// Add special token ID
						if id, ok := t.specialTokens[special]; ok {
							result = append(result, id)
						}
					}
					// Encode normal part
					if part != "" {
						chunks := t.preTokenize(part)
						for _, chunk := range chunks {
							ids := t.encodeChunk(chunk)
							result = append(result, ids...)
						}
					}
				}
				goto add_eos // Skip normal processing
			}
		}
	}

	// Encode each chunk
	for _, chunk := range chunks {
		ids := t.encodeChunk(chunk)
		result = append(result, ids...)
	}

add_eos:
	// Add EOS if requested
	if opts.EOS {
		result = append(result, t.eosID)
	}

	return result, nil
}

// Decode converts token IDs to text
func (t *Tokenizer) Decode(tokenIDs []int) string {
	var result strings.Builder

	for _, id := range tokenIDs {
		if token, ok := t.vocab[id]; ok {
			result.Write(token)
		}
		// Skip unknown tokens silently
	}

	// Post-process
	return postProcess(result.String())
}

// GetStats returns tokenizer statistics for debugging
func (t *Tokenizer) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})
	stats["vocabSize"] = t.vocabSize
	stats["baseTokens"] = 256
	stats["mergedTokens"] = t.vocabSize - 256 - NumReservedSpecialTokens
	stats["specialTokens"] = len(t.specialTokens)
	stats["numMerges"] = len(t.bpeRanks)

	return stats
}

// PrintTopTokens prints the most common tokens (for debugging)
func (t *Tokenizer) PrintTopTokens(n int) {
	// Get all merged tokens (IDs 256 to base vocab size)
	type tokenInfo struct {
		id   int
		text string
	}

	var tokens []tokenInfo
	for id := 256; id < t.vocabSize-NumReservedSpecialTokens; id++ {
		if token, ok := t.vocab[id]; ok {
			tokens = append(tokens, tokenInfo{id, string(token)})
		}
	}

	// Sort by ID (which corresponds to merge order)
	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i].id < tokens[j].id
	})

	fmt.Printf("\nTop %d merged tokens (by merge order):\n", n)
	for i := 0; i < n && i < len(tokens); i++ {
		fmt.Printf("  ID %d: %q\n", tokens[i].id, tokens[i].text)
	}
}
