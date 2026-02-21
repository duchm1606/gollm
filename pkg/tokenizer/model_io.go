package tokenizer

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Save writes the tokenizer to a file in LLaMA 3 tokenizer.model format
//
// File format:
//
//	<base64_encoded_token> <rank>
//
// Lines are ordered by rank (merge priority). The file contains:
// - 256 base tokens (bytes 0-255)
// - Merged tokens (ordered by when they were learned)
//
// Special tokens are NOT included in this file - they're defined
// programmatically starting at vocabSize.
func (t *Tokenizer) Save(filepath string) error {
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// Write base tokens (IDs 0-255) with ranks 0-255
	for i := 0; i < 256; i++ {
		token := t.vocab[i]
		encoded := base64.StdEncoding.EncodeToString(token)
		line := fmt.Sprintf("%s %d\n", encoded, i)
		if _, err := writer.WriteString(line); err != nil {
			return fmt.Errorf("failed to write base token %d: %w", i, err)
		}
	}

	// Write merged tokens ordered by rank
	// We need to sort by rank to maintain merge order
	type tokenWithRank struct {
		id    int
		token []byte
		rank  int
	}

	var mergedTokens []tokenWithRank
	for id := 256; id < t.vocabSize-NumSpecialTokens; id++ {
		if token, ok := t.vocab[id]; ok {
			// Find the rank for this token
			// The rank is the order in which it was merged
			rank := id - 256 // Simple mapping: merge order = id - 256
			mergedTokens = append(mergedTokens, tokenWithRank{id, token, rank})
		}
	}

	// Sort by rank
	for i := 0; i < len(mergedTokens)-1; i++ {
		for j := i + 1; j < len(mergedTokens); j++ {
			if mergedTokens[j].rank < mergedTokens[i].rank {
				mergedTokens[i], mergedTokens[j] = mergedTokens[j], mergedTokens[i]
			}
		}
	}

	// Write merged tokens
	for _, twr := range mergedTokens {
		encoded := base64.StdEncoding.EncodeToString(twr.token)
		line := fmt.Sprintf("%s %d\n", encoded, twr.rank)
		if _, err := writer.WriteString(line); err != nil {
			return fmt.Errorf("failed to write merged token %d: %w", twr.id, err)
		}
	}

	if err := writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush writer: %w", err)
	}

	return nil
}

// LoadTokenizer reads a tokenizer from a LLaMA 3 tokenizer.model file
//
// The file format is:
//
//	<base64_encoded_token> <rank>
//
// This function:
// 1. Reads all tokens and their ranks
// 2. Builds the vocabulary mapping
// 3. Reconstructs merge rules from the order (ranks)
// 4. Sets up special tokens
func LoadTokenizer(filepath string) (*Tokenizer, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	tok := NewTokenizer()
	tok.vocab = make(map[int][]byte)
	tok.inverseVocab = make(map[string]int)
	tok.bpeRanks = make(map[[2]int]int)

	scanner := bufio.NewScanner(file)
	lineNum := 0
	maxRank := -1

	// Read all tokens
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid line %d: expected 2 fields, got %d", lineNum, len(parts))
		}

		encoded := parts[0]
		rank, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, fmt.Errorf("invalid rank on line %d: %w", lineNum, err)
		}

		// Decode base64
		token, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, fmt.Errorf("failed to decode base64 on line %d: %w", lineNum, err)
		}

		// Determine token ID
		var id int
		if rank < 256 {
			// Base token (byte value)
			id = rank
		} else {
			// Merged token
			id = 256 + rank
		}

		tok.vocab[id] = token
		tok.inverseVocab[string(token)] = id

		if rank > maxRank {
			maxRank = rank
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	// Set vocabulary size (base + merged)
	if maxRank >= 256 {
		tok.vocabSize = 256 + (maxRank - 256 + 1)
	} else {
		tok.vocabSize = 256
	}

	// Reconstruct merge rules from vocabulary
	// For each merged token (id >= 256), find which pair it came from
	// This is tricky because we only have the final tokens, not the merge history
	// For now, we'll build bpeRanks from the token IDs
	for id := 256; id < tok.vocabSize; id++ {
		if token, ok := tok.vocab[id]; ok {
			rank := id - 256
			// Try to find the pair that would produce this token
			// This is a heuristic - we look for tokens that could be prefixes/suffixes
			// In practice, for inference we don't need perfect merge reconstruction
			// as long as the vocabulary is correct
			_ = token
			_ = rank
		}
	}

	// For a properly working tokenizer, we need to rebuild bpeRanks
	// This requires knowing the merge history, which we don't have from just the vocab
	// One approach: analyze the vocabulary to infer likely merges
	tok.rebuildBpeRanks()

	// Set special tokens
	tok.SetSpecialTokens(tok.vocabSize)
	tok.vocabSize += NumSpecialTokens

	return tok, nil
}

// rebuildBpeRanks attempts to reconstruct BPE merge rules from vocabulary
// This is used when loading a tokenizer.model file where we only have
// the final tokens, not the merge history.
//
// Strategy: For each merged token, try to split it into two smaller tokens
// that exist in the vocabulary. Prefer splits where both parts are common.
func (t *Tokenizer) rebuildBpeRanks() {
	// Simple heuristic: for each merged token, find the best split
	// into two existing tokens with lowest combined IDs
	// (lower IDs = learned earlier = should merge first)

	for id := 256; id < t.vocabSize-NumSpecialTokens; id++ {
		token, ok := t.vocab[id]
		if !ok || len(token) < 2 {
			continue
		}

		// Try all possible splits
		bestSplit := -1
		bestScore := int(^uint(0) >> 1) // Max int

		for i := 1; i < len(token); i++ {
			left := string(token[:i])
			right := string(token[i:])

			leftID, leftOk := t.inverseVocab[left]
			rightID, rightOk := t.inverseVocab[right]

			if leftOk && rightOk {
				// Score by sum of IDs (prefer earlier tokens)
				score := leftID + rightID
				if score < bestScore {
					bestScore = score
					bestSplit = i
				}
			}
		}

		// If we found a valid split, record it
		if bestSplit > 0 {
			leftID := t.inverseVocab[string(token[:bestSplit])]
			rightID := t.inverseVocab[string(token[bestSplit:])]
			pair := [2]int{leftID, rightID}
			t.bpeRanks[pair] = id - 256 // Rank based on merge order
		}
	}
}

// GetTokenInfo returns information about a token ID (for debugging)
func (t *Tokenizer) GetTokenInfo(id int) (string, error) {
	token, ok := t.vocab[id]
	if !ok {
		return "", fmt.Errorf("token ID %d not found", id)
	}

	tokenStr := string(token)
	info := fmt.Sprintf("ID: %d\n", id)
	info += fmt.Sprintf("Bytes: %v\n", token)
	info += fmt.Sprintf("String: %q\n", tokenStr)

	if id < 256 {
		info += "Type: Base token (byte)\n"
	} else if id >= t.vocabSize-NumSpecialTokens {
		info += "Type: Special token\n"
	} else {
		info += "Type: Merged token\n"
		rank := id - 256
		info += fmt.Sprintf("Rank: %d\n", rank)
	}

	return info, nil
}

// PrintVocabInfo prints vocabulary statistics
func (t *Tokenizer) PrintVocabInfo() {
	fmt.Println("=== Tokenizer Vocabulary Info ===")
	fmt.Printf("Total vocab size: %d\n", t.vocabSize)
	fmt.Printf("Base tokens: 256\n")

	mergedCount := 0
	for id := 256; id < t.vocabSize-NumSpecialTokens; id++ {
		if _, ok := t.vocab[id]; ok {
			mergedCount++
		}
	}
	fmt.Printf("Merged tokens: %d\n", mergedCount)
	fmt.Printf("Special tokens: %d\n", NumSpecialTokens)
	fmt.Printf("BPE merge rules: %d\n", len(t.bpeRanks))

	fmt.Println("\n=== Special Token Mapping ===")
	for name, id := range t.specialTokens {
		fmt.Printf("  %s -> %d\n", name, id)
	}

	fmt.Println("\n=== Sample Tokens ===")
	// Show some base tokens
	fmt.Println("Base tokens (first 10):")
	for i := 0; i < 10 && i < 256; i++ {
		token := t.vocab[i]
		fmt.Printf("  ID %d: %q\n", i, string(token))
	}

	// Show some merged tokens
	fmt.Println("\nMerged tokens (first 10):")
	count := 0
	for id := 256; id < t.vocabSize-NumSpecialTokens && count < 10; id++ {
		if token, ok := t.vocab[id]; ok {
			fmt.Printf("  ID %d: %q\n", id, string(token))
			count++
		}
	}
}
