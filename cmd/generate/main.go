package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"gollm/pkg/model"
	"gollm/pkg/tensor"
	"gollm/pkg/tokenizer"
)

func main() {
	// Define command line flags
	prompt := flag.String("prompt", "Hello, I am", "Input prompt text")
	maxTokens := flag.Int("max-tokens", 10, "Number of tokens to generate")
	contextSize := flag.Int("context-size", 1024, "Maximum context window size")

	flag.Parse()

	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("            GPT-2 Text Generation")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Println()

	// Create model configuration
	config := model.DefaultGPT2Config()
	if *contextSize < config.ContextLength {
		config.ContextLength = *contextSize
	}

	fmt.Printf("Model Configuration:\n")
	fmt.Printf("  Vocab Size: %d\n", config.VocabSize)
	fmt.Printf("  Context Length: %d\n", config.ContextLength)
	fmt.Printf("  Embedding Dim: %d\n", config.EmbeddingDim)
	fmt.Printf("  Num Heads: %d\n", config.NumHeads)
	fmt.Printf("  Num Layers: %d\n", config.NumLayers)
	fmt.Printf("  Dropout: %.1f\n", config.Dropout)
	fmt.Println()

	// Create the model
	fmt.Println("Initializing GPT-2 model...")
	gptModel := model.NewGPT2Model(config)

	// Set to evaluation mode (disable dropout)
	gptModel.SetTraining(false)

	fmt.Println("Model initialized successfully!")
	fmt.Println()

	// Initialize tokenizer
	fmt.Println("Initializing tokenizer...")
	tok := tokenizer.NewTokenizer()
	tok.InitializeVocab()
	tok.SetSpecialTokens(tokenizer.GPT2VocabSize - tokenizer.NumSpecialTokens)
	fmt.Printf("Tokenizer vocabulary size: %d (base: 256, special: %d)\n", tok.VocabSize(), tokenizer.NumSpecialTokens)
	fmt.Println("Note: Using byte-level fallback for generated tokens (model is untrained)")
	fmt.Println()

	// Encode the prompt
	fmt.Printf("Input prompt: %q\n", *prompt)
	encoded, err := tok.Encode(*prompt, tokenizer.EncodeOptions{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding text: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Encoded input: %v\n", encoded)
	fmt.Printf("Number of tokens: %d\n", len(encoded))
	fmt.Println()

	// Convert to tensor
	inputData := make([]float32, len(encoded))
	for i, token := range encoded {
		inputData[i] = float32(token)
	}
	idx, err := tensor.FromSlice(inputData, []int{1, len(encoded)})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating input tensor: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("              Generating Text...")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Println()

	// Generate text
	fmt.Printf("Generating %d tokens...\n\n", *maxTokens)

	result, err := model.GenerateTextSimple(gptModel, idx, *maxTokens, config.ContextLength)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating text: %v\n", err)
		os.Exit(1)
	}

	// Convert result to token IDs
	outputTokens := make([]int, result.Shape[1])
	for i := 0; i < result.Shape[1]; i++ {
		outputTokens[i] = int(result.Get([]int{0, i}))
	}

	// Decode the output
	outputText := tok.Decode(outputTokens)

	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("                Output")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Println()
	fmt.Printf("Generated tokens: %v\n", outputTokens)
	fmt.Printf("Output length: %d tokens\n", len(outputTokens))
	fmt.Println()
	fmt.Printf("Generated text:\n%s\n", outputText)
	fmt.Println()

	// Print statistics
	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("              Statistics")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("  Input tokens:  %d\n", len(encoded))
	fmt.Printf("  Output tokens: %d\n", len(outputTokens))
	fmt.Printf("  New tokens:    %d\n", len(outputTokens)-len(encoded))
}
