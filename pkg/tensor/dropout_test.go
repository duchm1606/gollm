package tensor

import (
	"testing"
)

func TestDropout_InferenceMode(t *testing.T) {
	// In inference mode (training=false), dropout should return a clone
	SetDropoutSeed(42)

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	tensor, err := FromSlice(data, []int{5})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	result := tensor.Dropout(0.5, false)

	// Should be a clone (same values)
	for i := range data {
		if result.Data[i] != data[i] {
			t.Errorf("Expected %f at index %d, got %f", data[i], i, result.Data[i])
		}
	}

	// Should be a different tensor (not the same pointer)
	if &result.Data[0] == &tensor.Data[0] {
		t.Error("Expected result to be a clone, not the same tensor")
	}
}

func TestDropout_ZeroProbability(t *testing.T) {
	// With p=0, all values should be kept (and scaled by 1.0)
	SetDropoutSeed(42)

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	tensor, err := FromSlice(data, []int{5})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	result := tensor.Dropout(0.0, true)

	// All values should remain unchanged
	for i := range data {
		if result.Data[i] != data[i] {
			t.Errorf("Expected %f at index %d, got %f", data[i], i, result.Data[i])
		}
	}
}

func TestDropout_TrainingMode(t *testing.T) {
	// In training mode, approximately p% of values should be dropped
	SetDropoutSeed(42)

	// Create a larger tensor to get statistically meaningful results
	data := make([]float32, 1000)
	for i := range data {
		data[i] = 1.0
	}
	tensor, err := FromSlice(data, []int{1000})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	p := float32(0.3)
	result := tensor.Dropout(p, true)

	// Count dropped values
	droppedCount := 0
	keptCount := 0
	for _, v := range result.Data {
		if v == 0 {
			droppedCount++
		} else if v == 1.0/(1.0-p) {
			keptCount++
		} else {
			t.Errorf("Unexpected value: %f (should be 0 or %f)", v, 1.0/(1.0-p))
		}
	}

	// With seed 42 and p=0.3, we expect approximately 30% dropped
	// Allow some variance (20% to 40%)
	dropRate := float32(droppedCount) / float32(len(data))
	if dropRate < 0.2 || dropRate > 0.4 {
		t.Errorf("Expected dropout rate around %f, got %f (dropped: %d, kept: %d)",
			p, dropRate, droppedCount, keptCount)
	}

	t.Logf("Dropout rate: %f (dropped: %d, kept: %d)", dropRate, droppedCount, keptCount)
}

func TestDropout_Scaling(t *testing.T) {
	// Test that kept values are properly scaled
	SetDropoutSeed(42)

	data := []float32{2.0, 2.0, 2.0, 2.0, 2.0}
	tensor, err := FromSlice(data, []int{5})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	p := float32(0.5)
	result := tensor.Dropout(p, true)

	// Kept values should be scaled by 1/(1-p) = 2.0
	expectedScale := 1.0 / (1.0 - p)
	for i, v := range result.Data {
		if v != 0 && v != 2.0*expectedScale {
			t.Errorf("Index %d: expected 0 or %f, got %f", i, 2.0*expectedScale, v)
		}
	}
}

func TestApplyDropout(t *testing.T) {
	// Test the convenience function
	SetDropoutSeed(42)

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	tensor, err := FromSlice(data, []int{5})
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	// Should work the same as Dropout method
	result := ApplyDropout(tensor, 0.5, false)

	for i := range data {
		if result.Data[i] != data[i] {
			t.Errorf("Expected %f at index %d, got %f", data[i], i, result.Data[i])
		}
	}
}
