"""
Generate reference test data using PyTorch for Go implementation testing.

This script generates binary test files for various components:
- Attention mechanisms (causal, multi-head, GQA)
- RoPE (Rotary Position Embeddings)
- Normalization layers (RMSNorm)
- Feedforward networks (SwiGLU)

Usage:
    python scripts/generate_test_weights.py
"""

import torch
import numpy as np
import json
import os
from pathlib import Path


def save_tensor(tensor, filepath):
    """Save PyTorch tensor as binary float32 (little-endian)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy float32 and save
    tensor = tensor.detach().cpu().to(torch.float32)
    tensor.numpy().tofile(filepath)
    print(f"  Saved: {filepath} (shape={list(tensor.shape)})")


def load_tensor(filepath, shape):
    """Load binary file as numpy array with given shape."""
    data = np.fromfile(filepath, dtype="float32")
    return data.reshape(shape)


def generate_causal_attention_tests():
    """Generate test data for causal self-attention."""
    print("\n=== Generating Causal Self-Attention Tests ===")

    # Config
    batch_size, seq_len = 2, 10
    d_in, d_out = 768, 768

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create input and weights
    x = torch.randn(batch_size, seq_len, d_in)
    w_q = torch.randn(d_in, d_out)
    w_k = torch.randn(d_in, d_out)
    w_v = torch.randn(d_in, d_out)

    # Forward pass: Q, K, V projections
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # Attention scores: Q @ K^T / sqrt(d_out)
    scores = q @ k.transpose(-2, -1) / (d_out**0.5)

    # Causal mask (upper triangle = -inf)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    # Softmax and output
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v

    # Save all tensors
    save_tensor(x, "testdata/attention/causal_input.bin")
    save_tensor(w_q, "testdata/attention/causal_w_q.bin")
    save_tensor(w_k, "testdata/attention/causal_w_k.bin")
    save_tensor(w_v, "testdata/attention/causal_w_v.bin")
    save_tensor(output, "testdata/attention/causal_expected.bin")

    # Also save intermediate values for debugging
    save_tensor(q, "testdata/attention/causal_q.bin")
    save_tensor(k, "testdata/attention/causal_k.bin")
    save_tensor(v, "testdata/attention/causal_v.bin")
    save_tensor(weights, "testdata/attention/causal_weights.bin")


def generate_multihead_attention_tests():
    """Generate test data for multi-head attention."""
    print("\n=== Generating Multi-Head Attention Tests ===")

    batch_size, seq_len = 2, 10
    d_in, d_out = 768, 768
    num_heads = 12
    head_dim = d_out // num_heads

    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch_size, seq_len, d_in)

    # Multi-head weights
    w_q = torch.randn(d_in, d_out)
    w_k = torch.randn(d_in, d_out)
    w_v = torch.randn(d_in, d_out)
    w_out = torch.randn(d_out, d_out)

    # Projections
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # Reshape to (batch, heads, seq, head_dim)
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Attention scores
    scores = q @ k.transpose(-2, -1) / (head_dim**0.5)

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    # Softmax and attention
    weights = torch.softmax(scores, dim=-1)
    attn_out = weights @ v

    # Reshape back and project
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_out)
    output = attn_out @ w_out

    # Save tensors
    save_tensor(x, "testdata/attention/mha_input.bin")
    save_tensor(w_q, "testdata/attention/mha_w_q.bin")
    save_tensor(w_k, "testdata/attention/mha_w_k.bin")
    save_tensor(w_v, "testdata/attention/mha_w_v.bin")
    save_tensor(w_out, "testdata/attention/mha_w_out.bin")
    save_tensor(output, "testdata/attention/mha_expected.bin")


def generate_gqa_attention_tests():
    """Generate test data for Grouped Query Attention (GQA)."""
    print("\n=== Generating Grouped Query Attention (GQA) Tests ===")

    batch_size, seq_len = 2, 10
    d_in, d_out = 768, 768
    num_heads = 12
    num_kv_groups = 4
    head_dim = d_out // num_heads

    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch_size, seq_len, d_in)

    # GQA weights: KV heads = num_kv_groups
    kv_heads = num_kv_groups
    w_q = torch.randn(d_in, num_heads * head_dim)
    w_k = torch.randn(d_in, kv_heads * head_dim)
    w_v = torch.randn(d_in, kv_heads * head_dim)
    w_out = torch.randn(d_out, d_out)

    # Projections
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # Reshape
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)

    # Repeat K, V to match Q heads
    k = k.repeat_interleave(num_heads // kv_heads, dim=1)
    v = v.repeat_interleave(num_heads // kv_heads, dim=1)

    # Attention
    scores = q @ k.transpose(-2, -1) / (head_dim**0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    attn_out = weights @ v

    # Reshape and project
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_out)
    output = attn_out @ w_out

    # Save tensors
    save_tensor(x, "testdata/attention/gqa_input.bin")
    save_tensor(w_q, "testdata/attention/gqa_w_q.bin")
    save_tensor(w_k, "testdata/attention/gqa_w_k.bin")
    save_tensor(w_v, "testdata/attention/gqa_w_v.bin")
    save_tensor(w_out, "testdata/attention/gqa_w_out.bin")
    save_tensor(output, "testdata/attention/gqa_expected.bin")


def generate_rope_tests():
    """Generate test data for RoPE (Rotary Position Embeddings)."""
    print("\n=== Generating RoPE Tests ===")

    batch_size = 1
    num_heads = 4
    seq_len = 10
    head_dim = 64
    max_seq = 20
    theta_base = 10000.0

    torch.manual_seed(42)
    np.random.seed(42)

    # Compute RoPE frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    angles = torch.outer(positions, inv_freq)

    # Duplicate for both dimensions (split-halves approach)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Save RoPE parameters
    save_tensor(cos, "testdata/attention/rope_cos.bin")
    save_tensor(sin, "testdata/attention/rope_sin.bin")

    # Test RoPE application
    test_x = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Apply RoPE (split-halves style)
    x1 = test_x[..., : head_dim // 2]
    x2 = test_x[..., head_dim // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)

    cos_slice = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin_slice = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x_rotated = test_x * cos_slice + rotated * sin_slice

    save_tensor(test_x, "testdata/attention/rope_input.bin")
    save_tensor(x_rotated, "testdata/attention/rope_expected.bin")

    # Generate config for RoPE
    config = {"head_dim": head_dim, "max_seq_len": max_seq, "theta_base": theta_base}

    with open("testdata/attention/rope_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: testdata/attention/rope_config.json")


def generate_rmsnorm_tests():
    """Generate test data for RMSNorm."""
    print("\n=== Generating RMSNorm Tests ===")

    batch_size, seq_len, d_model = 2, 10, 768
    eps = 1e-6

    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch_size, seq_len, d_model)

    # RMSNorm weights (learnable scale)
    gamma = torch.ones(d_model)

    # Compute RMSNorm
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    normalized = x / rms * gamma

    save_tensor(x, "testdata/attention/rmsnorm_input.bin")
    save_tensor(gamma, "testdata/attention/rmsnorm_gamma.bin")
    save_tensor(normalized, "testdata/attention/rmsnorm_expected.bin")


def generate_swiglu_tests():
    """Generate test data for SwiGLU feedforward."""
    print("\n=== Generating SwiGLU Tests ===")

    batch_size, seq_len = 2, 10
    d_model, hidden_dim = 768, 2048

    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(batch_size, seq_len, d_model)

    # SwiGLU weights
    w_gate = torch.randn(d_model, hidden_dim)
    w_up = torch.randn(d_model, hidden_dim)
    w_down = torch.randn(hidden_dim, d_model)

    # Forward pass
    gate = x @ w_gate
    up = x @ w_up

    # SiLU activation: x * sigmoid(x)
    silu_gate = gate * torch.sigmoid(gate)

    # Element-wise multiplication
    hidden = silu_gate * up

    # Output projection
    output = hidden @ w_down

    save_tensor(x, "testdata/attention/swiglu_input.bin")
    save_tensor(w_gate, "testdata/attention/swiglu_w_gate.bin")
    save_tensor(w_up, "testdata/attention/swiglu_w_up.bin")
    save_tensor(w_down, "testdata/attention/swiglu_w_down.bin")
    save_tensor(output, "testdata/attention/swiglu_expected.bin")


def generate_config():
    """Generate config JSON for Go tests."""
    print("\n=== Generating Test Config ===")

    config = {
        "vocab_size": 32768,
        "context_length": 2048,
        "embedding_dim": 768,
        "num_heads": 12,
        "num_kv_groups": 4,
        "head_dim": 64,
        "hidden_dim": 2048,
        "num_layers": 12,
        "dropout": 0.0,
        "use_bias": False,
        "rope_base": 10000.0,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
    }

    with open("testdata/attention/test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: testdata/attention/test_config.json")


def print_summary():
    """Print summary of generated test files."""
    print("\n" + "=" * 60)
    print("TEST DATA GENERATION SUMMARY")
    print("=" * 60)

    testdata_dir = Path("testdata/attention")
    if testdata_dir.exists():
        files = list(testdata_dir.glob("*.bin")) + list(testdata_dir.glob("*.json"))
        print(f"\nTotal files generated: {len(files)}")
        print(f"\nFiles in {testdata_dir}:")
        for f in sorted(files):
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name:40} ({size:>8.2f} KB)")

    print("\n" + "=" * 60)
    print("To run Go tests:")
    print("  cd /Users/duchoang/Projects/gollm && go test ./pkg/model/...")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING REFERENCE TEST DATA")
    print("=" * 60)

    # Ensure testdata directory exists
    os.makedirs("testdata/attention", exist_ok=True)

    # Generate all test data
    generate_causal_attention_tests()
    generate_multihead_attention_tests()
    generate_gqa_attention_tests()
    generate_rope_tests()
    generate_rmsnorm_tests()
    generate_swiglu_tests()
    generate_config()

    # Print summary
    print_summary()

    print("\nTest data generated successfully!")
