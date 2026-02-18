# Chapter 3: Understanding Self-Attention

## 1. The Intuition: Why Do We Need Attention?

Consider the sentence: "The cat sat on the mat and looked at the bird"

When processing the word "cat", which other words should it pay attention to?

- "sat" → **Very relevant** (what the cat did)
- "mat" → **Somewhat relevant** (where the cat sat)
- "bird" → **Relevant** (what the cat looked at)
- "the" → **Less relevant** (just an article)

This is the core insight of attention: **different words in a sentence have different degrees of relevance to each other**. Traditional approaches struggled with this:

- **RNNs** process words one-by-one, making it hard to connect distant words
- **Fixed windows** can only look at nearby words, missing long-range dependencies

Attention solves this by letting every word "look at" every other word simultaneously and compute relevance scores.

---

## 2. The Core Idea: Query, Key, Value

The library analogy helps understand attention intuitively:

**Imagine you're in a library looking for books about cats:**

- **Query (Q)**: "I want books about cats"  
  *What you're looking for*

- **Key (K)**: Book titles, descriptions, and tags  
  *What identifies each item*

- **Value (V)**: The actual book content  
  *What you'll actually get*

- **Attention Score**: How well the book's title/tags match your query  
  *Computed as similarity between Q and K*

- **Output**: A weighted combination of books, weighted by relevance  
  *You read a mix of books, focusing more on the most relevant ones*

**In neural networks:**
1. Each input vector is transformed into Q, K, and V through learned linear projections
2. We compute how much each Q matches each K (attention scores)
3. Softmax converts these scores to probabilities (sum to 1)
4. We take a weighted sum of the V vectors using these probabilities

---

## 3. Mathematical Foundation

### 3.1 Scaled Dot-Product Attention

The attention formula:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Breaking it down:

1. **Q, K, V matrices**:  
   - Q: (seq_len, d_k) — queries for each position
   - K: (seq_len, d_k) — keys for each position
   - V: (seq_len, d_v) — values for each position

2. **QK^T**:  
   - Matrix multiplication computes similarity scores
   - Result: (seq_len, seq_len) — "how much does position i attend to position j"

3. **/ √d_k**:  
   - Scales the dot products
   - Prevents extremely large values that would push softmax to extreme

4. **softmax**:  
   - Converts scores to probabilities (0-1, sum to 1)
   - Higher scores = more attention

5. **× V**:  
   - Weighted sum of value vectors
   - Result: (seq_len, d_v) — output for each position

### 3.2 Why Scale by √d_k?

Here's the problem: **dot products grow with dimension**.

**Example**:  
For d_k = 64, if Q and K have values around 1, their dot product averages around 64.

Without scaling:
- Values become very large (64, 128, etc.)
- Softmax becomes "spiky" — one value dominates, others become ~0
- This is like one-hot encoding — no gradient flows to other positions

With scaling (√64 = 8):
- Values stay in reasonable range (8, 16, etc.)
- Softmax distributes attention more evenly
- Better gradients, more stable training

**Intuition**: You're normalizing the variance so attention can be selective but not exclusive.

---

## 4. Causal (Autoregressive) Attention

### 4.1 The Problem

In language generation, we predict one token at a time. When predicting the 5th word, we **cannot** look at words 6, 7, 8, etc. — they haven't been generated yet!

### 4.2 The Solution: Masking

We apply a triangular mask that blocks future positions:

```
Mask Matrix (4 tokens):
[[  0, -inf, -inf, -inf],
 [  0,    0, -inf, -inf],
 [  0,    0,    0, -inf],
 [  0,    0,    0,    0]]
```

**How it works:**
- Upper triangle = future positions → set to -infinity
- Lower triangle + diagonal = present and past → set to 0
- After adding mask to QK^T, future positions have -inf
- Softmax turns -inf → 0 probability
- Result: position i only attends to positions 0 to i

### 4.3 Why This Matters

**Training**: We can process all positions at once (parallel)
- Mask ensures causal structure automatically
- Much faster than sequential processing

**Inference**: We generate one token at a time
- Each new token can attend to all previous tokens
- Mask isn't needed explicitly (we only have past tokens)

---

## 5. Multi-Head Attention

### 5.1 One Head = One Perspective

A single attention mechanism has limited expressiveness. It's like having one expert give an opinion — useful, but limited.

### 5.2 Multiple Heads = Multiple Perspectives

**Example with "The cat sat":**

```
Head 1 (Syntactic):    cat ←→ sat  (verb relationship)
Head 2 (Semantic):     cat ←→ animal (category membership)
Head 3 (Positional):   The ←→ cat  (adjacent words)
```

Each head learns to focus on different types of relationships.

### 5.3 Implementation Details

1. **Split embedding**: If d_model = 768 and we have 12 heads, each head gets d_k = 768/12 = 64 dimensions

2. **Parallel computation**: All 12 heads compute attention simultaneously

3. **Concatenate**: Stack outputs from all heads: 12 × 64 = 768 dimensions again

4. **Final projection**: Linear layer combines all heads' outputs

### 5.4 Why It Works

- **Diverse patterns**: Different heads learn different linguistic patterns
- **Increased capacity**: More parameters, more expressive power
- **Parallelizable**: All heads compute at the same time
- **Empirically proven**: Works much better than single-head attention

---

## 6. Grouped Query Attention (GQA)

### 6.1 The Memory Problem

During generation, we cache K and V tensors to avoid recomputation. The memory cost grows quickly:

**Calculation for 12 heads:**
```
Per layer: 2 × batch × num_heads × seq_len × head_dim
          = 2 × 1 × 12 × 2048 × 64
          = 3,145,728 floats
          = ~12 MB (fp32) or ~3 MB (fp16)

12 layers: 36 MB
```

This is per sequence! And it grows linearly with sequence length.

### 6.2 The Solution: Share K and V

Instead of 12 separate K and V tensors, we use fewer groups. Multiple query heads share the same K and V.

**Visual comparison:**

```
Traditional Multi-Head Attention (12 heads):
Q: [Q0][Q1][Q2][Q3][Q4][Q5][Q6][Q7][Q8][Q9][Q10][Q11]
K: [K0][K1][K2][K3][K4][K5][K6][K7][K8][K9][K10][K11]
V: [V0][V1][V2][V3][V4][V5][V6][V7][V8][V9][V10][V11]

Grouped Query Attention (12 queries, 4 groups):
Q: [Q0][Q1][Q2][Q3][Q4][Q5][Q6][Q7][Q8][Q9][Q10][Q11]
K: [K0        ][K1        ][K2        ][K3          ]
V: [V0        ][V1        ][V2        ][V3          ]
   ↑             ↑             ↑             ↑
   Q0,Q1,Q2 use K0,V0
   Q3,Q4,Q5 use K1,V1
   Q6,Q7,Q8 use K2,V2
   Q9,Q10,Q11 use K3,V3
```

### 6.3 Memory Savings

```
MHA (12 heads): 12 K + 12 V = 24 tensors
GQA (4 groups):  4 K +  4 V =  8 tensors

Memory reduction: (24 - 8) / 24 = 67%
```

### 6.4 Does Quality Suffer?

Surprisingly, **minimal impact**:
- Query heads in the same group tend to learn similar patterns anyway
- The attention mechanism is already quite expressive
- Trade-off: Slight quality reduction for massive memory savings
- Modern models (LLaMA 2/3, Mistral) all use GQA

---

## 7. RoPE: Rotary Position Embeddings

### 7.1 Why Not Learned Positions?

Traditional approach: Add learned position vectors to input embeddings.

**Problem**: Can't generalize to sequences longer than training data. Position 1000 is just a learned vector — it has no mathematical relationship to position 999.

### 7.2 The Key Insight

RoPE encodes position by **rotating vectors** instead of adding position vectors.

**2D Example:**
```
Vector [x, y] at position m:
- Position 0: [x, y] → stays [x, y]
- Position 1: [x, y] → rotate by θ → [x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ)]
- Position 2: [x, y] → rotate by 2θ → [x·cos(2θ) - y·sin(2θ), ...]
```

Each dimension pair rotates by a different frequency, encoding position in a continuous, differentiable way.

### 7.3 Split-Halves Style

Our implementation uses the split-halves approach (like Hugging Face):

```
For head_dim = 64:
- Split into first 32 dims and last 32 dims
- Pair dimension i with dimension i+32
- Apply rotation with frequency θ_i to each pair
- Frequencies vary: θ_i = base^(-2i/d) where base = 10,000
```

**Example with pairs:**
```
Original: [d0, d1, d2, ..., d31, d32, d33, ..., d63]
Pairs:    [(d0, d32), (d1, d33), ..., (d31, d63)]

Each pair rotates by different amount based on its index
```

### 7.4 Why It Works

1. **Relative positions**: The rotation naturally encodes relative distances — how far apart two positions are is encoded in the angle difference

2. **Extrapolation**: Can handle sequences longer than training (rotation formula works for any position)

3. **No extra parameters**: Unlike learned embeddings, RoPE adds zero parameters

4. **KV cache compatible**: We apply RoPE to each new token's Q and K as we generate

---

## 8. KV Cache for Efficient Generation

### 8.1 The Naive Approach

Without caching, generating "The cat sat":

```
Step 1 - "The":
  Compute K,V for position 0 → Generate "cat"

Step 2 - "cat":
  Compute K,V for positions 0,1 → Generate "sat"

Step 3 - "sat":
  Compute K,V for positions 0,1,2 → Generate "."

Total computations: 1 + 2 + 3 = 6 K,V calculations
```

### 8.2 The Cached Approach

With KV cache:

```
Step 1 - "The":
  Compute K,V for position 0 → Cache [K0]
  Generate "cat"

Step 2 - "cat":
  Compute K,V for position 1 → Cache [K0, K1]
  Generate "sat"

Step 3 - "sat":
  Compute K,V for position 2 → Cache [K0, K1, K2]
  Generate "."

Total computations: 1 + 1 + 1 = 3 K,V calculations
```

### 8.3 Complexity Analysis

| Method | Per Token | Total (N tokens) | Example (N=1000) |
|--------|-----------|------------------|------------------|
| Naive | O(current_seq_len) | O(N²) | ~500,000 ops |
| Cache | O(1) | O(N) | ~1,000 ops |

**Speedup: 500× for N=1000!**

### 8.4 Memory Cost

The cache stores:
```
size = 2 × num_kv_groups × max_seq_len × head_dim × 4 bytes

For our config (4 groups, 2048 length, 64 head_dim):
size = 2 × 4 × 2048 × 64 × 4 = 4,194,304 bytes ≈ 4 MB
```

This is a **one-time cost** that enables O(N) generation instead of O(N²).

---

## 9. Putting It All Together

### 9.1 Complete Forward Pass

1. **Project**: Input (batch, seq, 768) → Q (768), K/V (256 for GQA)

2. **Reshape**: Split into heads
   - Q: (batch, 12 heads, seq, 64)
   - K/V: (batch, 4 groups, seq, 64)

3. **Apply RoPE**: Rotate Q and K based on positions

4. **Expand K/V**: Repeat groups to match query heads count
   - K/V: (batch, 4, seq, 64) → (batch, 12, seq, 64)

5. **Compute scores**: Q @ K^T / √64
   - Result: (batch, 12, seq, seq)

6. **Apply causal mask**: Set future positions to -inf

7. **Softmax**: Convert to attention weights

8. **Weighted sum**: Attention_weights @ V
   - Result: (batch, 12, seq, 64)

9. **Reshape**: (batch, seq, 768)

10. **Output projection**: Final linear layer

### 9.2 Shapes Flow

```
Input:              (batch, seq_len, 768)
↓ Linear projection
Q:                  (batch, seq_len, 768)
K/V:                (batch, seq_len, 256)     # 4 groups × 64
↓ Reshape for heads
Q:                  (batch, 12, seq_len, 64)
K/V:                (batch, 4, seq_len, 64)
↓ Apply RoPE
Q_rotated:          (batch, 12, seq_len, 64)
K_rotated:          (batch, 4, seq_len, 64)
↓ Expand K/V (GQA)
K/V_expanded:       (batch, 12, seq_len, 64)  # Repeat groups
↓ Attention computation
Attention output:   (batch, 12, seq_len, 64)
↓ Reshape
Concatenated:       (batch, seq_len, 768)
↓ Output projection
Final output:       (batch, seq_len, 768)
```

---

## 10. Common Pitfalls

1. **Shape errors**: Forgetting to transpose K before matrix multiplication (QK^T, not QK)

2. **Mask application**: Applying mask to wrong dimension or forgetting to add to scores before softmax

3. **RoPE offset**: When using KV cache, must apply RoPE with correct position offset (not position 0 for new tokens)

4. **GQA expansion**: Repeating K/V in wrong dimension (should expand groups to match queries, not the other way around)

5. **Softmax stability**: Not subtracting max value before softmax can cause numerical overflow with large sequences

---

## 11. Further Reading

- **"Attention Is All You Need"** (Vaswani et al., 2017) — The original transformer paper

- **"LLaMA: Open and Efficient Foundation Language Models"** — LLaMA architecture with RoPE, RMSNorm, GQA

- **"GQA: Training Generalized Multi-Query Transformer Models"** — Grouped Query Attention paper

- **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** — RoPE paper

- **Our implementation**: `/Users/duchoang/Projects/gollm/pkg/model/`
  - `attention.go` — GQA implementation
  - `rope.go` — RoPE implementation
  - See code for practical details!
