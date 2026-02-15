# Byte-Pair Encoding (BPE) Tokenization - Deep Dive

> For Chapter 2: Working with Text Data

---

## Table of Contents

1. [Introduction to Tokenization](#1-introduction-to-tokenization)
2. [Why BPE?](#2-why-bpe)
3. [The BPE Algorithm](#3-the-bpe-algorithm)
4. [Phase 1: Training - Building the Vocabulary](#4-phase-1-training---building-the-vocabulary)
5. [Phase 2: Encoding - Text to Token IDs](#5-phase-2-encoding---text-to-token-ids)
6. [Phase 3: Decoding - Token IDs to Text](#6-phase-3-decoding---token-ids-to-text)
7. [Data Structures and Merging Strategies](#7-data-structures-and-merging-strategies)
8. [Pre-tokenization and Regex Patterns](#8-pre-tokenization-and-regex-patterns)
9. [Special Tokens Handling](#9-special-tokens-handling)
10. [Complete Example Walkthrough](#10-complete-example-walkthrough)
11. [Common Pitfalls and Edge Cases](#11-common-pitfalls-and-edge-cases)
12. [References](#12-references)

---

## 1. Introduction to Tokenization

### What is Tokenization?

Tokenization is the process of converting text into a sequence of integers (token IDs) that can be processed by a neural network. It's the bridge between human-readable text and machine-readable numbers.

**The Pipeline:**

```
Raw Text → Tokens → Token IDs → Embeddings → Neural Network
"Hello"    ["Hello"]   [15496]     vector(768)   ↓
                                                    ↓
                                               Output
```

### Why Not Just Use Characters?

Using individual characters (byte-level encoding):

```python
text = "This is some text"
byte_ary = bytearray(text, "utf-8")
ids = list(byte_ary)
# Result: [84, 104, 105, 115, 32, 105, 115, 32, 115, 111, 109, 101, 32, 116, 101, 120, 116]
# 17 tokens for 17 characters
```

**Problems:**

- 17 tokens for a short phrase
- Sequence length = number of characters
- Inefficient for transformer models (quadratic attention cost)

### Why Not Just Use Words?

Using whole words:

- English has ~170,000 words in common use
- Plus names, numbers, typos, multi-word expressions
- Vocabulary becomes huge
- Can't handle out-of-vocabulary (OOV) words

### Subword Tokenization: The Sweet Spot

**BPE** learns subword units:

- Common words become single tokens: `the`, `and`, `is`
- Rare words split into subwords: `tokenization` → `token` + `ization`
- Handles any text (falls back to bytes for unknown characters)

**Example:**

```
"This is some text"
→ ["This", " is", " some", " text"]
→ [1212, 318, 617, 2420]  (4 tokens instead of 17!)
```

---

## 2. Why BPE?

### Compression Ratio

| Approach    | "This is some text" | Tokens | Compression |
| ----------- | ------------------- | ------ | ----------- |
| Characters  | 17 chars            | 17     | 1.0x        |
| Words       | 4 words             | 4      | 4.25x       |
| BPE (GPT-2) | 4 subwords          | 4      | 4.25x       |

**Example**

```
"Tokenization is the process of converting text into tokens."
```

**Output:**

```
Token 0: ID=4421,  text='Token'        ← Start of "Tokenization" (capital T)
Token 1: ID=2860,  text='ization'      ← Suffix "ization"
Token 2: ID=382,   text=' is'          ← Space + "is"
Token 3: ID=290,   text=' the'         ← Space + "the"
Token 4: ID=2273,  text=' process'     ← Space + "process"
Token 5: ID=328,   text=' of'          ← Space + "of"
Token 6: ID=55111, text=' converting'  ← Space + "converting"
Token 7: ID=2201,  text=' text'        ← Space + "text"
Token 8: ID=1511,  text=' into'        ← Space + "into"
Token 9: ID=20290, text=' tokens'      ← Space + "tokens" (lowercase, plural)
```

**Key Observations:**

- `"Tokenization"` → `["Token", "ization"]` (2 tokens, not 1!)
  - The word is split because "Tokenization" is rare in training data
  - BPE reuses the common subword `"ization"` (learned from: realization, civilization, organization)
- `"Token"` (capitalized) and `" tokens"` (lowercase + space + plural) are completely different tokens with different IDs
- This is efficient: the model learns `"ization"` once and reuses it across many words
- Punctuation is a separate token

---

## 3. The BPE Algorithm

### High-Level Overview

BPE is a **compression algorithm** that iteratively merges the most frequent adjacent byte pairs.

**Core Idea:**

1. Start with a vocabulary of individual bytes (0-255)
2. Find the most frequent adjacent pair in the training data
3. Merge them into a new token
4. Add the new token to vocabulary
5. Repeat until target vocabulary size

### The Three Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                        BPE TOKENIZER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  PHASE 1: TRAIN  │──────▶│  PHASE 2: ENCODE │                 │
│  │  Build vocabulary│      │  Text → IDs      │                 │
│  └──────────────────┘      └────────┬─────────┘                 │
│          │                           │                           │
│          │ vocab.bpe                 │ ids                       │
│          │ encoder.json              │                           │
│          ▼                           ▼                           │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  PHASE 3: DECODE │◀─────│  NEURAL NETWORK  │                 │
│  │  IDs → Text      │      │  Process IDs     │                 │
│  └──────────────────┘      └──────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Training - Building the Vocabulary

### Step-by-Step Training Algorithm

#### Step 0: Initialization

**Start with 256 byte tokens:**

```
Vocabulary = {
    0: "\x00", 1: "\x01", ..., 255: "\xff"
}
```

These represent all possible byte values (0-255). In practice, many correspond to printable ASCII characters.

#### Step 1: Preprocessing

**Process the training text:**

```python
# Original text
text = "the cat in the hat"

# Replace spaces with 'Ġ' (GPT-2 convention)
# 'Ġ' = U+0120 (Latin Capital Letter G with Dot Above)
processed = "theĠcatĠinĠtheĠhat"

# Convert to token IDs using initial vocabulary (bytes)
token_ids = []
for char in processed:
    token_ids.append(ord(char))  # Get ASCII/Unicode code point

# Result: [116, 104, 101, 226, 132, 160, 99, 97, 116, ...]
# (Note: 'Ġ' is 3 bytes in UTF-8: [226, 132, 160])
```

**Why 'Ġ'?**

- GPT-2 uses 'Ġ' to represent leading spaces
- Makes space information explicit
- Distinguishes "word" from " word" (leading space)

#### Step 2: Count Adjacent Pairs

**Find all adjacent pairs and count frequencies:**

```
Token sequence: [t, h, e, Ġ, c, a, t, Ġ, i, n, Ġ, t, h, e, Ġ, h, a, t]
                [116, 104, 101, 226, 99, 97, 116, 226, 105, 110, 226, 116, 104, 101, 226, 104, 97, 116]

Adjacent pairs:
- (t, h): appears 2 times
- (h, e): appears 1 time
- (e, Ġ): appears 2 times
- (Ġ, c): appears 1 time
- (c, a): appears 1 time
- (a, t): appears 2 times (in "cat" and "hat")
- (t, Ġ): appears 1 time
- ...

Most frequent pairs:
1. (t, h): 2 times
2. (e, Ġ): 2 times
3. (a, t): 2 times
```

#### Step 3: Merge Most Frequent Pair

**Select and merge the most frequent pair:**

```
Most frequent: (t, h) = 2 occurrences

Create new token:
- New ID: 256 (next available)
- Token string: "th" (concatenation)

Update vocabulary:
vocab[256] = "th"

Record merge:
bpe_merges[(116, 104)] = 256  # (t, h) → 256

Replace in text:
Before: [t, h, e, Ġ, c, a, t, Ġ, i, n, Ġ, t, h, e, Ġ, h, a, t]
         [116, 104, 101, 226, 99, 97, 116, 226, 105, 110, 226, 116, 104, 101, 226, 104, 97, 116]
After:  [256, e, Ġ, c, a, t, Ġ, i, n, Ġ, 256, e, Ġ, h, a, t]
         [256, 101, 226, 99, 97, 116, 226, 105, 110, 226, 256, 101, 226, 104, 97, 116]
```

**Visual representation:**

```
Before: t h e   c a t   i n   t h e   h a t
        ↑ ↑                 ↑ ↑
        └──┘                └──┘
        pair 1              pair 2

After:  th e   c a t   i n   th e   h a t
        ↑↑                    ↑↑
        └┘                    └┘
        new token 256        new token 256
```

#### Step 4: Repeat Until Target Size

**Iteration 2:**

```
Current: [th, e, Ġ, c, a, t, Ġ, i, n, Ġ, th, e, Ġ, h, a, t]

Count pairs:
- (th, e): 2 times  ← Most frequent!
- (e, Ġ): 2 times
- (a, t): 2 times
- ...

Merge (th, e):
- New ID: 257
- Token: "the"
- bpe_merges[(256, 101)] = 257

Result: [257, Ġ, c, a, t, Ġ, i, n, Ġ, 257, Ġ, h, a, t]
```

**Iteration 3:**

```
Current: [the, Ġ, c, a, t, Ġ, i, n, Ġ, the, Ġ, h, a, t]

Count pairs:
- (the, Ġ): 2 times  ← Most frequent!
- (a, t): 2 times
- (Ġ, c): 1 time
- ...

Merge (the, Ġ):
- New ID: 258
- Token: "theĠ" (the + space)
- bpe_merges[(257, 226)] = 258

Result: [258, c, a, t, Ġ, i, n, Ġ, 258, h, a, t]
```

**Iteration 4:**

```
Current: [the , c, a, t, Ġ, i, n, Ġ, the , h, a, t]

Count pairs:
- (a, t): 2 times  ← Most frequent!
- (c, a): 1 time
- ...

Merge (a, t):
- New ID: 259
- Token: "at"
- bpe_merges[(97, 116)] = 259

Result: [the , c, at, Ġ, i, n, Ġ, the , h, at]
```

#### Continue Until Vocabulary Full

For GPT-2, we continue until vocabulary has 50,257 tokens:

- Start: 256 tokens (bytes)
- Iterations: 50,257 - 256 = 50,001 merges
- Each iteration adds 1 token to vocabulary

### Complete Training Example

**Training Corpus:**

```
"low lower lowest"
```

**Initial state:**

```
Text: "low lower lowest"
Processed: "lowĠlowerĠlowest"
Bytes: [108, 111, 119, 226, 108, 111, 119, 101, 114, 226, 108, 111, 119, 101, 115, 116]
```

**Iteration 1:**

```
Pairs: (l,o):3, (o,w):3, (w,Ġ):1, (Ġ,l):2, (w,e):1, (e,r):1, ...
Most frequent: (l, o) = 3 times

Merge: (l, o) → 256
New vocab: 256="lo"

Result: [256, w, Ġ, 256, w, e, r, Ġ, 256, w, e, s, t]
```

**Iteration 2:**

```
Pairs: (256,w):3, (w,Ġ):1, (Ġ,256):2, (w,e):2, ...
Most frequent: (256, w) = 3 times  ("low" appears 3 times!)

Merge: (256, w) → 257
New vocab: 257="low"

Result: [257, Ġ, 257, e, r, Ġ, 257, e, s, t]
```

**Iteration 3:**

```
Pairs: (257,Ġ):2, (Ġ,257):2, (e,r):1, (e,s):1, ...
Most frequent: tie between several pairs at 2 times
Choose: (257, Ġ) → 258
New vocab: 258="low " ("low" + space)

Result: [258, 257, e, r, Ġ, 257, e, s, t]
```

**Final vocabulary snapshot:**

```
256: "lo"
257: "low"
258: "low "
259: "er"      (from "lower")
260: "est"     (from "lowest")
...
```

---

## 5. Phase 2: Encoding - Text to Token IDs

### The Encoding Process

Encoding applies the learned merges to new text to convert it to token IDs.

**Algorithm:**

```python
def encode(text, vocab, bpe_merges):
    # 1. Pre-tokenize
    chunks = pre_tokenize(text)

    # 2. Process each chunk
    token_ids = []
    for chunk in chunks:
        # 3. Convert to initial byte tokens
        chunk_ids = bytes_to_ids(chunk)

        # 4. Apply BPE merges
        chunk_ids = apply_bpe_merges(chunk_ids, bpe_merges)

        token_ids.extend(chunk_ids)

    return token_ids
```

### Step 1: Pre-tokenization

**Purpose:** Split text into manageable chunks before BPE

**GPT-2 Regex Pattern:**

```regex
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

**What it matches:**

1. `ʻ(?i:[sdmt]|ll|ve|re)` - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
2. `[^\r\n\p{L}\p{N}]?\p{L}+` - Words with optional leading punctuation
3. `\p{N}{1,3}` - Numbers (1-3 digits)
4. ` ?[^\s\p{L}\p{N}]+[\r\n]*` - Punctuation clusters
5. `\s*[\r\n]+` - Newlines
6. `\s+(?!\S)|\s+` - Whitespace

**Example:**

```
Text: "Hello world! It's 2024."

Pre-tokenization:
["Hello", " world", "!", " It", "'s", " 202", "4", "."]

Note:
- "Hello" → no leading space (first word)
- " world" → leading space
- "2024" → split into "202" and "4" (numbers 1-3 digits)
```

### Step 2: Handle Special Tokens

Special tokens like `<|endoftext|>` must be handled separately:

```python
# If allowed_special contains "<|endoftext|>"
text = "Hello<|endoftext|>world"

# Split on special tokens
parts = ["Hello", "<|endoftext|>", "world"]

# Encode "Hello" and "world" normally
# Pass through "<|endoftext|>" as its special token ID
```

### Step 3: Apply BPE Merges

**Greedy Algorithm:**

```python
def apply_bpe_merges(token_ids, bpe_merges):
    """
    Apply BPE merges to a sequence of token IDs.
    Uses greedy approach: always merge the highest priority pair.
    """
    changed = True
    while changed and len(token_ids) > 1:
        changed = False

        # Find all adjacent pairs
        pairs = [(token_ids[i], token_ids[i+1])
                for i in range(len(token_ids)-1)]

        # Find pair with highest priority (lowest rank)
        best_pair = None
        best_rank = float('inf')

        for pair in pairs:
            if pair in bpe_merges:
                rank = bpe_merges[pair]
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

        # If no mergeable pairs, we're done
        if best_pair is None:
            break

        # Apply the merge
        new_id = bpe_merges[best_pair]
        token_ids = merge_pair(token_ids, best_pair, new_id)
        changed = True

    return token_ids

def merge_pair(token_ids, pair_to_merge, new_id):
    """Replace all occurrences of pair with new_id."""
    result = []
    i = 0
    while i < len(token_ids):
        # Check if we can merge at position i
        if (i < len(token_ids) - 1 and
            token_ids[i] == pair_to_merge[0] and
            token_ids[i+1] == pair_to_merge[1]):
            result.append(new_id)
            i += 2  # Skip both tokens
        else:
            result.append(token_ids[i])
            i += 1
    return result
```

**Encoding Example:**

```
Vocabulary learned from "the cat in the hat":
256: "th"
257: "the"
258: "the "
259: "at"
260: "c" + "at" = "cat"
261: "h" + "at" = "hat"
262: "in" (learned later)

Input text: "the hat"
Pre-processed: "theĠhat"

Step 1: Convert to bytes
[t, h, e, Ġ, h, a, t]
[116, 104, 101, 226, 104, 97, 116]

Step 2: Apply merges

Pass 1:
Pairs: (t,h), (h,e), (e,Ġ), (Ġ,h), (h,a), (a,t)
Mergeable: (t,h) → 256 (rank 0)

After merge 1:
[256, e, Ġ, h, a, t]  (th, e, space, h, a, t)

Pass 2:
Pairs: (256,e), (e,Ġ), (Ġ,h), (h,a), (a,t)
Mergeable: (256,e) → 257 "the" (rank 1)
         (a,t) → 259 "at" (rank 3)
Choose: (256,e) has lower rank

After merge 2:
[257, Ġ, h, a, t]  (the, space, h, a, t)

Pass 3:
Pairs: (257,Ġ), (Ġ,h), (h,a), (a,t)
Mergeable: (257,Ġ) → 258 "the " (rank 2)
         (a,t) → 259 "at" (rank 3)
Choose: (257,Ġ) has lower rank

After merge 3:
[258, h, a, t]  (the , h, a, t)

Pass 4:
Pairs: (258,h), (h,a), (a,t)
Mergeable: (a,t) → 259 "at" (rank 3)

After merge 4:
[258, h, 259]  (the , h, at)

Pass 5:
Pairs: (258,h), (h,259)
Mergeable: (h,259) → 261 "hat" (rank 5)

After merge 5:
[258, 261]  (the , hat)

Final result: [258, 261]
"the hat" → [258, 261]
```

---

## 6. Phase 3: Decoding - Token IDs to Text

### The Decoding Process

Decoding is the reverse: convert token IDs back to text.

**Algorithm:**

```python
def decode(token_ids, vocab):
    """Convert token IDs back to text."""
    result = []

    for tid in token_ids:
        if tid not in vocab:
            raise ValueError(f"Unknown token ID: {tid}")

        token = vocab[tid]

        # Handle special character mappings
        if tid == 198 or token == "\n":
            result.append("\n")
        elif tid == 201 or token == "\r":
            result.append("\r")
        elif token.startswith("Ġ"):
            # GPT-2 space marker: Ġword → " word"
            result.append(" " + token[1:])
        else:
            result.append(token)

    return "".join(result)
```

### Decoding Example

```
Input: [258, 261]

Step 1: Look up token IDs
- 258 → "theĠ" (from vocab)
- 261 → "hat" (from vocab)

Step 2: Apply transformations
- "theĠ" starts with "Ġ" → " the"
- "hat" doesn't start with "Ġ" → "hat"

Step 3: Concatenate
" the" + "hat" = " thehat"

Wait, that's wrong! Let's check...

Actually, looking back at our training:
258 was "the " (the + space)
261 was "hat"

So: "the " + "hat" = "the hat" ✓

But with Ġ notation:
If 258 = "theĠ" (where Ġ represents a space)
Then "theĠ" + "hat" = "the hat"

The "Ġ" is an internal representation, not an actual character in output.
```

### Handling Merge Recursion

Some tokens in the vocabulary are themselves merges of other merged tokens. When decoding, we recursively expand:

```python
def decode_token(token_id, vocab, bpe_merges, memo=None):
    """Recursively decode a token ID to its string representation."""
    if memo is None:
        memo = {}

    if token_id in memo:
        return memo[token_id]

    # Base case: single byte
    if token_id < 256:
        result = chr(token_id)
        memo[token_id] = result
        return result

    # Find what this token is a merge of
    for (id1, id2), merged_id in bpe_merges.items():
        if merged_id == token_id:
            # Recursively decode components
            part1 = decode_token(id1, vocab, bpe_merges, memo)
            part2 = decode_token(id2, vocab, bpe_merges, memo)
            result = part1 + part2
            memo[token_id] = result
            return result

    # Not found in merges, use vocab directly
    result = vocab[token_id]
    memo[token_id] = result
    return result
```

---

## 7. Data Structures and Merging Strategies

### Core Data Structures

```python
class BPETokenizer:
    def __init__(self):
        # Vocabulary: token_id -> token_string
        # Example: {0: "!", 256: "th", 257: "the", ...}
        self.vocab = {}

        # Inverse vocabulary: token_string -> token_id
        # Example: {"!": 0, "th": 256, "the": 257, ...}
        self.inverse_vocab = {}

        # BPE merges: (id1, id2) -> merged_id
        # Records which pairs were merged and in what order
        # Example: {(116, 104): 256, (256, 101): 257, ...}
        self.bpe_merges = {}

        # BPE ranks: (token1, token2) -> rank
        # For OpenAI compatibility: lower rank = learned earlier
        # Example: {("t", "h"): 0, ("th", "e"): 1, ...}
        self.bpe_ranks = {}
```

### Two Merging Strategies

#### Strategy 1: By Frequency (Training)

During training, merge by **frequency count**:

```python
def find_most_frequent_pair(token_ids):
    """Find the pair that appears most often."""
    pair_counts = Counter(zip(token_ids, token_ids[1:]))
    if not pair_counts:
        return None
    return max(pair_counts.items(), key=lambda x: x[1])[0]
```

**Example:**

```
text: "abababab"

Initial: [a, b, a, b, a, b, a, b]

Count pairs:
- (a, b): 4 times
- (b, a): 3 times

Merge: (a, b) → new_id (appears most frequently)

After: [ab, ab, ab, ab]  (4 tokens → 4 tokens, but now each is one ID)

Next iteration:
Count pairs:
- (ab, ab): 3 times

Merge: (ab, ab) → new_id

After: [abab, abab]  (4 tokens → 2 tokens)
```

#### Strategy 2: By Rank (Encoding)

During encoding, merge by **priority rank**:

```python
def find_best_pair_to_merge(token_ids, bpe_ranks):
    """Find the pair with highest priority (lowest rank)."""
    pairs = set(zip(token_ids, token_ids[1:]))

    best_pair = None
    best_rank = float('inf')

    for pair in pairs:
        if pair in bpe_ranks:
            rank = bpe_ranks[pair]
            if rank < best_rank:
                best_rank = rank
                best_pair = pair

    return best_pair
```

**Why ranks matter:**

- Ensures deterministic encoding
- Earlier merges (lower rank) should be applied first
- Maintains consistency with how vocabulary was built

**Example:**

```
Given ranks:
("t", "h"): 0      (learned first)
("th", "e"): 1     (learned second)
("t", "he"): 5     (learned later)

Input: "the"
Bytes: [t, h, e]

Possible merges:
1. Merge (t, h) → "th" first: [th, e] → then merge (th, e) → "the"
2. Merge (h, e) → "he" first: [t, he] → no further merge

Result differs! We use ranks to decide:
- (t, h) has rank 0
- (h, e) might have rank 10
- Merge rank 0 first
```

### Vocabulary Organization

**Byte tokens (0-255):**

```
0-31:   Control characters (NULL, TAB, LF, etc.)
32:     Space
33-47:  ! " # $ % & ' ( ) * + , - . /
48-57:  0 1 2 3 4 5 6 7 8 9
58-64:  : ; < = > ? @
65-90:  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
91-96:  [ \ ] ^ _ `
97-122: a b c d e f g h i j k l m n o p q r s t u v w x y z
123-255: {|}~... and extended ASCII
```

**Merged tokens (256+):**

```
256: "th"          (most common pair)
257: "er"
258: "in"
259: "an"
260: "re"
...
30000: "ing"
30001: "tion"
...
50256: "<|endoftext|>"  (special token)
```

---

## 8. Pre-tokenization and Regex Patterns

### Why Pre-tokenize?

Pre-tokenization splits text into chunks **before** BPE:

1. **Efficiency**: BPE on smaller chunks is faster
2. **Control**: Handle special cases (contractions, numbers)
3. **Consistency**: Ensures predictable splitting

### GPT-2/GPT-4 Regex Pattern Explained

```regex
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

**Breaking it down:**

#### Pattern 1: Contractions

```regex
'(?i:[sdmt]|ll|ve|re)
```

- `'` - literal apostrophe
- `(?i:...)` - case-insensitive
- `[sdmt]` - matches 's, 't, 'd, 'm
- `|ll|ve|re` - or 'll, 've, 're

**Matches:** `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`  
**Example:** `don't` → `don`, `'t`

#### Pattern 2: Words

```regex
[^\r\n\p{L}\p{N}]?\p{L}+
```

- `[^\r\n\p{L}\p{N}]?` - Optional single character that is NOT:
  - `\r\n` - carriage return/newline
  - `\p{L}` - any letter
  - `\p{N}` - any number
- `\p{L}+` - One or more letters

**Matches:** `Hello`, `world`, `.Hello`  
**Example:** `Hello` → `Hello`, `world.` → `world`, `.`

#### Pattern 3: Numbers

```regex
\p{N}{1,3}
```

- `\p{N}` - any digit (0-9)
- `{1,3}` - between 1 and 3 digits

**Matches:** `1`, `42`, `365`  
**Example:** `2024` → `202`, `4`

#### Pattern 4: Punctuation

```regex
 ?[^\s\p{L}\p{N}]+[\r\n]*
```

- ` ?` - Optional space
- `[^\s\p{L}\p{N}]+` - One or more characters that are NOT:
  - `\s` - whitespace
  - `\p{L}` - letters
  - `\p{N}` - numbers
- `[\r\n]*` - Optional carriage returns/newlines

**Matches:** `!!!`, `???`, `...`  
**Example:** `Hello!!!` → `Hello`, `!!!`

#### Pattern 5: Newlines

```regex
\s*[\r\n]+
```

- `\s*` - Zero or more whitespace
- `[\r\n]+` - One or more CR/LF

**Matches:** `\n`, `\r\n`, `  \n`  
**Example:** `line1\nline2` → `line1`, `\n`, `line2`

#### Pattern 6: Whitespace

```regex
\s+(?!\S)|\s+
```

- `\s+(?!\S)` - Whitespace not followed by non-whitespace (trailing spaces)
- `|\s+` - OR any whitespace

**Matches:** ` `, `   `, trailing spaces

### Complete Pre-tokenization Example

```
Input: "Don't eat 100 apples!!!\nWhy?"

Pattern matches:
1. "Don't"     → Pattern 2 (word with apostrophe split)
   - "Don" + "'t"
2. " "          → Pattern 6 (space)
3. "eat"        → Pattern 2
4. " "          → Pattern 6
5. "100"        → Pattern 3 → "10" + "0" (1-3 digits each)
   - Actually: "100" matches as one token
6. " "          → Pattern 6
7. "apples"     → Pattern 2
8. "!!!"        → Pattern 4
9. "\n"         → Pattern 5
10. "Why"       → Pattern 2
11. "?"         → Pattern 4

Result: ["Don", "'t", " ", "eat", " ", "100", " ", "apples", "!!!", "\n", "Why", "?"]
```

### nanochat Variation

nanochat uses a slightly modified pattern:

```regex
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

**Changes:**

- `\p{N}{1,2}` instead of `{1,3}` - Numbers split into 1-2 digits (better for small vocab)
- Possessive quantifiers `?+` and `++` for performance

---

## 9. Special Tokens Handling

### Types of Special Tokens

#### 1. Structural Tokens

| Token             | Purpose           | Example Usage              |
| ----------------- | ----------------- | -------------------------- |
| `<endoftext>`     | Document boundary | Separate training examples |
| `<begin_of_text>` | Document start    | LLaMA 3 style              |
| `<end_of_text>`   | Document end      | LLaMA 3 style              |

#### 2. Chat/Conversation Tokens

| Token                      | Purpose                    |
| -------------------------- | -------------------------- |
| `<    user_start       >`  | Start of user message      |
| `<    user_end        >`   | End of user message        |
| `<     assistant_start  >` | Start of assistant message |
| `<    assistant_end   >`   | End of assistant message   |
| `<    eot_id          >`   | End of turn                |

#### 3. Special Characters

| Token | ID (GPT-2) | Represents      |
| ----- | ---------- | --------------- |
| `Ċ`   | 198        | `\n` (newline)  |
| `Ġ`   | 220        | Leading space   |
| `ĠĠ`  | multiple   | Multiple spaces |

### Handling Special Tokens During Encoding

```python
def encode_with_special_tokens(text, allowed_special=None):
    """
    Encode text, handling special tokens appropriately.

    Args:
        text: Input text string
        allowed_special: Set of special tokens allowed in text
                         None = no special tokens allowed
    """
    if allowed_special is None:
        # Check if any special tokens appear in text
        disallowed = [tok for tok in SPECIAL_TOKENS if tok in text]
        if disallowed:
            raise ValueError(f"Disallowed special tokens: {disallowed}")

    # Split on special tokens
    if allowed_special:
        pattern = "(" + "|".join(re.escape(t) for t in allowed_special) + ")"
        parts = re.split(pattern, text)
    else:
        parts = [text]

    # Encode each part
    token_ids = []
    for part in parts:
        if part in allowed_special:
            # Pass through special token ID
            token_ids.append(inverse_vocab[part])
        else:
            # Normal encoding
            token_ids.extend(encode(part))

    return token_ids
```

### Special Token Example

```python
text = "Hello<|endoftext|>World"

# Without allowed special
encode(text)  # Raises ValueError!

# With allowed special
encode(text, allowed_special={"<|endoftext|>"})
# Result: [15496, 50256, 18798]
#         Hello  <|endoftext|>  World

# Visual breakdown
tokens = ["Hello", "<|endoftext|>", "World"]
ids =      [15496,    50256,          18798]
```

### GPT-2 Special Token IDs

```python
# Key IDs in GPT-2 vocabulary
ID_198 = "Ċ"        # Newline representation
ID_220 = "Ġ"        # Space marker
ID_50256 = "<|endoftext|>"  # Special boundary token

# Special characters that need mapping
decode_map = {
    "Ġ": " ",           # Leading space
    "Ċ": "\n",          # Newline
    "\r": "\r",         # Carriage return (ID 201)
}
```

---

## 10. Complete Example Walkthrough

### Training a Tiny BPE Tokenizer

**Training corpus:**

```
corpus = [
    "low lower lowest",
    "high higher highest",
]
```

**Target vocabulary size:** 300 (for demo)

#### Initialization

```python
# Start with 256 byte tokens
vocab = {i: chr(i) for i in range(256)}
inverse_vocab = {chr(i): i for i in range(256)}
bpe_merges = {}
```

#### Iteration 1: First Merge

**Preprocess:**

```
"low lower lowest" → "lowĠlowerĠlowest"
"high higher highest" → "highĠhigherĠhighest"
```

**Count all pairs across corpus:**

```
Most frequent pairs:
1. (l, o): 6 times   (in low, lower, lowest, high, higher, highest)
2. (o, w): 3 times   (in low, lower, lowest)
3. (g, h): 3 times   (in high, higher, highest)
4. (h, i): 3 times   (in high, higher, highest)
...
```

**Merge (l, o):**

```
New ID: 256
New token: "lo"
bpe_merges[(108, 111)] = 256  # ord('l')=108, ord('o')=111

Text after merge:
"lo wĠlo werĠlo west"
"highĠhigherĠhighest"  # unchanged
```

#### Iteration 2: Second Merge

**Count pairs:**

```
"lo wĠlo werĠlo west"

Pairs with "lo":
- (lo, w): 3 times  ← Most frequent!
- (w, Ġ): 1 time
- (Ġ, lo): 2 times
- ...

Merge: (256, 119) → 257  # "lo" + "w" = "low"
```

**Result:**

```
"low Ġlow erĠlow est"
# "low" appears 3 times in first sentence
```

#### Iteration 3: Third Merge

**Count pairs:**

```
- (low, Ġ): 2 times
- (Ġ, low): 2 times
- ...
```

**Merge (low, Ġ) → 258:**

```
"low low erĠlow est"
# "low " (low + space) appears 2 times
```

#### Iteration 4: Discovering Patterns

**Count pairs:**

```
- (low, low): 1 time  # Not helpful
- (e, r): 2 times     # in "lower" and "higher"
- ...
```

**Merge (e, r) → 259:**

```
"low low erĠlow est"  # "er" is suffix
"high high erĠhigh est"  # "er" appears here too!
```

Now we see cross-word patterns emerging!

#### Iteration 5: Continuing

**Merge (er, Ġ) → 260:**

```
# After previous merges
Text: "low lower Ġlow est"
Wait, that's wrong. Let me trace carefully.

Actually, after merging (e,r)→259:
"low low 259Ġlow est"  # where 259="er"
"high high 259Ġhigh est"

Now (259, Ġ) appears 2 times!
Merge: (259, Ġ) → 260  # "er "
```

#### Continuing to Target Size

After many iterations, key merges learned:

```
256: "lo"
257: "low"
258: "low "
259: "er"
260: "er "
261: "est"
262: "est "
263: "high"
264: "high "
265: "igh"
...
```

### Using the Trained Tokenizer

**Encode:** `"lower"`

```
Step 1: Preprocess → "lower"
Step 2: Byte encoding → [l, o, w, e, r]
Step 3: Apply merges (greedy by rank):
  - (l,o)→256: [256, w, e, r]  # "lo"
  - (256,w)→257: [257, e, r]   # "low"
  - (e,r)→259: [257, 259]      # "er"
  - Done: "low" + "er" = "lower"

Result: [257, 259]
```

**Decode:** `[257, 259]`

```
257 → "low" (from vocab)
259 → "er" (from vocab)
Concatenate: "low" + "er" = "lower"
```

## 11. Common Pitfalls and Edge Cases

### Pitfall 1: Order of Merges Matters

```python
# Consider text "ababa"
# Merges learned:
# - (a, b) → X
# - (b, a) → Y
# - (X, a) → Z

# Encoding depends on merge order!
# If (a,b) learned first: "ababa" → XaX → ?
# If (b,a) learned first: "ababa" → aYa → ?

# Solution: Use ranks (priority) during encoding
```

### Pitfall 2: Ambiguous Boundaries

```
Text: "tokenization"

Could be tokenized as:
- [token, ization]
- [to, kenization]
- [tokenization]
- [tok, enization]

BPE picks the FIRST merge with lowest rank.
Not necessarily linguistically correct!
```

### Pitfall 3: Case Sensitivity

```python
"Hello" ≠ "hello" ≠ "HELLO"

In GPT-2:
- "Hello" → [15496]
- "hello" → [31373]
- "HELLO" → different ID

This triples vocabulary usage for case variations!
```

### Pitfall 4: Numbers

```python
"1234" → ["12", "34"]  # or ["1", "23", "4"], depends on pattern
"2024" → ["20", "24"]  # Not treated as single token
"1000000" → many tokens

Numbers are not semantic in BPE!
```

### Pitfall 5: Unicode and Multi-byte Characters

```python
# Emoji: 👍 (4 bytes in UTF-8)
bytes = [240, 159, 145, 133]
# Becomes 4 separate tokens if not merged

# Chinese: 你好
# Each character is 3 bytes in UTF-8
# 6 tokens for 2 characters!

# Solution: BPE eventually learns common multibyte sequences
```

### Pitfall 6: Infinite Loops in Decoding

```python
# If vocab contains:
# 256: "ab"
# 257: "bc"
# 258: "abc"

# And bpe_merges says:
# (a, b) → 256
# (256, c) → 258
# But vocab[258] = "abc"

# When decoding 258, do we:
# - Use vocab directly: "abc"
# - Or expand: (256, c) → "ab" + "c" → "abc"

# Either works, but must be consistent!
```

### Pitfall 7: Special Token Injection

```python
# Dangerous: Allowing user input with special tokens
text = user_input  # "Click here <|endoftext|> for prize"
encode(text, allowed_special=None)  # Should raise error

# But if allowed:
encode(text, allowed_special={"<|endoftext|>"})
# IDs: [..., 50256, ...]

# During training, this could prematurely end the document!
```

### Edge Case: Empty String

```python
encode("")  # Should return []
decode([])  # Should return ""
```

### Edge Case: Only Whitespace

```python
encode("   ")  # [220, 220, 220]  # Three "Ġ" tokens
decode([220, 220, 220])  # "   "
```

### Edge Case: Unknown Characters

```python
# Character not in training data
encode("🇯🇵")  # Rare emoji

# Falls back to bytes:
# Each flag is 4 bytes, 2 flags = 8 tokens!
```

---

## 12. References

### From This Repository

- **Educational implementation**: `.docs/LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb`
  - Complete Python implementation from scratch
  - Includes training and encoding
- **Simple usage**: `.docs/LLMs-from-scratch/pkg/llms_from_scratch/ch02.py`
  - Wrapper around tiktoken
  - Dataset and dataloader utilities
- **Production tokenizer**: `.docs/nanochat/nanochat/tokenizer.py`
  - GPT-4 style tokenizer
  - RustBPE + tiktoken hybrid
  - Chat format handling
- **LLaMA 3 tokenizer**: `.docs/llama-models/models/llama3/tokenizer.py`
  - tiktoken-based
  - Special tokens for chat

### External References

- **Original BPE paper**: "A New Algorithm for Data Compression" (Gage, 1994)
  - https://github.com/tpn/pdfs/blob/master/A%20New%20Algorithm%20for%20Data%20Compression%20(1994).pdf
- **OpenAI GPT-2**: https://github.com/openai/gpt-2
  - Original tokenizer implementation
- **tiktoken**: https://github.com/openai/tiktoken
  - Fast BPE in Rust
  - Used by GPT-2, GPT-4
- **minBPE**: https://github.com/karpathy/minbpe
  - Karpathy's minimal BPE implementation
  - Good reference for understanding
- **BPE Paper for NMT**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
  - Popularized BPE for NLP
