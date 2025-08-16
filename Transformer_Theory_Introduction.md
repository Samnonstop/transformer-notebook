# Transformer Architecture: Theory and Data Flow

## Introduction

Welcome to this comprehensive guide on the Transformer architecture, the revolutionary neural network model introduced in "Attention is All You Need" (Vaswani et al., 2017). This document will provide you with a solid theoretical foundation before we dive into the practical implementation.

## Table of Contents
1. [Overview of the Transformer](#overview)
2. [Key Components](#components)
3. [Data Flow Analysis](#dataflow)
4. [Mathematical Foundations](#math)
5. [Advantages and Applications](#advantages)

## 1. Overview of the Transformer {#overview}

The Transformer is a sequence-to-sequence model that relies entirely on attention mechanisms, eliminating the need for recurrence and convolutions. It consists of an encoder-decoder architecture where:

- **Encoder**: Processes the input sequence and creates representations
- **Decoder**: Generates the output sequence using encoder representations and previously generated tokens

### Key Innovation
The Transformer introduces **self-attention** as the primary mechanism for capturing dependencies between positions in a sequence, regardless of their distance.

## 2. Key Components {#components}

### 2.1 Multi-Head Attention (MHA)

You're already familiar with multi-head attention and QKV concepts. In the Transformer:

- **Query (Q)**: What information are we looking for?
- **Key (K)**: What information is available?
- **Value (V)**: The actual information content

The attention mechanism computes:
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention** runs multiple attention heads in parallel:
- Each head learns different types of relationships
- Heads are concatenated and linearly projected
- Allows the model to attend to different representation subspaces

### 2.2 Position Encoding

Since Transformers have no inherent notion of sequence order, we add positional encodings:
- **Sinusoidal encodings**: Use sine and cosine functions of different frequencies
- **Learned embeddings**: Alternative approach where positions are learned parameters
- Added to input embeddings before the first layer

### 2.3 Feed-Forward Networks (FFN)

Each layer contains a position-wise feed-forward network:
- Two linear transformations with ReLU activation
- Applied to each position separately and identically
- Structure: Linear → ReLU → Linear
- Dimension typically 4x the model dimension

### 2.4 Layer Normalization and Residual Connections

- **Residual connections**: Add input to output of each sub-layer
- **Layer normalization**: Applied after each sub-layer
- Order: `LayerNorm(x + Sublayer(x))`
- Helps with gradient flow and training stability

### 2.5 Encoder Stack

Each encoder layer contains:
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections and layer normalization around each

The original paper uses 6 encoder layers.

### 2.6 Decoder Stack

Each decoder layer contains:
1. **Masked multi-head self-attention**: Prevents attending to future positions
2. **Multi-head cross-attention**: Attends to encoder output
3. Position-wise feed-forward network
4. Residual connections and layer normalization around each

The original paper uses 6 decoder layers.

## 3. Data Flow Analysis {#dataflow}

### 3.1 Input Processing

1. **Token Embedding**: Convert input tokens to dense vectors
2. **Positional Encoding**: Add position information
3. **Input Representation**: Sum of token embeddings and positional encodings

### 3.2 Encoder Flow

For each encoder layer (repeated 6 times):

```
Input → Multi-Head Self-Attention → Add & Norm → 
Feed-Forward → Add & Norm → Output to next layer
```

**Detailed Steps:**
1. **Self-Attention**: Each position attends to all positions in the input
2. **Residual Connection**: Add input to attention output
3. **Layer Normalization**: Normalize the result
4. **Feed-Forward**: Apply position-wise FFN
5. **Residual Connection**: Add input to FFN output
6. **Layer Normalization**: Final normalization

### 3.3 Decoder Flow

For each decoder layer (repeated 6 times):

```
Target Input → Masked Self-Attention → Add & Norm → 
Cross-Attention (with Encoder) → Add & Norm → 
Feed-Forward → Add & Norm → Output to next layer
```

**Detailed Steps:**
1. **Masked Self-Attention**: Attend only to previous positions
2. **Residual + LayerNorm**: Stabilize training
3. **Cross-Attention**: Query from decoder, Key/Value from encoder
4. **Residual + LayerNorm**: Stabilize training
5. **Feed-Forward**: Position-wise processing
6. **Residual + LayerNorm**: Final normalization

### 3.4 Output Generation

1. **Linear Projection**: Map decoder output to vocabulary size
2. **Softmax**: Convert to probability distribution
3. **Token Selection**: Choose next token (training: teacher forcing, inference: sampling/greedy)

## 4. Mathematical Foundations {#math}

### 4.1 Scaled Dot-Product Attention

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

Where:
- Q ∈ ℝ^(n×d_k): Query matrix
- K ∈ ℝ^(n×d_k): Key matrix  
- V ∈ ℝ^(n×d_v): Value matrix
- d_k: Dimension of keys (for scaling)

### 4.2 Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Parameters:
- W_i^Q ∈ ℝ^(d_model×d_k): Query projection for head i
- W_i^K ∈ ℝ^(d_model×d_k): Key projection for head i
- W_i^V ∈ ℝ^(d_model×d_v): Value projection for head i
- W^O ∈ ℝ^(hd_v×d_model): Output projection

### 4.3 Position-wise Feed-Forward

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Where:
- W_1 ∈ ℝ^(d_model×d_ff): First linear transformation
- W_2 ∈ ℝ^(d_ff×d_model): Second linear transformation
- d_ff = 4 × d_model (typically)

### 4.4 Positional Encoding

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

## 5. Advantages and Applications {#advantages}

### 5.1 Advantages over RNNs/CNNs

1. **Parallelization**: All positions processed simultaneously
2. **Long-range dependencies**: Direct connections between any two positions
3. **Computational efficiency**: Better utilization of modern hardware
4. **Interpretability**: Attention weights provide insight into model behavior

### 5.2 Applications

- **Machine Translation**: Original application
- **Text Summarization**: Abstractive and extractive
- **Language Modeling**: GPT family models
- **Question Answering**: BERT and variants
- **Code Generation**: GitHub Copilot, CodeT5
- **Computer Vision**: Vision Transformer (ViT)

## Key Takeaways

1. **Attention is central**: The Transformer relies entirely on attention mechanisms
2. **Parallelization**: Unlike RNNs, all positions are processed simultaneously
3. **Residual connections**: Critical for training deep networks
4. **Layer normalization**: Stabilizes training and improves convergence
5. **Positional encoding**: Necessary to provide sequence order information

## Next Steps

Now that you understand the theoretical foundations, we'll implement each component step-by-step in PyTorch. This will help you:
- Understand the practical details of each component
- See how theory translates to code
- Gain hands-on experience with the architecture
- Build intuition for hyperparameter choices

The implementation will follow the same structure as this theoretical overview, making it easy to connect concepts with code.
