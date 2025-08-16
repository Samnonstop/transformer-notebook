# Simple Training Example for the Transformer

This supplementary file provides a basic training loop example for the Transformer model we implemented. This demonstrates how to train the model on a simple copy task.

## Training Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CopyDataset(Dataset):
    """
    Simple dataset for training Transformer to copy sequences.
    This is a toy task where the model learns to output the same sequence as input.
    """
    
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence (excluding 0 which we use for padding)
        sequence = torch.randint(1, self.vocab_size, (self.seq_len,))
        
        # For copy task, source and target are the same
        # But we shift target by one position for teacher forcing
        source = sequence
        target_input = torch.cat([torch.tensor([1]), sequence[:-1]])  # Start with SOS token
        target_output = sequence  # What we want to predict
        
        return source, target_input, target_output

# Training hyperparameters
VOCAB_SIZE = 1000
SEQ_LEN = 20
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Create dataset and dataloader
train_dataset = CopyDataset(VOCAB_SIZE, SEQ_LEN, 10000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create model
model = Transformer(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=256,  # Smaller for faster training
    n_heads=8,
    n_layers=4,   # Fewer layers for simplicity
    d_ff=1024,
    max_seq_len=100,
    dropout=0.1
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def create_masks(src, tgt_input):
    """Create attention masks for training."""
    batch_size = src.size(0)
    src_len = src.size(1)
    tgt_len = tgt_input.size(1)
    
    # Source mask (no masking for copy task)
    src_mask = torch.ones(batch_size, 1, 1, src_len).to(device)
    
    # Target causal mask
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_mask.expand(batch_size, 1, tgt_len, tgt_len).to(device)
    
    return src_mask, tgt_mask

# Training loop
print("Starting training...")
model.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
        # Move to device
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt_input)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Update learning rate
    scheduler.step()
    
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {avg_loss:.4f}')

print("Training completed!")

# Test the trained model
print("\nTesting the trained model...")
model.eval()

with torch.no_grad():
    # Create a test sequence
    test_src = torch.randint(1, VOCAB_SIZE, (1, SEQ_LEN)).to(device)
    
    # Generate output
    generated = model.generate(test_src, max_len=SEQ_LEN, start_token=1, end_token=2)
    
    print(f"Input sequence:  {test_src[0].cpu().tolist()}")
    print(f"Generated sequence: {generated[0].cpu().tolist()}")
    
    # Check accuracy (how many tokens match)
    if generated.size(1) > 1:  # Remove start token
        generated_tokens = generated[0, 1:SEQ_LEN+1].cpu()
        accuracy = (test_src[0].cpu() == generated_tokens).float().mean()
        print(f"Accuracy: {accuracy:.2%}")
```

## Key Training Concepts

### 1. Teacher Forcing
During training, we feed the correct target tokens as input to the decoder, rather than using the model's own predictions. This stabilizes training.

### 2. Loss Function
We use Cross-Entropy Loss with padding token ignoring. The model learns to predict the next token in the sequence.

### 3. Gradient Clipping
Prevents exploding gradients, which can be common in deep networks like Transformers.

### 4. Learning Rate Scheduling
Gradually reduces learning rate to help the model converge to a better solution.

### 5. Masking
- **Source mask**: Usually all 1s (no masking) unless dealing with padding
- **Target mask**: Causal mask prevents attending to future tokens

## Advanced Training Techniques

For production models, consider:

1. **Warmup scheduling**: Gradually increase learning rate at the beginning
2. **Label smoothing**: Prevents overconfident predictions
3. **Mixed precision training**: Faster training with FP16
4. **Gradient accumulation**: Simulate larger batch sizes
5. **Beam search**: Better decoding strategy than greedy

## Real Dataset Training

For real applications, replace the CopyDataset with:
- **Translation**: WMT datasets
- **Text generation**: WikiText, OpenWebText
- **Summarization**: CNN/DailyMail, XSum
- **Question Answering**: SQuAD, Natural Questions

This training example provides a foundation for understanding how Transformers are trained in practice!
