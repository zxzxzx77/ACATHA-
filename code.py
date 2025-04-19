import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import gc

# Dataset class for QA-HMEs with smaller image size
class HMEDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None, max_len=150):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        
        # Read label file
        self.data = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    input_expr, img_file, target_expr = parts
                    self.data.append((img_file, target_expr))
        
        # Create vocabulary
        self.vocab = set()
        for _, target_expr in self.data:
            for token in target_expr.split():
                self.vocab.add(token)
        
        self.vocab = ["<PAD>", "< SOS >", "<EOS>", "<UNK>"] + sorted(list(self.vocab))
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_file, target_expr = self.data[idx]
        img_path = os.path.join(self.root_dir, img_file)
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize target
        tokens = ["< SOS >"] + target_expr.split() + ["<EOS>"]
        target_indices = [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
        
        # Pad target
        if len(target_indices) < self.max_len:
            target_indices += [self.token2idx["<PAD>"]] * (self.max_len - len(target_indices))
        else:
            target_indices = target_indices[:self.max_len]
        
        return {
            'image': image,
            'target': torch.tensor(target_indices, dtype=torch.long),
            'target_length': min(len(tokens), self.max_len),
            'raw_target': target_expr
        }

# Positional Encoding for Transformer with smaller dimensions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Memory-efficient Hierarchical Feature Extractor with reduced channels
class HierarchicalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):  # Reduced feature dimension
        super(HierarchicalFeatureExtractor, self).__init__()
        
        # Full resolution branch with reduced channels
        self.full_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),  # Reduced from 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Reduced from 64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Half resolution branch with reduced channels
        self.half_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # Reduced from 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Reduced from 64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Quarter resolution branch with reduced channels
        self.quarter_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # Reduced from 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Reduced from 64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Eighth resolution branch with reduced channels
        self.eighth_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # Reduced from 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Reduced from 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Reduced from 64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Feature fusion with reduced channels
        self.fusion = nn.Sequential(
            nn.Conv2d(32*4, feature_dim, kernel_size=1, stride=1),  # Reduced from 64*4
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Get features at different scales
        full_features = self.full_branch(x)
        half_features = self.half_branch(x)
        quarter_features = self.quarter_branch(x)
        eighth_features = self.eighth_branch(x)
        
        # Resize to match full resolution but with smaller size
        h, w = full_features.size(2), full_features.size(3)
        half_up = nn.functional.interpolate(half_features, size=(h, w), mode='bilinear', align_corners=True)
        quarter_up = nn.functional.interpolate(quarter_features, size=(h, w), mode='bilinear', align_corners=True)
        eighth_up = nn.functional.interpolate(eighth_features, size=(h, w), mode='bilinear', align_corners=True)
        
        # Concatenate features
        concat_features = torch.cat([full_features, half_up, quarter_up, eighth_up], dim=1)
        
        # Fuse features
        fused_features = self.fusion(concat_features)
        
        # Clean up intermediate tensors to save memory
        del full_features, half_features, quarter_features, eighth_features, half_up, quarter_up, eighth_up, concat_features
        torch.cuda.empty_cache()
        
        return fused_features

# Memory-efficient Adaptive Context Integration
class AdaptiveContextIntegration(nn.Module):
    def __init__(self, feature_dim=128):  # Reduced feature dimension
        super(AdaptiveContextIntegration, self).__init__()
        
        # Confidence predictor with reduced channels
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//4, kernel_size=3, padding=1),  # Reduced from feature_dim//2
            nn.BatchNorm2d(feature_dim//4),
            nn.ReLU(),
            nn.Conv2d(feature_dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Context processor with reduced channels and fewer dilation layers
        self.context_processor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
            # Removed the third dilated convolution to save memory
        )
        
    def forward(self, x):
        # Process in two steps to save memory
        confidence = self.confidence_predictor(x)
        context_features = self.context_processor(x)
        
        # Efficient integration
        integrated_features = x * (1 - confidence) + context_features * confidence
        
        # Clean up to save memory
        del context_features
        torch.cuda.empty_cache()
        
        return integrated_features, confidence

# Memory-efficient Bidirectional Structure-Aware Decoder - FIXED to ensure float outputs
class BidirectionalStructureAwareDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, nhead=4, num_layers=2, dropout=0.1):  # Reduced parameters
        super(BidirectionalStructureAwareDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Structure-aware transformer layers with fewer heads and layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Embed tokens and add positional encoding
        tgt_embedded = self.token_embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Transformer decoder
        output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask, 
                                     tgt_key_padding_mask=tgt_key_padding_mask)
        
        # Project to vocabulary
        output = self.output_layer(output)
        
        # CRITICAL FIX: Explicitly ensure output is float32
        output = output.to(torch.float32)
        
        return output

# Memory-efficient ACATHA model with fixed mask generation and data type handling
class ACATHA(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, nhead=4, num_decoder_layers=2, dropout=0.1):  # Reduced parameters
        super(ACATHA, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Hierarchical Multi-Scale Feature Extraction
        self.feature_extractor = HierarchicalFeatureExtractor(in_channels=3, feature_dim=hidden_dim)
        
        # Adaptive Context Integration
        self.context_integrator = AdaptiveContextIntegration(feature_dim=hidden_dim)
        
        # Feature-to-sequence converter with reduced output size
        self.feature_converter = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16))  # Reduced from (8, 32)
        )
        
        # Bidirectional Structure-Aware Decoder
        self.decoder = BidirectionalStructureAwareDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, img, tgt=None, teacher_forcing_ratio=1.0):
        # Feature extraction
        features = self.feature_extractor(img)
        
        # Adaptive context integration
        features, confidence = self.context_integrator(features)
        
        # Convert to sequence representation
        features = self.feature_converter(features)
        batch_size, channels, height, width = features.size()
        
        # Reshape for transformer input (seq_len, batch, features)
        memory = features.view(batch_size, channels, -1).permute(2, 0, 1)
        
        # Clean up to save memory
        del features
        torch.cuda.empty_cache()
        
        if self.training and tgt is not None:
            # Training mode with teacher forcing
            tgt_input = tgt[:, :-1]  # Remove last token (<EOS> or <PAD>)
            tgt_mask = self._generate_square_subsequent_mask(tgt_input.size(1)).to(tgt.device)
            tgt_input = tgt_input.permute(1, 0)  # (seq_len, batch)
            
            output = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
            
            # CRITICAL FIX: Ensure output is float32
            output = output.to(torch.float32)
            
            output = output.permute(1, 0, 2)  # (batch, seq_len, vocab_size)
            
            # Clean up to save memory
            del memory, tgt_mask, tgt_input
            torch.cuda.empty_cache()
            
            return output, confidence
        else:
            # Inference mode
            device = next(self.parameters()).device
            batch_size = img.size(0)
            max_len = 100  # Reduced from 150 for memory efficiency
            
            # Start with < SOS > token
            output_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for i in range(max_len - 1):
                tgt_mask = self._generate_square_subsequent_mask(output_tokens.size(1)).to(device)
                tgt_input = output_tokens.permute(1, 0)  # (seq_len, batch)
                
                decoder_output = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
                
                # CRITICAL FIX: Ensure output is float32
                decoder_output = decoder_output.to(torch.float32)
                
                decoder_output = decoder_output.permute(1, 0, 2)  # (batch, seq_len, vocab_size)
                
                # Get next token
                next_tokens = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_tokens], dim=1)
                
                # Clean up intermediate tensors
                del decoder_output, tgt_mask, tgt_input
                torch.cuda.empty_cache()
                
                # Stop if all sequences have <EOS>
                if (next_tokens == 2).all():  # <EOS> token
                    break
            
            # Clean up
            del memory
            torch.cuda.empty_cache()
            
            return output_tokens, confidence

# Function to calculate token-level accuracy
def calculate_token_accuracy(predictions, targets, ignore_index=0):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    mask = targets != ignore_index
    correct = (predictions == targets)[mask].sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0

# Function to calculate sequence-level accuracy (ExpRate)
def calculate_sequence_accuracy(predictions, targets):
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        # Find where the target sequence ends (EOS token index is 2)
        eos_indices = np.where(target == 2)[0]
        if len(eos_indices) > 0:
            target_end = eos_indices[0]
        else:
            # If no EOS, find the last non-padding token
            non_pad_indices = np.where(target != 0)[0]
            if len(non_pad_indices) > 0:
                target_end = non_pad_indices[-1] + 1
            else:
                target_end = 1  # Only SOS token
        
        # Target sequence (remove SOS token and include up to EOS or end)
        target_seq = target[1:target_end]
        
        # Find where the prediction sequence ends
        eos_indices = np.where(pred == 2)[0]
        if len(eos_indices) > 0:
            pred_end = eos_indices[0]
        else:
            pred_end = len(pred)
        
        # Prediction sequence (remove SOS token and include up to EOS or end)
        pred_seq = pred[1:pred_end]
        
        # Check if sequences match
        if len(pred_seq) == len(target_seq) and np.array_equal(pred_seq, target_seq):
            correct += 1
        
        total += 1
    
    return correct / total if total > 0 else 0

# Function to display examples during training
def display_examples(model, test_loader, idx2token, device, num_examples=5):
    model.eval()
    examples = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['target']
            raw_targets = batch['raw_target']
            
            predictions, _ = model(images)
            
            for i in range(min(num_examples, len(images))):
                # Decode target
                target_tokens = []
                for idx in targets[i].numpy():
                    if idx == 0:  # <PAD>
                        continue
                    if idx == 2:  # <EOS>
                        break
                    if idx == 1:  # <SOS>
                        continue
                    target_tokens.append(idx2token[idx])
                
                # Decode prediction
                pred_tokens = []
                for idx in predictions[i].cpu().numpy():
                    if idx == 0:  # <PAD>
                        continue
                    if idx == 2:  # <EOS>
                        break
                    if idx == 1:  # <SOS>
                        continue
                    pred_tokens.append(idx2token[idx])
                
                examples.append({
                    'image': images[i].cpu(),
                    'target': ' '.join(target_tokens),
                    'prediction': ' '.join(pred_tokens),
                    'raw_target': raw_targets[i]
                })
            
            if len(examples) >= num_examples:
                break
    
    # Display examples
    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 4 * num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i, example in enumerate(examples):
        # Convert tensor to image
        img = example['image'].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        
        axes[i].imshow(img)
        axes[i].set_title(f"Target: {example['raw_target']}\nPrediction: {example['prediction']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('acatha_examples.png')
    plt.show()
    
    return examples

# FIXED training function with proper data handling and float conversion
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, gradient_accumulation_steps=4):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs, _ = model(images, targets)
                
                # Ensure outputs are float type
                outputs = outputs.to(torch.float32)
                
                if outputs.dim() == 3:
                    outputs = outputs[:, :targets.size(1)-1, :]
                    outputs_flat = outputs.reshape(-1, outputs.size(-1))
                    targets_flat = targets[:, 1:].reshape(-1)
                else:
                    print(f"WARNING: Unexpected output shape in training: {outputs.shape}")
                    continue

                loss = criterion(outputs_flat, targets_flat) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps

            with torch.no_grad():
                _, predicted = outputs_flat.max(1)
                mask = targets_flat != 0
                train_correct += (predicted == targets_flat)[mask].sum().item()
                train_total += mask.sum().item()

            del images, targets, outputs, outputs_flat, targets_flat
            if 'predicted' in locals():
                del predicted
            if 'mask' in locals():
                del mask
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = batch['image'].to(device)
                targets = batch['target'].to(device)

                # Force training mode to get logits instead of token indices
                model.train()
                outputs, _ = model(images, targets)
                model.eval()
                
                # Ensure outputs are float type
                outputs = outputs.to(torch.float32)
                
                if outputs.dim() == 3:
                    outputs = outputs[:, :targets.size(1)-1, :]
                    outputs_flat = outputs.reshape(-1, outputs.size(-1))
                    targets_flat = targets[:, 1:].reshape(-1)
                else:
                    print(f"WARNING: Unexpected output shape in validation: {outputs.shape}")
                    continue

                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()

                _, predicted = outputs_flat.max(1)
                mask = targets_flat != 0
                val_correct += (predicted == targets_flat)[mask].sum().item()
                val_total += mask.sum().item()

                # Get sequence predictions
                predictions, _ = model(images)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['target'].cpu().numpy())

                del images, targets, outputs, outputs_flat, targets_flat, predicted, mask, predictions
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Calculate sequence accuracy
        seq_acc = calculate_sequence_accuracy(all_predictions, all_targets)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Seq Acc: {seq_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_acatha_model.pth')
            print("Saved best model!")

        gc.collect()
        torch.cuda.empty_cache()

    return train_losses, val_losses, train_accuracies, val_accuracies

# FIXED evaluation function with proper data handling
def evaluate(model, test_loader, device, idx2token):
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Get token-level predictions for accuracy calculation
            # We need to run in training mode to get logits instead of token indices
            model.train()  # Temporarily set to train mode
            outputs, _ = model(images, targets)
            model.eval()  # Set back to eval mode
            
            # Ensure outputs are float type
            outputs = outputs.to(torch.float32)
            
            if outputs.dim() == 3:
                # Process the outputs for token-level accuracy
                outputs = outputs[:, :targets.size(1)-1, :]
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets[:, 1:].reshape(-1)
                
                _, predicted = outputs_flat.max(1)
                mask = targets_flat != 0  # Ignore <PAD> tokens
                test_correct += (predicted == targets_flat)[mask].sum().item()
                test_total += mask.sum().item()
            else:
                print(f"WARNING: Unexpected output shape in testing: {outputs.shape}")
                continue
            
            # Get sequence-level predictions for ExpRate calculation
            # This runs in eval mode to get token indices
            seq_outputs, _ = model(images)
            all_predictions.extend(seq_outputs.cpu().numpy())
            all_targets.extend(batch['target'].cpu().numpy())
            
            # Clean up memory
            del images, targets, outputs, outputs_flat, targets_flat, predicted, mask, seq_outputs
            torch.cuda.empty_cache()
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    seq_acc = calculate_sequence_accuracy(all_predictions, all_targets)
    
    return test_acc, seq_acc

# Plot training statistics
def plot_statistics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('acatha_training_stats.png')
    plt.show()

# Main function with memory optimizations and robust type handling
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set memory-efficient hyperparameters
    batch_size = 8  # Reduced from 16
    hidden_dim = 128  # Reduced from 256
    num_epochs = 20  # Reduced from 30
    learning_rate = 0.001
    gradient_accumulation_steps = 4  # Accumulate gradients for effective batch size of 32
    
    # Data paths
    root_dir = '/content/QA-HMEs'
    train_img_dir = os.path.join(root_dir, 'new_train_images')
    train_label_file = os.path.join(root_dir, 'new_train.txt')
    val_img_dir = train_img_dir  # For this example, we'll use a subset of training data for validation
    val_label_file = train_label_file
    test_img_dir = os.path.join(root_dir, 'new_test_images')
    test_label_file = os.path.join(root_dir, 'new_test.txt')
    
    # Data transforms with reduced image size
    transform = transforms.Compose([
        transforms.Resize((64, 256)),  # Reduced from (128, 512)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
    # Create datasets
    train_dataset = HMEDataset(train_img_dir, train_label_file, transform=transform)
    
    # For this example, we'll create a small validation set from the training data
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    test_dataset = HMEDataset(test_img_dir, test_label_file, transform=transform)
    test_dataset.token2idx = train_dataset.dataset.token2idx
    test_dataset.idx2token = train_dataset.dataset.idx2token
    
    # Create dataloaders with fewer workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Reduced workers
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # Reduced workers
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # Reduced workers
    
    # Vocabulary size
    vocab_size = len(train_dataset.dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create memory-efficient model
    model = ACATHA(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
    # Loss function (ignore padding index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Clear cache before training
    gc.collect()
    torch.cuda.empty_cache()
    
    # Train model with memory optimizations
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device, gradient_accumulation_steps
    )
    
     # Plot statistics
    plot_statistics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_acatha_model.pth'))
    test_acc, exp_rate = evaluate(model, test_loader, device, train_dataset.dataset.idx2token)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"ExpRate (Sequence Accuracy): {exp_rate:.4f}")
    print(f"Training Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()