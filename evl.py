def calculate_sequence_accuracy(predictions, targets):
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        
        eos_indices = np.where(target == 2)[0]
        if len(eos_indices) > 0:
            target_end = eos_indices[0]
        else:
            
            non_pad_indices = np.where(target != 0)[0]
            if len(non_pad_indices) > 0:
                target_end = non_pad_indices[-1] + 1
            else:
                target_end = 1  # Only SOS token
        
        
        target_seq = target[1:target_end]
        
        
        eos_indices = np.where(pred == 2)[0]
        if len(eos_indices) > 0:
            pred_end = eos_indices[0]
        else:
            pred_end = len(pred)
        
        
        pred_seq = pred[1:pred_end]
        
        
        if len(pred_seq) == len(target_seq) and np.array_equal(pred_seq, target_seq):
            correct += 1
        
        total += 1
    
    return correct / total if total > 0 else 0


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
                    if idx == 0:  
                        continue
                    if idx == 2:  
                        break
                    if idx == 1:  
                        continue
                    target_tokens.append(idx2token[idx])
                
                # Decode prediction
                pred_tokens = []
                for idx in predictions[i].cpu().numpy():
                    if idx == 0:  
                        continue
                    if idx == 2:  
                        break
                    if idx == 1: 
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
    
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 4 * num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i, example in enumerate(examples):
        
        img = example['image'].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        
        axes[i].imshow(img)
        axes[i].set_title(f"Target: {example['raw_target']}\nPrediction: {example['prediction']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('acatha_examples.png')
    plt.show()
    
    return examples


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

                
                model.train()
                outputs, _ = model(images, targets)
                model.eval()
                
                
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

                predictions, _ = model(images)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['target'].cpu().numpy())

                del images, targets, outputs, outputs_flat, targets_flat, predicted, mask, predictions
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

       
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
            
            
         
            model.train()  # Temporarily set to train mode
            outputs, _ = model(images, targets)
            model.eval()  # Set back to eval mode
            
            
            outputs = outputs.to(torch.float32)
            
            if outputs.dim() == 3:
            
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
            
            
         
            seq_outputs, _ = model(images)
            all_predictions.extend(seq_outputs.cpu().numpy())
            all_targets.extend(batch['target'].cpu().numpy())
            
           
            del images, targets, outputs, outputs_flat, targets_flat, predicted, mask, seq_outputs
            torch.cuda.empty_cache()
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    seq_acc = calculate_sequence_accuracy(all_predictions, all_targets)
    
    return test_acc, seq_acc


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

def main():
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    batch_size = 8  # Reduced from 16
    hidden_dim = 128  # Reduced from 256
    num_epochs = 20  # Reduced from 30
    learning_rate = 0.001
    gradient_accumulation_steps = 4  # Accumulate gradients for effective batch size of 32
    
  
    root_dir = '/content/QA-HMEs'
    train_img_dir = os.path.join(root_dir, 'new_train_images')
    train_label_file = os.path.join(root_dir, 'new_train.txt')
    val_img_dir = train_img_dir  # For this example, we'll use a subset of training data for validation
    val_label_file = train_label_file
    test_img_dir = os.path.join(root_dir, 'new_test_images')
    test_label_file = os.path.join(root_dir, 'new_test.txt')
    
   
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
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Reduced workers
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # Reduced workers
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # Reduced workers
    
   
    vocab_size = len(train_dataset.dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
   
    model = ACATHA(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
   
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    gc.collect()
    torch.cuda.empty_cache()
    
    
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device, gradient_accumulation_steps
    )
    
    
    plot_statistics(train_losses, val_losses, train_accuracies, val_accuracies)
    
   
    model.load_state_dict(torch.load('best_acatha_model.pth'))
    test_acc, exp_rate = evaluate(model, test_loader, device, train_dataset.dataset.idx2token)
    

    print("\nFinal Results:")
    print(f"Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"ExpRate (Sequence Accuracy): {exp_rate:.4f}")
    print(f"Training Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()