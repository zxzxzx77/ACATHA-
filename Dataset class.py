class HMEDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None, max_len=150):
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        
        
        self.data = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    input_expr, img_file, target_expr = parts
                    self.data.append((img_file, target_expr))
        
        
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
        
        
        tokens = ["< SOS >"] + target_expr.split() + ["<EOS>"]
        target_indices = [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
        
        
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