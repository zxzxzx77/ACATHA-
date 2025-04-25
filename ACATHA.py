class ACATHA(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, nhead=4, num_decoder_layers=2, dropout=0.1):  # Reduced parameters
        super(ACATHA, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        
        self.feature_extractor = HierarchicalFeatureExtractor(in_channels=3, feature_dim=hidden_dim)
        
        # Adaptive Context Integration
        self.context_integrator = AdaptiveContextIntegration(feature_dim=hidden_dim)
        
        
        self.feature_converter = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16))  #  (8, 32)
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
        
        features = self.feature_extractor(img)
        
        
        features, confidence = self.context_integrator(features)
        
        
        features = self.feature_converter(features)
        batch_size, channels, height, width = features.size()
        
        
        memory = features.view(batch_size, channels, -1).permute(2, 0, 1)
        
        
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
            
            
            del memory, tgt_mask, tgt_input
            torch.cuda.empty_cache()
            
            return output, confidence
        else:
            
            device = next(self.parameters()).device
            batch_size = img.size(0)
            max_len = 100  # Reduced from 150 for memory efficiency
            
            
            output_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for i in range(max_len - 1):
                tgt_mask = self._generate_square_subsequent_mask(output_tokens.size(1)).to(device)
                tgt_input = output_tokens.permute(1, 0)  # (seq_len, batch)
                
                decoder_output = self.decoder(tgt_input, memory, tgt_mask=tgt_mask)
                
               
                decoder_output = decoder_output.to(torch.float32)
                
                decoder_output = decoder_output.permute(1, 0, 2)  # (batch, seq_len, vocab_size)
                
               
                next_tokens = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_tokens], dim=1)
                
                del decoder_output, tgt_mask, tgt_input
                torch.cuda.empty_cache()
                
                
                if (next_tokens == 2).all():  # <EOS> token
                    break
            
            
            del memory
            torch.cuda.empty_cache()
            
            return output_tokens, confidence