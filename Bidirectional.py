class BidirectionalStructureAwareDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, nhead=4, num_layers=2, dropout=0.1):  # Reduced parameters
        super(BidirectionalStructureAwareDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        
        tgt_embedded = self.token_embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        
        output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask, 
                                     tgt_key_padding_mask=tgt_key_padding_mask)
        
       
        output = self.output_layer(output)
        
        
        output = output.to(torch.float32)
        
        return output
