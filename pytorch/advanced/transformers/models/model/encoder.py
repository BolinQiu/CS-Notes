from torch import nn
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])
    
    def forward(self, x, src_mask):
        # Shape of x: [batch_size, seq_len]
        x = self.emb(x) # Shape: [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x, src_mask)
        return x # Shape: [batch_size, seq_len, d_model]