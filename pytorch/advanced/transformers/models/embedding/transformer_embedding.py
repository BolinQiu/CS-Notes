from torch import nn
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding



class TransformerEmbedding(nn.Module):
    """
        Token embedding + positional encoding (sinusoid)
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # Shape of x: [batch_size, seq_len]

        tok_emb = self.tok_emb(x) # Shape: [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x) # Shape: [batch_size, seq_len, d_model]
        return self.drop_out(tok_emb + pos_emb) # Shape: [batch_size, seq_len, d_model]