from torch import nn


class TokenEmbedding(nn.Embedding):
    """
        nn.Embedding maps the given vocab to a dense vector representation, which
        is often used for word embedding in NLP
    """
    def __init__(self, vocab_size, d_model):
        """
        vocab_size: Size of the vocab (# of words in vocab)
        d_model: Dimension of embeddings
        padding_idx: 
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)