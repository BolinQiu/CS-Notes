import torch
from torch import nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size,
                 d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # Shape: [batch_size, 1, 1, src_len]
        return src_mask
    
    def make_trg_mask(self, trg):
        # Shape of trg: [batch_size, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3) # Shape: [batch_size, 1, trg_len, 1]
        trg_len = trg.shape[1]
        # torch.tril creates a lower triangular matrix
        # type(torch.ByteTensor) is used to create a mask of type Byte
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len).type(torch.ByteTensor).to(self.device))
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        # Shape of src: [batch_size, src_len]
        # Shape of trg: [batch_size, trg_len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output
    

"""
    There exists some problems in mask implementation?
"""