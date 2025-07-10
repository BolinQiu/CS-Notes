import math
from torch import nn
import torch


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None, e=1e-12):
        """
            Input is 4 dimension tensor: [batch_size, head, length, d_tensor]
        """
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        # Shape of q: [batch_size, head, length, d_tensor]
        # Shape of k_t: [batch_size, head, d_tensor, length]
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)
        # Shape of score: [batch_size, head, length, length]
        # Apply masking (opt)
        if mask is not None:
            # Shape of mask: [batch_size, 1, 1, length]
            score = score.masked_fill(mask == 0, -10000)
            # For masked_fill function, the shape of mask should be broadcastable to the shape of score

        score = self.softmax(score)
        v = torch.matmul(score, v)
        return v, score