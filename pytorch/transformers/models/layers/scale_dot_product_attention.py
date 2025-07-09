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
        score = torch.mul(q, k_t) / math.sqrt(d_tensor)

        # Apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = torch.mul(score, v)
        return v, score