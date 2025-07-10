import torch
from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # Split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # Shape of q, k, v: [batch_size, head, length, d_tensor]
        # Shape of mask: [batch_size, 1, 1, length]
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)

        # To-do: Visualize attention map
        return out

    def split(self, tensor: torch.Tensor):
        """
            Split tensor by number of heads
            [batch_size, length, d_model] -> [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        
        return tensor
    
    def concat(self, tensor: torch.Tensor):
        """
            Inverse function of self.split
            [batch_size, head, length, d_tensor] -> [batch_size, length, head * d_tensor]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor