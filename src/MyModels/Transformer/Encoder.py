import torch
from torch import nn
from .MultiHeadAttention import MultiHeadAttention
from .PositionWiseFeedForward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float,kan: bool) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob,kan)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.attention(x, x, x, mask)
        
        x = self.dropout1(x)
        x = self.norm1(x+_x)

        _x = x
        x = self.ffn(x) 

        x = self.dropout2(x)
        x = self.norm2(x+_x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int,
                 ffn_hidden: int,
                 n_head: int,
                 n_layers: int,
                 drop_prob: float,
                 kan:bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob,kan) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
