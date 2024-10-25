import torch
from torch import nn

from .MultiHeadAttention import MultiHeadAttention
from .PositionWiseFeedForward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, drop_prob: float,kan: bool) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob,kan)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec: torch.Tensor, enc: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        _x = dec
        x = self.self_attention(dec, dec, dec, trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x+_x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(x, enc, enc, src_mask)

            x = self.dropout2(x)
            x = self.norm2(x+_x)

        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x+_x)
        return x


class Decoder(nn.Module):
    def __init__(self, dev_voc_size: int,
                 max_len: int,
                 d_model: int,
                 ffn_hidden: int,
                 n_head: int,
                 n_layers: int,
                 drop_prob: float,
                 kan: bool,
                 device: torch.device) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dev_voc_size)

    def forward(self, trg: torch.Tensor, enc_src: int, trg_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output
