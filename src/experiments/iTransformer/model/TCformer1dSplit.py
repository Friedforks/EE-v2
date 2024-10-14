import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.kernel_size = 5
        self.stride = 2

        # 1D CNN for processing first 80% of input
        self.cnn = nn.Conv1d(in_channels=self.enc_in,
                             out_channels=self.d_model,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=0)

        # Linear layer to project raw input to d_model dimension
        self.cnn_proj = nn.Linear(self.d_model, self.enc_in)

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.d_model, configs.n_heads),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        self.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        self.d_model, configs.n_heads),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def mark_enc_interpolation(self,x_combined, x_mark_enc):
        # x_mark_enc shape: [batch_size, seq_len, features]
        # x_combined shape: [batch_size, new_seq_len, new_features]

        batch_size, target_length, _ = x_combined.shape
        _, _, num_features = x_mark_enc.shape

        # Reshape for interpolation
        # [batch_size, features, seq_len]
        x_mark_enc = x_mark_enc.permute(0, 2, 1)

        # Interpolate
        x_mark_enc_interp = F.interpolate(
            x_mark_enc, size=target_length, mode='linear', align_corners=False)

        # Reshape back
        x_mark_enc_interp = x_mark_enc_interp.permute(
            0, 2, 1)  # [batch_size, new_seq_len, features]

        return x_mark_enc_interp

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Split input: 80% for CNN, 20% raw
        split_point = int(0.8 * x_enc.shape[1])
        x_cnn = x_enc[:, :split_point, :]
        x_raw = x_enc[:, split_point:, :]

        # Process 80% with CNN
        x_cnn = x_cnn.permute(0, 2, 1)  # [B, D, L] for conv1d
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 1)  # [B, L, D]

        # Project raw 20% to d_model dimension
        # print(
        #     f"x_raw.shape:{x_raw.shape}. x_cnn.shape:{x_cnn.shape} While d_model:{self.d_model} and enc_in:{self.enc_in}")
        x_cnn = self.cnn_proj(x_cnn)

        # Concatenate CNN output with projected raw 20%
        x_combined = torch.cat([x_cnn, x_raw], dim=1)

        # calculate sequence length reduction, mentioned in parameter reduction section of the paper
        total_seq_len = (int(
            0.8*x_enc.shape[1])-self.kernel_size)//self.stride+1+x_enc.shape[1]-int(0.8*x_enc.shape[1])

        # print(f"x_combined.shape: {x_combined.shape}")
        # print(f"total_seq_len: {total_seq_len}")
        # # Embedding
        # print(
        #     f"x_enc.shape: {x_enc.shape}, x_mark_enc.shape: {x_mark_enc.shape}")

        x_mark_enc_interp = self.mark_enc_interpolation(x_combined, x_mark_enc)
        # print(
            # f"x_mark_enc_interp.shape: {x_mark_enc_interp.shape}")
        enc_out = self.enc_embedding(x_combined, x_mark_enc_interp)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
