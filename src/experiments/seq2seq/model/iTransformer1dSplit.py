# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Embed import DataEmbedding_inverted
# import numpy as np

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         self.use_norm = configs.use_norm
#         self.d_model = configs.d_model
#         self.enc_in = configs.enc_in

#         # CNN parameters
#         self.kernel_size = 5
#         self.stride = 2

#         # 1D CNN for processing first 80% of input
#         self.cnn = nn.Conv1d(in_channels=self.enc_in,
#                              out_channels=self.d_model,
#                              kernel_size=self.kernel_size,
#                              stride=self.stride,
#                              padding=0)

#         # Linear layer to project raw input to d_model dimension
#         self.cnn_proj = nn.Linear(self.d_model, self.enc_in)

#         combined_seq_len = (int(
#             0.8*configs.seq_len)-self.kernel_size)//self.stride+1+configs.seq_len-int(0.8*configs.seq_len)
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(combined_seq_len, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         self.class_strategy = configs.class_strategy

#         # Encoder-only architecture
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             # Normalization from Non-stationary Transformer
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev

#         B, L, N = x_enc.shape # B L N

#         # Split input: 80% for CNN, 20% raw
#         split_point = int(0.8 * L)
#         x_cnn = x_enc[:, :split_point, :]
#         x_raw = x_enc[:, split_point:, :]

#         # Process 80% with CNN
#         x_cnn = x_cnn.permute(0, 2, 1)  # [B, N, L] for conv1d
#         x_cnn = self.cnn(x_cnn)
#         x_cnn = x_cnn.permute(0, 2, 1)  # [B, L, N]

#         # Project CNN output to d_model dimension
#         x_cnn = self.cnn_proj(x_cnn)

#         # print(f"x_cnn.shape :{x_cnn.shape} x_raw.shape:{x_raw.shape}")
#         # Concatenate CNN output with raw 20%
#         x_combined = torch.cat([x_cnn, x_raw], dim=1)

#         # Interpolate x_mark_enc to match x_combined length
#         x_mark_enc_interp = self.mark_enc_interpolation(x_combined, x_mark_enc)

#         # Embedding
#         # print(f"x_combined: {x_combined.shape} and x_mark_enc_interp:{x_mark_enc_interp.shape}")
#         enc_out = self.enc_embedding(x_combined, x_mark_enc_interp)

#         # Encoder
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         # Projection
#         dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

#         if self.use_norm:
#             # De-Normalization from Non-stationary Transformer
#             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

#         return dec_out

#     def mark_enc_interpolation(self, x_combined, x_mark_enc):
#         batch_size, target_length, _ = x_combined.shape
#         _, _, num_features = x_mark_enc.shape

#         x_mark_enc = x_mark_enc.permute(0, 2, 1)
#         x_mark_enc_interp = F.interpolate(
#             x_mark_enc, size=target_length, mode='linear', align_corners=False)
#         x_mark_enc_interp = x_mark_enc_interp.permute(0, 2, 1)

#         return x_mark_enc_interp

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#         return dec_out[:, -self.pred_len:, :]  # [B, L, D]

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in

        # CNN parameters
        self.kernel_size_1 = 5
        self.stride_1 = 2
        self.kernel_size_2 = 3
        self.stride_2 = 1

        self.split_factor=0.8
        combined_seq_len = self.cnn_seq_calc(self.cnn_seq_calc(int(
            self.split_factor*configs.seq_len), self.kernel_size_1, self.stride_1), self.kernel_size_2, self.stride_2)+configs.seq_len-int(self.split_factor*configs.seq_len)

        # First 1D CNN layer
        self.cnn1 = nn.Conv1d(in_channels=self.enc_in,
                              out_channels=self.d_model,
                              kernel_size=self.kernel_size_1,
                              stride=self.stride_1,
                              padding=0)

        # Second 1D CNN layer
        self.cnn2 = nn.Conv1d(in_channels=self.d_model,
                              out_channels=self.d_model,
                              kernel_size=self.kernel_size_2,
                              stride=self.stride_2,
                              padding=0)

        self.dropout1 = nn.Dropout(p=0.2)
        # Linear layer to project raw input to d_model dimension
        self.cnn_proj = nn.Linear(self.d_model, self.enc_in)

        # combined_seq_len = (int(
        #     0.8*configs.seq_len) - self.kernel_size) // self.stride + 1 + configs.seq_len - int(0.8*configs.seq_len)
        print(f"combined_seq_len: {combined_seq_len}")

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(combined_seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(
            configs.d_model, configs.pred_len, bias=True)

    def cnn_seq_calc(self, seq_len, kernel_size, stride):
        return (seq_len - kernel_size) // stride + 1

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape

        split_point = int(self.split_factor * L)
        x_cnn = x_enc[:, :split_point, :]
        x_raw = x_enc[:, split_point:, :]

        # Process 80% with first CNN layer
        x_cnn = x_cnn.permute(0, 2, 1)  # [B, N, L] for conv1d
        x_cnn = self.cnn1(x_cnn)

        x_cnn = F.relu(x_cnn)

        # Process with second CNN layer
        x_cnn = self.cnn2(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 1)  # [B, L, N]

        x_cnn = F.relu(x_cnn)
        # Project CNN output to d_model dimension
        x_cnn=self.cnn_proj(x_cnn)
        x_cnn = F.relu(x_cnn)
        # Concatenate CNN output with raw 20%
        x_combined = torch.cat([x_cnn, x_raw], dim=1)

        # Interpolate x_mark_enc to match x_combined length
        # x_mark_enc_interp = self.mark_enc_interpolation(x_combined, x_mark_enc)

        # Embedding
        # print(f"x_combined: {x_combined.shape} and x_mark_enc_interp:{x_mark_enc_interp.shape}")
        enc_out = self.enc_embedding(x_combined, None)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def mark_enc_interpolation(self, x_combined, x_mark_enc):
        batch_size, target_length, _ = x_combined.shape
        _, _, num_features = x_mark_enc.shape

        x_mark_enc = x_mark_enc.permute(0, 2, 1)
        x_mark_enc_interp = F.interpolate(
            x_mark_enc, size=target_length, mode='linear', align_corners=False)
        x_mark_enc_interp = x_mark_enc_interp.permute(0, 2, 1)

        return x_mark_enc_interp

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
