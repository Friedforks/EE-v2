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
        self.kernel_size = (15,5)
        self.stride = (5, 1)
        self.padding = (7, 2)

        # 2D CNN for preprocessing
        self.cnn = nn.Conv2d(in_channels=1,
                             out_channels=self.d_model,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding)

        # Linear layer to project CNN output to enc_in dimension
        self.cnn_proj = nn.Linear(self.d_model*self.enc_in, self.enc_in)

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

    def preprocess_with_cnn2d(self, x):
        batch_size, seq_len, features = x.shape
        
        # Reshape for 2D CNN
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, features]
        
        # Apply 2D CNN
        x = self.cnn(x)  # [batch_size, d_model, new_seq_len, features]
        
        # Reshape back
        new_seq_len = x.shape[2]
        x = x.permute(0, 2, 3, 1)  # [batch_size, new_seq_len, features, d_model]
        x = x.reshape(batch_size, new_seq_len, -1)  # [batch_size, new_seq_len, features * d_model]
        
        # Project back to original feature dimension
        # print(f"Before projection: {x.shape}. dmodel: {self.d_model}, enc_in: {self.enc_in}")
        x = self.cnn_proj(x)  # [batch_size, new_seq_len, enc_in]
        
        return x

    def mark_enc_interpolation(self, x_combined, x_mark_enc):
        batch_size, target_length, _ = x_combined.shape
        _, _, num_features = x_mark_enc.shape

        x_mark_enc = x_mark_enc.permute(0, 2, 1)
        x_mark_enc_interp = F.interpolate(
            x_mark_enc, size=target_length, mode='linear', align_corners=False)
        x_mark_enc_interp = x_mark_enc_interp.permute(0, 2, 1)

        return x_mark_enc_interp

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Preprocess with 2D CNN
        x_processed = self.preprocess_with_cnn2d(x_enc)

        # Interpolate mark_enc to match new sequence length
        x_mark_enc_interp = self.mark_enc_interpolation(x_processed, x_mark_enc)

        # Embedding
        enc_out = self.enc_embedding(x_processed, x_mark_enc_interp)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]