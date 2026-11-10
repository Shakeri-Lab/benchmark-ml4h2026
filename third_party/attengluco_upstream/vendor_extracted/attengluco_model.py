"""
Reference AttenGluco model extracted from AttenGluco_CohortWise_Insulin.py.
This preserves the original implementation to allow importable use.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
        super().__init__()
        self.attn_AB = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.attn_AC = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, d_model),
        )

    def forward(self, A, B, C, need_weights=False):
        """
        A, B, C => (seq_len, batch_size, d_model)
        Returns:
          out => (seq_len, batch_size, d_model)
          wts_AB, wts_AC (if need_weights=True), else None
        """
        # A->B
        attn_out_AB, attn_wts_AB = self.attn_AB(A, B, B, need_weights=need_weights)
        A_out_AB = self.ln1(A + self.dropout(attn_out_AB))

        # A->C
        attn_out_AC, attn_wts_AC = self.attn_AC(A, C, C, need_weights=need_weights)
        A_out_AC = self.ln1(A + self.dropout(attn_out_AC))

        # Suppose we average the two outputs (example design)
        A_merged = 0.5 * (A_out_AB + A_out_AC)

        # Feed-forward
        ff_out = self.ffn(A_merged)
        out = self.ln2(A_merged + self.dropout(ff_out))

        if need_weights:
            return out, attn_wts_AB, attn_wts_AC
        return out, None, None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class MultiScaleTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
        super(MultiScaleTransformerBlock, self).__init__()

        self.attn_high = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.attn_low_2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.attn_low_4 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.attn_low_8 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

        self.pool_2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool_4 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.pool_8 = nn.AvgPool1d(kernel_size=8, stride=8)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, d_model),
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape => (seq_len, batch, d_model)
        attn_high_out, _ = self.attn_high(x, x, x)
        high_scale = self.layernorm1(x + self.dropout(attn_high_out))

        x_t = high_scale.permute(1, 2, 0)

        # factor=2
        x_low_2_t = self.pool_2(x_t)
        x_low_2 = x_low_2_t.permute(2, 0, 1)
        attn_low_2_out, _ = self.attn_low_2(x_low_2, x_low_2, x_low_2)
        up_t_2 = attn_low_2_out.permute(1, 2, 0)
        up_t_interp_2 = F.interpolate(up_t_2, scale_factor=2, mode="nearest")
        up_out_2 = up_t_interp_2.permute(2, 0, 1)

        # factor=4
        x_low_4_t = self.pool_4(x_t)
        x_low_4 = x_low_4_t.permute(2, 0, 1)
        attn_low_4_out, _ = self.attn_low_4(x_low_4, x_low_4, x_low_4)
        up_t_4 = attn_low_4_out.permute(1, 2, 0)
        up_t_interp_4 = F.interpolate(up_t_4, scale_factor=4, mode="nearest")
        up_out_4 = up_t_interp_4.permute(2, 0, 1)

        fused = high_scale + self.dropout(up_out_2) + self.dropout(up_out_4)
        ffn_out = self.ffn(fused)
        out = self.layernorm2(fused + self.dropout(ffn_out))
        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_time_steps, n_features, d_model, n_heads, ff_units, prediction_horizon):
        super(TimeSeriesTransformer, self).__init__()

        # Embeddings for each channel (A=Glucose, B=Walking Steps, C=Walking Interval).
        self.embed_A = nn.Linear(1, d_model)
        self.embed_B = nn.Linear(1, d_model)
        self.embed_C = nn.Linear(1, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=n_time_steps)

        self.cross_attn_block = CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            ff_units=ff_units,
            dropout=0.1,
        )

        # Apply multi-scale blocks after the cross-attention
        self.blocks = nn.ModuleList(
            [
                MultiScaleTransformerBlock(d_model, n_heads, ff_units, dropout=0.1),
                MultiScaleTransformerBlock(d_model, n_heads, ff_units, dropout=0.1),
                MultiScaleTransformerBlock(d_model, n_heads, ff_units, dropout=0.1),
            ]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flat_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.fc_out = nn.Linear(d_model, prediction_horizon)

    def forward(self, x, return_attn=False):
        """
        x shape: (batch_size, seq_len, 3) => channels: [0]=Glucose, [1]=WalkingSteps, [2]=WalkingInterval
        We'll treat Glucose as A, Steps as B, Intervals as C (just an example).
        """
        # 1) Split
        A = x[..., 0:1]  # (batch, seq_len, 1) -> Glucose
        B = x[..., 1:2]  # (batch, seq_len, 1) -> WalkingSteps
        C = x[..., 2:3]  # (batch, seq_len, 1) -> WalkingInterval

        # 2) Embed each
        A_e = self.embed_A(A)  # => (batch, seq_len, d_model)
        B_e = self.embed_B(B)
        C_e = self.embed_C(C)

        # 3) Positional encoding
        A_e = self.pos_encoding(A_e)  # => (batch, seq_len, d_model)
        B_e = self.pos_encoding(B_e)
        C_e = self.pos_encoding(C_e)

        # 4) Permute => (seq_len, batch, d_model) for attention
        A_e = A_e.permute(1, 0, 2)
        B_e = B_e.permute(1, 0, 2)
        C_e = C_e.permute(1, 0, 2)

        # 5) CROSS-ATTENTION: A queries B, then A queries C
        A_out, wts_AB, wts_AC = self.cross_attn_block(A_e, B_e, C_e, need_weights=return_attn)
        # A_out => (seq_len, batch, d_model)

        # 6) MULTI-SCALE BLOCKS
        # Now pass the updated A through your multi-scale blocks
        x = A_out
        for block in self.blocks:
            x = block(x)  # => (seq_len, batch, d_model)

        # 7) Flatten or pool to final predictions
        # => (batch, d_model, seq_len)
        x = x.permute(1, 2, 0)

        # Flatten approach
        batch_size, d_ch, seq_len = x.shape
        x = x.reshape(batch_size, d_ch * seq_len)

        x = self.flat_fc(x)
        out = self.fc_out(x)  # => (batch, prediction_horizon)
        if return_attn:
            return out, wts_AB, wts_AC
        return out
