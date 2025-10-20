# ===== Range Mixture Swin Transformer — Improved (torchvision‑based) =====
# This version swaps timm for torchvision's official Swin‑3D implementation and
# plugs the user‑supplied build_range_mixture_distribution() + loss utilities.

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily

# torchvision >=0.19 (May 2025) contains Swin3D‑B
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

from prob import build_range_mixture_distribution

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def build_sinusoidal_embedding(length: int, dim: int) -> torch.Tensor:
    """
    Give every time-step t in the history a unique, deterministic vector 
    so the attention layer can tell "how far back" a memory token lies.
    Sinusoidal encodes relative distances linearly (the dot-product of two encodings depends only on ||t1-t2|| )
    """
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def adapt_conv3d_in_channels(conv: nn.Conv3d, new_in: int):
    old_w = conv.weight.data
    extra = new_in - conv.in_channels
    if extra < 0:
        conv.weight.data = old_w[:, :new_in]
    else:
        mean = old_w.mean(1, keepdim=True)
        conv.weight.data = torch.cat([old_w, mean.repeat(1, extra, 1, 1, 1)], dim=1)
    conv.in_channels = new_in


# --------------------------------------------------
# 1. Temporal Self‑Attention (with residual & FFN)
# --------------------------------------------------
class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, max_T: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # nn.MultiheadAttention expects input of shape [batch, seq_len, embed_dim]. 
        # Creating (B x Hc x Wc) independent mini-patches each one a time-series of length T with embedding size C.
        # Attention is computed within each of these rows (over time), and never between different rows.
        
        self.norm2 = nn.LayerNorm(dim)  # Follows pre‑norm Transformer style Attention -> better gradient flow in deep nets
        # Add non‑linear channel mixing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # learnable scalar gates (residuals) gamma (initialised 0) so the net can start as the identity and gradually turn on attention 
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.register_buffer("pos_emb", build_sinusoidal_embedding(max_T, dim), persistent=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, T, Hc, Wc = feat.shape
        x = feat.permute(0, 3, 4, 2, 1).reshape(B * Hc * Wc, T, C)
        x = x + self.pos_emb[:T].unsqueeze(0)   # inject absolute time info see build_sinusoidal_embedding description
        x = x + self.gamma1 * self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        x = x.reshape(B, Hc, Wc, T, C).permute(0, 4, 3, 1, 2).contiguous()
        return x


# --------------------------------------------------
# 2. Transformer Forecast Head (μ, σ, α)
#     — leave all squashing / softmax for build_range_mixture_distribution
# --------------------------------------------------
class TemporalForecastHead(nn.Module):
    """Emits un-normalised raw parameters; downstream utility applies activations."""

    def __init__(self, C_feat: int, F: int, mdn_K: int, d_model: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.F, self.K = F, mdn_K
        self.in_proj = nn.Linear(C_feat, d_model)
        self.queries = nn.Parameter(torch.randn(F, d_model))
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers)
        self.out_proj = nn.Linear(d_model, 3 * mdn_K)
        
        # better init
        with torch.no_grad():
            # mu (first  K): centre around 10 m with small spread
            nn.init.normal_(self.out_proj.weight[:mdn_K],  mean=10.0 / mdn_K, std=0.02)
            nn.init.constant_(self.out_proj.bias [:mdn_K], 10.0 / mdn_K)

            # log sigma (next K): start at log(1.0)
            nn.init.constant_(self.out_proj.bias[mdn_K:2*mdn_K],  0.0)
            nn.init.normal_ (self.out_proj.weight[mdn_K:2*mdn_K], 0.0, 0.01)

            # logits (last K): small random so α ≈ uniform
            nn.init.normal_(self.out_proj.weight[2*mdn_K:], 0.0, 0.01)
            nn.init.constant_(self.out_proj.bias [2*mdn_K:], 0.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, T, Hc, Wc = feat.shape
        N = B * Hc * Wc
        mem = feat.permute(0, 3, 4, 2, 1).reshape(N, T, C)  # Turns the 5‑D tensor into N independent "mini‑batches", one per ray.
        mem = self.in_proj(mem)
        # memory encodes “where” (specific ray) and align feature length with decoder dimension
        
        q = self.queries.unsqueeze(0).expand(N, -1, -1)     
        # Query table encodes "when" (future step)
        
        y = self.dec(tgt=q, memory=mem) # [N, F, d_model]
        # memory (key/value=where/what happend before) each ray sees its own past feature vectors.
        # Each q[i,k] is updated by cross‑attention with that ray's unique memory tokens (its own T‑long feature history)
        # Given this ray’s past, what mixture of Gaussians best predicts the future range at time+k
        
        raw = self.out_proj(y)          # [N, F, 3K]
        # For each ray & horizon, output K triplets (coarse mu, sigma, alpha embeddings).
        return raw.view(B, Hc, Wc, self.F, 3 * self.K).permute(0, 3, 1, 2, 4)

# --------------------------------------------------
# 3. Alias‑free Upsampling Blocks
# --------------------------------------------------
class UpsampleBlock(nn.Module):
    """
    Nearest-neighbhour resize + 1x1 conv + 3x3 conv -> to blow up spatial resolution from (Hc,Wc) to (H,W)
    Mitigates checkerboard artefacts and erases vertical stripes.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.block(x)


# --------------------------------------------------
# 4. Main model
# --------------------------------------------------
class RangeMixtureSwinTransformerModel(nn.Module):
    def __init__(self, cfg, use_skip: bool = True):
        super().__init__()
        mp = cfg["model_params"]
        self.F, self.K = mp["forecast_horizon"], mp["mdn_num_gaussians"]
        self.H, self.W, self.C = mp["grid_height"], mp["grid_width"], mp["grid_channels"]
        init_temp = cfg.get("train_params", {}).get("alpha_temperature_init", 0.25)
        self.alpha_temp = nn.Parameter(torch.tensor(init_temp))
        self.with_attention = cfg["model_params"].get("with_attention", False)

        self.backbone = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)
        if self.C != self.backbone.patch_embed.proj.in_channels:
            adapt_conv3d_in_channels(self.backbone.patch_embed.proj, self.C)
        self.backbone.head = nn.Identity(); self.backbone.avgpool = nn.Identity()

        # determine coarse dims
        with torch.no_grad():
            dummy = torch.zeros(1, self.C, self.F, self.H, self.W)
            feat = self._forward_backbone(dummy)
            self.C_feat = feat.shape[-1]
            self.Hc, self.Wc = feat.shape[2:4]

        self.temporal_attn = TemporalSelfAttention(self.C_feat)
        self.forecast_head = TemporalForecastHead(self.C_feat, self.F, self.K)

        # pool time to F frames (keeps spatial coarse dims)
        self.temporal_pool = nn.AdaptiveAvgPool3d((self.F, None, None))
        
        scale = self.H // self.Hc
        layers: List[nn.Module] = []
        if self.with_attention:
            in_ch = 3 * self.K
        else:
            in_ch = self.C_feat
        for _ in range(int(math.log2(scale))):
            out_ch = max(in_ch // 2, 32)
            layers.append(UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.upsampler = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_ch, 3 * self.K, kernel_size=1)
        

    def _forward_backbone(self, x):
        # x.shape -> [B, C, T_in, H, W]

        x = self.backbone.patch_embed(x)
        # patch_embed is Conv3d(4 -> 128, kernel=(2,4,4), stride=(2,4,4)) and LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        # after Conv3d: time halves, H quarters, W quarters: [B, C=128, T_in/2, H/4, W/4]
        # However, PatchEmbed3d then effectively permute(0, 2, 3, 4, 1) -> [B, T_in/2, H/4, W/4, 128] to apply LayerNorm on C-dim
        # x.shape -> [B, T_in/2, H/4, W/4, C=128]
        
        x = self.backbone.pos_drop(x)   
        # Dropout-Layer: if p>0, this would randomly zero whole elements in that channel‐last tensor
        
        x = self.backbone.features(x)
        # features is a huge 3-stage Swin stack with intermediate "patch-merging":
        # PatchMerging rearranges the spatial 2x2 (H and W) neighborhoods into four patches and concatenates along channel axis (so C will be 4*C)
        # LayerNorm normalizes over that last-dim and projects the channel down with Linear layers:
        # PatchMerging applied at stage 1, 3, 5 (total num stages: 7)
        # x.shape -> [B, T_in/2, H/ (4*2**3), W/ (4*2**3), C=1024] = [B, T_in/2, H/ 32, W/ 32, C=1024]
        
        x = self.backbone.norm(x)
        # Backbone result: coarse (T_in//2)-frame, (H/32) x (W/32) feature volume with 1024 channels
        # x.shape: [B, Tc, Hc, Wc, C_feat] = [B, T_in/2, H/ 32, W/ 32, C_feat]
        return x

    def forward(self, hist_xyz):
        B, T_in, _, _, _ = hist_xyz.shape
        x = hist_xyz.permute(0, 2, 1, 3, 4)
        
        #@@@- Swin3D BACKBONE -@@@#
        feat = self._forward_backbone(x)
            # x.shape: [B, Tc, Hc, Wc, C_feat] = [B, T_in/2, H/ 32, W/ 32, C_feat]
        feat = feat.permute(0, 4, 1, 2, 3) 
            # x.shape: [B, C_feat, Tc, Hc, Wc]
        
        if self.with_attention:
            #@@@- Temporal Self-Attention -@@@#
            feat = self.temporal_attn(feat)
            
            #@@@- Transformer Cross-Attention Encoder-Decoder -@@@#
            raw_params = self.forecast_head(feat)
            
            Bf, Ff, Hc, Wc, Ch = raw_params.shape
            y = raw_params.permute(0, 1, 4, 2, 3).reshape(B * Ff, Ch, Hc, Wc)
            
            #@@@- Upsampling (blows up spatial resolution from (Hc,Wc) to (H,W)) -@@@#
            y = self.upsampler(y)
            y = self.final_conv(y)  # feature channel from 32 to (3 * K)
            y = y.view(B, Ff, self.H, self.W, 3 * self.K)
        else:
            # pool temporal dimension down to exactly F frames
            feat = self.temporal_pool(feat)
            #   AdaptiveAvgPool3d((F, None, None)) squeezes the Tc frames -> F frames (output horizon).
            # feat.shape -> [B, C_feat, F, Hc, Wc]
            
            # now upsample each of the F feature frames independently
            # 1) move F into batch
            f = feat.permute(0,2,1,3,4)                     # [B, F, C_feat, Hc, Wc]
            f = f.reshape(B*self.F, self.C_feat, self.Hc, self.Wc)
            # 2) decode spatially
            up = self.upsampler(f)                          # [B*F, mid=32, H, W]
            out = self.final_conv(up)                       # [B*F, out_ch, H, W]
            # 3) restore [B, F, …]
            out = out.view(B, self.F, -1, self.H, self.W)   # [B, F, out_ch, H, W]
            y = out.permute(0,1,3,4,2).contiguous()       # [B, F, H, W, out_ch]
        
        return y


    def build_mixture(self, cfg, output):
        return build_range_mixture_distribution(cfg, output, self.alpha_temp)
