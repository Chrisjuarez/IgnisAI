"""
ConvLSTM-UNet v2.

Why this file was rewritten:
  - The previous "UNetDecoder" had no skip connections — it was just a 2x-2x
    upsampler fed by an encoder that never downsampled. Calling it a U-Net was
    generous; it was a bottleneck-to-blur pipe. Predictions came out smooth and
    low-contrast because high-frequency spatial detail was gone by the time the
    decoder ran.
  - The encoder didn't pool, so ConvLSTM operated at full input resolution.
    That is both expensive and wrong — temporal fusion belongs at the semantic
    (coarser) level, with spatial detail reinjected via skips.
  - The decoder up-sampled 4x past the input, forcing _resize_to_target() in
    the training loop to bilinearly downsample logits every step. Wasteful.

This v2:
  - Two-stage downsample encoder on BOTH the dynamic and static branches so
    skip features are aligned at each resolution.
  - ConvLSTM runs at the bottleneck (H/4, W/4), where it actually helps.
  - Static branch feeds the bottleneck AND both skip levels (terrain/fuel
    affects fire spread at every scale).
  - Decoder mirrors the encoder with skip concatenation and outputs at native
    input resolution (no more 4x upsample-then-downsample dance).

Constructor signature is backward-compatible with train_nautilus.py:
    ConvLSTMUNet(Cd, Cs, hidden=64, lstm_layers=2, drop=0.1, drop_decoder=None)

    `drop_decoder` defaults to `drop` when not provided, preserving old behavior.
    Stage-3 training passes a higher decoder dropout (0.15) while leaving the
    encoder at 0.10, which tends to help sparse-segmentation overfitting without
    starving the encoder of signal.

Note: state_dicts from v1 checkpoints will NOT load cleanly here. Train with
strict=False and accept that warm-start effectively re-initializes most layers.
Give the new checkpoint a distinct filename (the trainer does this via the
arch version tag).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- toggles ----
USE_GROUPNORM = True     # GN is more stable than BN for small/variable batches.
GN_GROUPS     = 8
USE_ASPP      = True     # Keep the light context pyramid at the bottleneck.
ARCH_VERSION  = "v3"     # Production currently serves the v3 checkpoint.
                         # Bump this only when the matching checkpoint and
                         # serving configuration are promoted together.


def Norm2d(num_channels: int):
    if USE_GROUPNORM:
        groups = GN_GROUPS if num_channels % GN_GROUPS == 0 else 1
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


def conv_block(in_c, out_c, drop=0.1):
    """Two 3x3 convs + norm + ReLU + dropout. Standard U-Net block."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        Norm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        Norm2d(out_c), nn.ReLU(inplace=True),
        nn.Dropout2d(drop),
    )


class ASPP(nn.Module):
    """Light atrous spatial pyramid pooling (bottleneck context module)."""

    def __init__(self, in_ch, out_ch, rates=(1, 2, 4), mid=None, drop=0.0):
        super().__init__()
        mid = mid or max(in_ch // 4, 8)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False),
                Norm2d(mid), nn.ReLU(inplace=True),
            )
        ])
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=r, dilation=r, bias=False),
                Norm2d(mid), nn.ReLU(inplace=True),
            ))
        fused_in = mid * (1 + len(rates))
        self.proj = nn.Sequential(
            nn.Conv2d(fused_in, out_ch, 1, bias=False),
            Norm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.proj(torch.cat(feats, dim=1))


# ------------------------- ConvLSTM -------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.x2h = nn.Conv2d(in_ch, 4 * hid_ch, k, padding=p, bias=True)
        self.h2h = nn.Conv2d(hid_ch, 4 * hid_ch, k, padding=p, bias=True)
        self.hid_ch = hid_ch

    def init(self, B, H, W, device, dtype):
        h = torch.zeros(B, self.hid_ch, H, W, device=device, dtype=dtype)
        c = torch.zeros_like(h)
        return (h, c)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.x2h(x) + self.h2h(h), 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class ConvLSTM(nn.Module):
    """Stack of ConvLSTM layers; returns the last timestep's hidden state."""

    def __init__(self, in_ch, hid_ch, layers=2):
        super().__init__()
        self.cells = nn.ModuleList([
            ConvLSTMCell(in_ch if i == 0 else hid_ch, hid_ch) for i in range(layers)
        ])

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h_seq = x
        for cell in self.cells:
            state = cell.init(B, H, W, x.device, x.dtype)
            outs = []
            for t in range(T):
                out, state = cell(h_seq[:, t], state)
                outs.append(out)
            h_seq = torch.stack(outs, dim=1)
        return h_seq[:, -1]  # [B, hid, H, W]


# ------------------------- U-Net pieces -------------------------

class Down(nn.Module):
    """Strided 2x2 downsample then conv_block."""

    def __init__(self, in_c, out_c, drop=0.1):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = conv_block(in_c, out_c, drop)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Bilinear upsample + 1x1 channel reduction (cheaper and less checkerboard-prone
    than ConvTranspose) + skip concatenation + conv_block.
    """

    def __init__(self, in_c, skip_c, out_c, drop=0.1):
        super().__init__()
        self.reduce = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.norm = Norm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.conv = conv_block(out_c + skip_c, out_c, drop)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.act(self.norm(self.reduce(x)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ------------------------- Full model -------------------------

class ConvLSTMUNet(nn.Module):
    """
    Inputs:
      x_dyn:  [B, T, Cd, H, W]
      x_stat: [B, Cs, H, W]
    Output:
      logits: [B, 1, H, W]   (same spatial size as inputs)
    """

    def __init__(self, Cd, Cs, hidden=64, lstm_layers=2, drop=0.1, drop_decoder=None):
        super().__init__()
        # Separate encoder and decoder dropout. Default: decoder matches encoder
        # (old behavior). Stage-3 passes drop_decoder=0.15 with drop=0.10.
        drop_enc = float(drop)
        drop_dec = float(drop if drop_decoder is None else drop_decoder)

        h1 = hidden
        h2 = hidden * 2
        h3 = hidden * 4

        # --- Dynamic branch encoder (shared weights across T) ---
        self.dyn_enc1 = conv_block(Cd, h1, drop_enc)       # [B*T, h1, H, W]
        self.dyn_down1 = Down(h1, h2, drop_enc)            # [B*T, h2, H/2, W/2]
        self.dyn_down2 = Down(h2, h3, drop_enc)            # [B*T, h3, H/4, W/4]

        # --- Static branch encoder ---
        self.stat_enc1  = conv_block(Cs, h1, drop_enc)     # [B, h1, H, W]
        self.stat_down1 = Down(h1, h2, drop_enc)           # [B, h2, H/2, W/2]
        self.stat_down2 = Down(h2, h3, drop_enc)           # [B, h3, H/4, W/4]

        # --- Temporal fusion at the bottleneck ---
        self.rnn = ConvLSTM(h3, h3, layers=lstm_layers)    # [B, h3, H/4, W/4]

        # --- Bottleneck fuse (dyn_rnn + stat_bottleneck) + ASPP ---
        self.fuse_bn = nn.Sequential(
            nn.Conv2d(h3 * 2, h3, 1, bias=False),
            Norm2d(h3), nn.ReLU(inplace=True),
        )
        # ASPP sits between encoder and decoder — use decoder dropout here.
        self.aspp = ASPP(h3, h3, rates=(1, 2, 4), drop=drop_dec) if USE_ASPP else nn.Identity()

        # --- Skip fusion convs: blend dyn (last timestep) with static at each level ---
        # Halves the skip channel count back down to the level's feature size so the
        # Up blocks see a predictable (out_c + level_c) input.
        self.skip1_fuse = nn.Sequential(
            nn.Conv2d(h1 * 2, h1, 1, bias=False), Norm2d(h1), nn.ReLU(inplace=True),
        )
        self.skip2_fuse = nn.Sequential(
            nn.Conv2d(h2 * 2, h2, 1, bias=False), Norm2d(h2), nn.ReLU(inplace=True),
        )

        # --- Decoder (uses drop_dec) ---
        self.up2 = Up(in_c=h3, skip_c=h2, out_c=h2, drop=drop_dec)   # -> [B, h2, H/2, W/2]
        self.up1 = Up(in_c=h2, skip_c=h1, out_c=h1, drop=drop_dec)   # -> [B, h1, H,   W  ]
        # Extra head dropout right before the 1x1 logits conv.
        self.head_drop = nn.Dropout2d(drop_dec)

        # --- Output head ---
        self.out = nn.Conv2d(h1, 1, 1)                           # logits

        # Expose for train_nautilus.init_final_bias_to_prior compatibility.
        # (It walks model.dec.out.bias; keep an alias so existing helper works.)
        self.dec = nn.Module()
        self.dec.out = self.out

    def forward(self, x_dyn, x_stat):
        B, T, Cd, H, W = x_dyn.shape

        # --- Dynamic branch: encode each timestep, collapse T into the batch dim ---
        x_flat = x_dyn.reshape(B * T, Cd, H, W)
        d1 = self.dyn_enc1(x_flat)                       # [B*T, h1, H,   W  ]
        d2 = self.dyn_down1(d1)                          # [B*T, h2, H/2, W/2]
        d3 = self.dyn_down2(d2)                          # [B*T, h3, H/4, W/4]

        # Reshape back to sequences for ConvLSTM.
        d3_seq = d3.reshape(B, T, d3.shape[1], d3.shape[2], d3.shape[3])
        dyn_bot = self.rnn(d3_seq)                       # [B, h3, H/4, W/4]

        # For skips, use the LAST timestep's features. The bottleneck already has
        # the full temporal summary; skips are there to restore spatial detail.
        d1_last = d1.reshape(B, T, d1.shape[1], d1.shape[2], d1.shape[3])[:, -1]
        d2_last = d2.reshape(B, T, d2.shape[1], d2.shape[2], d2.shape[3])[:, -1]

        # --- Static branch ---
        s1 = self.stat_enc1(x_stat)                      # [B, h1, H,   W  ]
        s2 = self.stat_down1(s1)                         # [B, h2, H/2, W/2]
        s3 = self.stat_down2(s2)                         # [B, h3, H/4, W/4]

        # --- Fuse at bottleneck and run context pyramid ---
        z = self.fuse_bn(torch.cat([dyn_bot, s3], dim=1))
        z = self.aspp(z)                                 # [B, h3, H/4, W/4]

        # --- Build multi-scale skips (dyn + static) at each level ---
        skip1 = self.skip1_fuse(torch.cat([d1_last, s1], dim=1))   # [B, h1, H,   W  ]
        skip2 = self.skip2_fuse(torch.cat([d2_last, s2], dim=1))   # [B, h2, H/2, W/2]

        # --- Decode ---
        z = self.up2(z, skip2)                           # [B, h2, H/2, W/2]
        z = self.up1(z, skip1)                           # [B, h1, H,   W  ]
        z = self.head_drop(z)                            # Stage-3: pre-logits dropout.
        return self.out(z)                               # [B, 1,  H,   W  ]
