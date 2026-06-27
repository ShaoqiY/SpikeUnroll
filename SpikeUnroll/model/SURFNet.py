import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def TFP(spk, channel_step=1):
    T = spk.size(1)
    assert T % (2 * channel_step) == 0, \
        f"spk.size(1)={T} must be divisible by 2*channel_step={2*channel_step}"

    dim = T // (2 * channel_step)
    outs = []

    for i in range(dim):
        if i == 0:
            cur = spk.mean(dim=1)
        else:
            cur = spk[:, i * channel_step: -i * channel_step].mean(dim=1)
        outs.append(cur)

    return torch.stack(outs, dim=1)

class SpikeResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.norm = nn.GroupNorm(8, ch)

        self.dwconv = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pwconv = nn.Conv2d(ch, ch, 1)

        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 1),
            nn.GELU(),
            nn.Conv2d(ch * 2, ch, 1)
        )

    def forward(self, x):
        identity = x

        h = self.norm(x)
        h = self.dwconv(h)
        h = self.pwconv(h)

        g = self.gate(h)
        x = identity + g * h

        x = x + self.ffn(x)

        return x

class ImgResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.norm = nn.GroupNorm(8, ch)

        self.dw3 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.dw5 = nn.Conv2d(ch, ch, 5, padding=2, groups=ch)

        self.fuse = nn.Conv2d(ch * 2, ch, 1)

        self.ffn = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 1),
            nn.GELU(),
            nn.Conv2d(ch * 2, ch, 1)
        )

    def forward(self, x):
        identity = x

        h = self.norm(x)

        h3 = self.dw3(h)
        h5 = self.dw5(h)

        h = self.fuse(torch.cat([h3, h5], dim=1))

        x = identity + h
        x = x + self.ffn(x)

        return x


class ImgEncoder(nn.Module):
    def __init__(self, in_ch, chs=(64, 128, 256), stem_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=1, padding=1),
            nn.GroupNorm(8, stem_ch),
            nn.GELU()
        )

        self.blocks = nn.ModuleList()
        prev_ch = stem_ch

        for ch in chs:
            layer = nn.Sequential(
                ImgResBlock(prev_ch),
                nn.Conv2d(prev_ch, ch, 3, 2, 1),
                nn.GroupNorm(8, ch),
                nn.GELU(),
            )
            self.blocks.append(layer)
            prev_ch = ch

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats

class SpikeEncoder(nn.Module):
    def __init__(
        self,
        chs=(64, 128, 256),
        spike_T=20,
        tfp_channel_step=1,
        stem_ch=32
    ):
        super().__init__()

        self.tfp_channel_step = tfp_channel_step
        fusion_in_ch = spike_T // (2 * tfp_channel_step)
         
        self.stem = nn.Sequential(
            nn.Conv2d(fusion_in_ch, stem_ch, kernel_size=1),
            nn.GroupNorm(8, stem_ch),
            nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()

        self.enc_blocks.append(nn.Sequential(
            SpikeResBlock(stem_ch),
            nn.Conv2d(stem_ch, chs[0], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, chs[0]),
            nn.GELU()
        ))

        prev = chs[0]
        for ch in chs[1:]:
            self.enc_blocks.append(nn.Sequential(
                SpikeResBlock(prev),
                nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, ch),
                nn.GELU()
            ))
            prev = ch

    def forward(self, x):
        x = TFP(x, channel_step=self.tfp_channel_step,)
        x = self.stem(x)

        feats = []
        for blk in self.enc_blocks:
            x = blk(x)
            feats.append(x)

        return feats

class ConvFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3, padding=1, groups=hidden_dim
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.ffn(x)


class BiWindowCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size=8,
        num_heads=4,
        attn_dropout=0.0,
        mlp_ratio=2.0
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ---------- QKV projection (Conv) ----------
        self.q_proj_img = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.k_proj_img = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.v_proj_img = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )

        self.q_proj_spk = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.k_proj_spk = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.v_proj_spk = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )

        # ---------- Output projection ----------
        self.out_proj_img = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )
        self.out_proj_spk = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )

        # ---------- Norm ----------
        self.ln_img_attn = nn.GroupNorm(1, dim)
        self.ln_spk_attn = nn.GroupNorm(1, dim)

        self.ln_img_ffn = nn.GroupNorm(1, dim)
        self.ln_spk_ffn = nn.GroupNorm(1, dim)

        # ---------- ConvFFN ----------
        self.ffn_img = ConvFFN(dim, mlp_ratio, attn_dropout)
        self.ffn_spk = ConvFFN(dim, mlp_ratio, attn_dropout)

        self.attn_drop = nn.Dropout(attn_dropout)

    # ==================================================
    # Window utils
    # ==================================================
    def window_partition(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))

        H_pad, W_pad = H + pad_h, W + pad_w

        x = rearrange(
            x,
            'b c (h ws1) (w ws2) -> (b h w) (ws1 ws2) c',
            ws1=ws, ws2=ws
        )
        return x, H_pad, W_pad, pad_h, pad_w

    def window_reverse(self, windows, H_pad, W_pad, pad_h, pad_w, B):
        ws = self.window_size
        C = windows.shape[-1]

        x = rearrange(
            windows,
            '(b h w) (ws1 ws2) c -> b c (h ws1) (w ws2)',
            b=B,
            h=H_pad // ws,
            w=W_pad // ws,
            ws1=ws,
            ws2=ws
        )

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H_pad - pad_h, :W_pad - pad_w]
        return x

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, img_feat, spk_feat):
        B, C, H, W = img_feat.shape

        # ---------- Pre-LN ----------
        img_ln = self.ln_img_attn(img_feat)
        spk_ln = self.ln_spk_attn(spk_feat)

        # ---------- QKV ----------
        qi = self.q_proj_img(img_ln)
        ki = self.k_proj_img(img_ln)
        vi = self.v_proj_img(img_ln)

        qs = self.q_proj_spk(spk_ln)
        ks = self.k_proj_spk(spk_ln)
        vs = self.v_proj_spk(spk_ln)

        # ---------- Window partition ----------
        qi, H_pad, W_pad, pad_h, pad_w = self.window_partition(qi)
        ki, _, _, _, _ = self.window_partition(ki)
        vi, _, _, _, _ = self.window_partition(vi)

        qs, _, _, _, _ = self.window_partition(qs)
        ks, _, _, _, _ = self.window_partition(ks)
        vs, _, _, _, _ = self.window_partition(vs)

        # ---------- Multi-head reshape ----------
        def reshape(x):
            return rearrange(
                x,
                'n l (h d) -> n h l d',
                h=self.num_heads,
                d=self.head_dim
            )

        qi, ki, vi = reshape(qi), reshape(ki), reshape(vi)
        qs, ks, vs = reshape(qs), reshape(ks), reshape(vs)

        # ---------- Bi-directional cross attention ----------
        attn_i2s = (qi @ ks.transpose(-2, -1)) * self.scale
        attn_s2i = (qs @ ki.transpose(-2, -1)) * self.scale

        attn_i2s = self.attn_drop(F.softmax(attn_i2s, dim=-1))
        attn_s2i = self.attn_drop(F.softmax(attn_s2i, dim=-1))

        out_i = rearrange(attn_i2s @ vs, 'n h l d -> n l (h d)')
        out_s = rearrange(attn_s2i @ vi, 'n h l d -> n l (h d)')

        # ---------- Window reverse + output proj ----------
        out_img = self.window_reverse(out_i, H_pad, W_pad, pad_h, pad_w, B)
        out_img = self.out_proj_img(out_img)

        out_spk = self.window_reverse(out_s, H_pad, W_pad, pad_h, pad_w, B)
        out_spk = self.out_proj_spk(out_spk)

        # ---------- Residual + ConvFFN ----------
        img_feat = img_feat + out_img
        img_feat = img_feat + self.ffn_img(
            self.ln_img_ffn(img_feat)
        )

        spk_feat = spk_feat + out_spk
        spk_feat = spk_feat + self.ffn_spk(
            self.ln_spk_ffn(spk_feat)
        )

        return img_feat, spk_feat

class LiteFusionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.img_gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

        self.spk_gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, out_img, out_spk):
        pre_cat = torch.cat([out_img, out_spk], dim=1)   # only 2C

        g_img = self.img_gate(pre_cat)
        g_spk = self.spk_gate(pre_cat)

        out = g_img * out_img + g_spk * out_spk

        return out

class MultiScaleFusion(nn.Module):
    def __init__(self, channels_list=(64, 128, 256), window_size=(16, 8, 4), num_heads=(2, 4, 8)):
        super().__init__()

        self.attn_blocks = nn.ModuleList([
            BiWindowCrossAttention(dim=ch, window_size=window_size[i], num_heads=num_heads[i])
            for i, ch in enumerate(channels_list)
        ])

        self.fusion_blocks = nn.ModuleList([
            LiteFusionBlock(ch) for ch in channels_list
        ])

    def forward(self, img_feats, spike_feats):
        fused = []

        for img, spk, attn, fuse in zip(
            img_feats, spike_feats, self.attn_blocks, self.fusion_blocks
        ):
            out_img, out_spk = attn(img, spk)
            out = img + fuse(out_img, out_spk)
            fused.append(out)

        return fused
    
class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, kernel_size=1):
        super().__init__()
        hidden_ch = in_ch if hidden_ch is None else hidden_ch
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch

        self.conv_zr = nn.Conv2d(
            in_ch + hidden_ch,
            2 * hidden_ch,
            kernel_size=kernel_size,
            padding=padding
        )
        self.conv_h = nn.Conv2d(
            in_ch + hidden_ch,
            hidden_ch,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        z, r = torch.chunk(self.conv_zr(combined), 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        candidate = torch.cat([x, r * h_prev], dim=1)
        h_hat = torch.tanh(self.conv_h(candidate))
        h = (1.0 - z) * h_prev + z * h_hat
        return h

    def init_state(self, feat):
        B, _, H, W = feat.shape
        return feat.new_zeros(B, self.hidden_ch, H, W)


class MultiScaleConvGRU(nn.Module):
    def __init__(self, channels, kernel_size=1, init_alpha=-2.0):
        super().__init__()
        self.cells = nn.ModuleList([
            ConvGRUCell(ch, ch, kernel_size=kernel_size)
            for ch in channels
        ])
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.full((1, ch, 1, 1), init_alpha))
            for ch in channels
        ])

    def init_states(self, feats):
        return [cell.init_state(feat) for feat, cell in zip(feats, self.cells)]

    @staticmethod
    def _states_match(feats, states):
        if states is None or len(states) != len(feats):
            return False
        return all(
            h.shape[0] == feat.shape[0]
            and h.shape[1] == feat.shape[1]
            and h.shape[2:] == feat.shape[2:]
            and h.device == feat.device
            and h.dtype == feat.dtype
            for h, feat in zip(states, feats)
        )

    def forward(self, feats, states=None):
        if not self._states_match(feats, states):
            states = self.init_states(feats)

        new_feats, new_states = [], []
        for feat, h_prev, cell, alpha_param in zip(feats, states, self.cells, self.alpha):
            h_new = cell(feat, h_prev)
            alpha = torch.sigmoid(alpha_param)  # sigmoid(-2)≈0.12，初期以当前帧为主
            out = (1.0 - alpha) * feat + alpha * h_new
            new_feats.append(out)
            new_states.append(h_new)

        return new_feats, new_states
    
class ResidualBlock(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, k, padding=k // 2),
            nn.GELU(),
            nn.Conv2d(ch, ch, k, padding=k // 2),
        )

    def forward(self, x):
        return x + self.block(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            ResidualBlock(out_ch)
        )
        self.ca = SEBlock(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.ca(x)
        return x
    


class LiteDecoder(nn.Module):
    def __init__(self, encoder_chs, fused_chs, out_ch=3):
        super().__init__()

        self.bottleneck = UpBlock(fused_chs[-1] + encoder_chs[-1], encoder_chs[-1])

        self.up_blocks = nn.ModuleList()
        in_ch = encoder_chs[-1]
        for e_ch in reversed(encoder_chs[:-1]):
            self.up_blocks.append(
                UpBlock(in_ch + e_ch * 2, e_ch * 2)
            )
            in_ch = e_ch * 2

        self.res_out = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, fused_feats, encoder_feats, img):
        x = self.bottleneck(torch.cat([encoder_feats[-1], fused_feats[-1]], dim=1))

        for up, enc_skip, fus_skip in zip(
                self.up_blocks,
                reversed(encoder_feats[:-1]),
                reversed(fused_feats[:-1]),
        ):
            x = up(torch.cat([x, enc_skip, fus_skip], dim=1))

        res = self.res_out(x)
        out = img + res
        return out

class SpikeImageFusion(nn.Module):
    def __init__(self, img_ch=3, encoder_chs=(64, 128, 256),
                 num_heads=(2, 4, 8), window_size=(16, 8, 4),
                 use_temporal_gru=True, gru_kernel_size=1):
        super().__init__()
        self.spike_encoder = SpikeEncoder(encoder_chs)
        self.img_encoder = ImgEncoder(img_ch, encoder_chs)

        self.fusion = MultiScaleFusion(encoder_chs, window_size=window_size, num_heads=num_heads)

        self.use_temporal_gru = use_temporal_gru
        if use_temporal_gru:
            self.temporal_gru = MultiScaleConvGRU(encoder_chs, kernel_size=gru_kernel_size)
            self._temporal_states = None

        self.decoder = LiteDecoder(encoder_chs, encoder_chs, out_ch=3)

    def reset_state(self):
        if self.use_temporal_gru:
            self._temporal_states = None

    def detach_state(self):
        if self.use_temporal_gru and self._temporal_states is not None:
            self._temporal_states = [h.detach() for h in self._temporal_states]

    def forward(self, img, spike, reset_state=False):
        if reset_state:
            self.reset_state()

        spike_feats = self.spike_encoder(spike)
        img_feats = self.img_encoder(img)

        fused_feats = self.fusion(img_feats, spike_feats)

        if self.use_temporal_gru:
            fused_feats, self._temporal_states = self.temporal_gru(
                fused_feats, self._temporal_states
            )

        out = self.decoder(fused_feats, img_feats, img)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    B = 2
    img_H, img_W = 504, 800
    img = torch.rand(B, 3, img_H, img_W).to(device)

    spk_H, spk_W = img_H // 2, img_W // 2
    spk = torch.rand(B, 20, spk_H, spk_W).to(device)

    net = SpikeImageFusion().to(device)

    y = net(img, spk)
    print("out", y.shape)  # expected (B,3,320,448)
