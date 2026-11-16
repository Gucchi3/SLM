# @file hybrid_block.py
# @brief hybrid モデルの 1 ブロック: トークンミキサー (短畳み込み、Gated DeltaNet、注意) と MLP を、それぞれ正規化の後に置く

from functools import partial
from torch     import nn
from torch.nn  import functional as F
from typing    import Literal, cast

import torch

from .gated_delta import gated_delta_chunk
from .int_norm    import IntNorm
from .types       import HybridSpec, MixerName



class HybridBlock(nn.Module):
    '''
    #* 1 ブロックを持つ: x + mix(norm(x))、続いて x + mlp(norm(x))。
    #? HybridLm (hybrid_lm.py) が HybridSpec.mixers の 1 文字ごとに 1 つ作る。計算はアーキテクチャ探索の ArchBlock と同じ。
    #? "c" は LFM2 のゲート付き短畳み込み。qkv から B、C、h を作り、(B x h) を因果的な深さ方向の畳み込みに通してから C を掛ける。
    #? "g" は Gated DeltaNet (Qwen3.5 の形)。"a" と "w" は kv_heads 組の因果注意 (GQA) で、"w" は直近 window トークンだけを見る。
    #! 線形層にバイアスは無い。重みの初期値は標準偏差 0.02 の正規分布。
    '''
    
    def __init__(self, spec: HybridSpec, mixer: MixerName, index: int) -> None:
        '''
        #* 正規化、ミキサーの層、MLP の層を作る。
        #! index はブロックの番号 (0 から)。ループで k と v を使い回すときの目印に使う。
        '''
        
        # 正規化 (sandwich のときはミキサーと MLP の出力にも掛ける)
        super().__init__()
        kind        = spec.norm.type
        norm_class  = nn.LayerNorm if kind == "layer" else nn.RMSNorm if kind == "rms" else partial(IntNorm, cast(Literal["rmsp2", "dyt"], kind))
        self.norm1  = norm_class(spec.dim)
        self.norm2  = norm_class(spec.dim)
        self.out1   = norm_class(spec.dim) if spec.norm.sandwich else None
        self.out2   = norm_class(spec.dim) if spec.norm.sandwich else None
        self.mixer  = mixer
        self.index  = index
        self.spec   = spec
        self.attn   = spec.attn_global if mixer == "a" else spec.attn_window
        
        # ミキサーの層: 畳み込み (c)、ゲートと頭ごとの正規化を持つ Gated DeltaNet (g)、注意 (a, w)
        if mixer == "c":
            self.qkv  = nn.Linear(spec.dim, 3 * spec.dim, bias=False)
            self.conv = nn.Conv1d(spec.dim, spec.dim, spec.conv.kernel, groups=spec.dim, bias=False)
            self.proj = nn.Linear(spec.dim, spec.dim, bias=False)
        elif mixer == "g":
            width        = spec.gdn.heads * spec.gdn.head_dim
            self.qkv     = nn.Linear(spec.dim, 4 * width + 2 * spec.gdn.heads, bias=False)
            self.conv    = nn.Conv1d(3 * width, 3 * width, spec.gdn.kernel, groups=3 * width, bias=False)
            self.proj    = nn.Linear(width, spec.dim, bias=False)
            self.hnorm   = nn.RMSNorm(spec.gdn.head_dim)
            self.a_log   = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, spec.gdn.heads)))
            self.dt_bias = nn.Parameter(torch.zeros(spec.gdn.heads))
        else:
            # 注意: q と、k と v (MLA では潜在ベクトルを作り、kv_up で広げる)、q と k の正規化、出力のゲート
            attn       = self.attn
            kv_width   = 2 * attn.kv_heads * attn.head_dim
            self.qkv   = nn.Linear(spec.dim, attn.heads * attn.head_dim + (attn.kv.mla.latent if attn.kv.mla.enable else kv_width), bias=False)
            self.kv_up = nn.Linear(attn.kv.mla.latent, kv_width, bias=False) if attn.kv.mla.enable else None
            self.proj  = nn.Linear(attn.heads * attn.head_dim, spec.dim, bias=False)
            self.qn    = nn.RMSNorm(attn.head_dim) if attn.qk_norm else None
            self.kn    = nn.RMSNorm(attn.head_dim) if attn.qk_norm else None
            self.gate  = nn.Linear(spec.dim, attn.heads * (attn.head_dim if attn.out_gate == "element" else 1), bias=False) if attn.out_gate != "none" else None
        
        # MLP の層 (GELU、2 乗した ReLU、または SwiGLU。SwiGLU はゲートと up を 1 つの行列にまとめる)
        self.fc  = nn.Linear(spec.dim, (2 if spec.ffn.type == "swiglu" else 1) * spec.ffn.hidden, bias=False)
        self.out = nn.Linear(spec.ffn.hidden, spec.dim, bias=False)
        
        # 線形層の重みの初期値
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        return None
    
    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, memo: dict[int, tuple[torch.Tensor, torch.Tensor]], lora: tuple[nn.Module, nn.Module] | None) -> torch.Tensor:
        '''
        #* x (batch x tokens x dim) に対するブロックの出力を返す。
        #? HybridLm.forward() (hybrid_lm.py) から呼ぶ。
        #! rope はこのブロックの注意で使う (cos, sin) で、形は (tokens, head_dim / 2)。位置を回さないときは None。各トークンは自分と前のトークンだけを見る。
        #! memo は forward 1 回ごとの辞書。kv.share のとき、ブロックの番号をキーにして 1 周目の k と v を置き、2 周目以降はそれを使う。
        #! lora はループの今の周回の (down, up) で、qkv の出力に up(down(h)) を足す。ループの外や lora_rank が 0 のときは None。
        '''
        
        # qkv の出力 (ループの周回ごとの低ランクの差分を足す)
        batch, tokens, dim = x.shape
        spec               = self.spec
        h                  = self.norm1(x)
        qkv                = self.qkv(h) if lora is None else self.qkv(h) + lora[1](lora[0](h))
        
        # ゲート付き短畳み込み: (B x h) を時間方向の因果的な深さ方向の畳み込みに通し (左を 0 で埋める)、C を掛ける
        if self.mixer == "c":
            b, c, v = qkv.split(dim, dim=2)
            gated   = F.pad((b * v).transpose(1, 2), (spec.conv.kernel - 1, 0))
            mixed   = self.proj(self.conv(gated).transpose(1, 2) * c)
        
        # Gated DeltaNet: q、k、v を因果的な短い畳み込みと SiLU に通し、q と k を L2 正規化して gated delta rule を計算し、頭ごとに正規化して SiLU(z) を掛ける
        elif self.mixer == "g":
            nh, hd       = spec.gdn.heads, spec.gdn.head_dim
            width        = nh * hd
            qkv_c, z, ab = qkv.split([3 * width, width, 2 * nh], dim=2)
            qkv_c        = F.silu(self.conv(F.pad(qkv_c.transpose(1, 2), (spec.gdn.kernel - 1, 0))).transpose(1, 2))
            q, k, v      = (part.view(batch, tokens, nh, hd).transpose(1, 2) for part in qkv_c.split(width, dim=2))
            q            = F.normalize(q, dim=-1) * hd**-0.5
            k            = F.normalize(k, dim=-1)
            a, b         = ab.split(nh, dim=2)
            g            = (-self.a_log.exp() * F.softplus(a + self.dt_bias)).transpose(1, 2)
            out          = gated_delta_chunk(q, k, v, g, torch.sigmoid(b).transpose(1, 2))
            out          = self.hnorm(out.transpose(1, 2)) * F.silu(z.view(batch, tokens, nh, hd))
            mixed        = self.proj(out.reshape(batch, tokens, width))
        
        # 注意: q、k、v を頭に分けて正規化と回転を掛け、前の全トークン (a) か直近 window トークン (w) を見て、頭ごとにゲートを掛ける
        else:
            # q と、k と v (MLA では潜在ベクトルから広げる)。正規化してから RoPE で回す
            attn              = self.attn
            heads, groups, hd = attn.heads, attn.kv_heads, attn.head_dim
            q                 = qkv[..., :heads * hd].view(batch, tokens, heads, hd).transpose(1, 2)
            kv                = qkv[..., heads * hd:] if self.kv_up is None else self.kv_up(qkv[..., heads * hd:])
            k, v              = kv.view(batch, tokens, 2 * groups, hd).transpose(1, 2).split([groups, groups], dim=1)
            if self.qn is not None and self.kn is not None:
                q = self.qn(q)
                k = self.kn(k)
            if rope is not None:
                cos, sin = rope
                q        = torch.cat((q[..., 0::2] * cos - q[..., 1::2] * sin, q[..., 0::2] * sin + q[..., 1::2] * cos), dim=-1)
                k        = torch.cat((k[..., 0::2] * cos - k[..., 1::2] * sin, k[..., 0::2] * sin + k[..., 1::2] * cos), dim=-1)
            if attn.kv.share:
                k, v = memo.setdefault(self.index, (k, v))
            
            # 見てよいトークンの印 (w は直近 window トークンだけ)
            index   = torch.arange(tokens, device=x.device)
            visible = index[None, :] <= index[:, None]
            if self.mixer == "w":
                visible = visible & (index[None, :] > index[:, None] - attn.window)
            
            # softmax の注意、または SimA (q と k をトークンごとに L1 正規化し、(q k^T / sqrt(head_dim)) v を見たトークンの数で割る、softmax なし)
            if attn.score == "sima":
                q        = q / q.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
                k        = k / k.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
                k, v     = k.repeat_interleave(heads // groups, dim=1), v.repeat_interleave(heads // groups, dim=1)
                scores   = (q @ k.transpose(-1, -2) * hd**-0.5).masked_fill(~visible, 0)
                attended = scores @ v / visible.sum(dim=-1, keepdim=True).to(v.dtype)
            elif self.mixer == "w":
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=visible, enable_gqa=groups != heads)
            else:
                attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=groups != heads)
            
            # 出力のゲート (ブロックの入力を正規化したものから作る sigmoid)
            if self.gate is not None:
                attended = attended * torch.sigmoid(self.gate(h)).view(batch, tokens, heads, -1).transpose(1, 2)
            mixed = self.proj(attended.transpose(1, 2).reshape(batch, tokens, heads * hd))
        x = x + (mixed if self.out1 is None else self.out1(mixed))
        
        # MLP
        h = self.fc(self.norm2(x))
        if spec.ffn.type == "swiglu":
            gate, up = h.chunk(2, dim=-1)
            h        = F.silu(gate) * up
        else:
            h = F.relu(h) ** 2 if spec.ffn.type == "relu2" else F.gelu(h)
        h = self.out(h)
        return x + (h if self.out2 is None else self.out2(h))
