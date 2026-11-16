# @file hybrid_block.py
# @brief One block of the hybrid model: a token mixer (short convolution, Gated DeltaNet, or attention) and an MLP, each after a norm

from torch    import nn
from torch.nn import functional as F

import torch

from .gated_delta import gated_delta_chunk
from .types       import HybridSpec, MixerName



class HybridBlock(nn.Module):
    '''
    #* Hold one block: x + mix(norm(x)), then x + mlp(norm(x)).
    #? Made by HybridLm in hybrid_lm.py, one block for each letter of HybridSpec.mixers. The math is the same as ArchBlock of the architecture search.
    #? The "c" mixer is the gated short convolution of LFM2: qkv makes B, C, and h, (B x h) goes through a causal depthwise convolution, and is then multiplied by C.
    #? The "g" mixer is Gated DeltaNet (Qwen3.5 style). The "a" and "w" mixers are causal attention with kv_heads groups (GQA), and "w" sees only the last window tokens.
    #! The linear layers have no biases. Their first weights are normal with std 0.02.
    '''
    
    def __init__(self, spec: HybridSpec, mixer: MixerName) -> None:
        '''
        #* Make the norms, the layers of the mixer, and the layers of the MLP.
        '''
        
        # The norms
        super().__init__()
        norm_class = nn.LayerNorm if spec.norm == "layer" else nn.RMSNorm
        self.norm1 = norm_class(spec.dim)
        self.norm2 = norm_class(spec.dim)
        self.mixer = mixer
        self.spec  = spec
        
        # The layers of the mixer: the convolution (c), Gated DeltaNet with its gates and a norm on each head (g), or the attention (a, w)
        if mixer == "c":
            self.qkv  = nn.Linear(spec.dim, 3 * spec.dim, bias=False)
            self.conv = nn.Conv1d(spec.dim, spec.dim, spec.c_kernel, groups=spec.dim, bias=False)
            self.proj = nn.Linear(spec.dim, spec.dim, bias=False)
        elif mixer == "g":
            width        = spec.gdn_heads * spec.gdn_dim
            self.qkv     = nn.Linear(spec.dim, 4 * width + 2 * spec.gdn_heads, bias=False)
            self.conv    = nn.Conv1d(3 * width, 3 * width, spec.g_kernel, groups=3 * width, bias=False)
            self.proj    = nn.Linear(width, spec.dim, bias=False)
            self.hnorm   = nn.RMSNorm(spec.gdn_dim)
            self.a_log   = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, spec.gdn_heads)))
            self.dt_bias = nn.Parameter(torch.zeros(spec.gdn_heads))
        else:
            self.qkv  = nn.Linear(spec.dim, (spec.heads + 2 * spec.kv_heads) * spec.head_dim, bias=False)
            self.proj = nn.Linear(spec.heads * spec.head_dim, spec.dim, bias=False)
        
        # The layers of the MLP (GELU, squared ReLU, or SwiGLU with a gate and an up layer in one matrix)
        self.fc  = nn.Linear(spec.dim, (2 if spec.mlp == "swiglu" else 1) * spec.mlp_hidden, bias=False)
        self.out = nn.Linear(spec.mlp_hidden, spec.dim, bias=False)
        
        # The first weights of the linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        return None
    
    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        '''
        #* Return the block output for x (batch x tokens x dim).
        #? Called from HybridLm.forward() in hybrid_lm.py.
        #! rope is (cos, sin) of shape (tokens, head_dim / 2). Each token sees only itself and the tokens before it.
        '''
        
        # The qkv output
        batch, tokens, dim = x.shape
        spec               = self.spec
        qkv                = self.qkv(self.norm1(x))
        
        # Gated short convolution: (B x h) through a causal depthwise convolution over time (padded on the left), then times C
        if self.mixer == "c":
            b, c, v = qkv.split(dim, dim=2)
            gated   = F.pad((b * v).transpose(1, 2), (spec.c_kernel - 1, 0))
            mixed   = self.proj(self.conv(gated).transpose(1, 2) * c)
        
        # Gated DeltaNet: q, k, v through a causal short convolution and SiLU, q and k L2-normalized, the gated delta rule, then a norm on each head times SiLU(z)
        elif self.mixer == "g":
            nh, hd       = spec.gdn_heads, spec.gdn_dim
            width        = nh * hd
            qkv_c, z, ab = qkv.split([3 * width, width, 2 * nh], dim=2)
            qkv_c        = F.silu(self.conv(F.pad(qkv_c.transpose(1, 2), (spec.g_kernel - 1, 0))).transpose(1, 2))
            q, k, v      = (part.view(batch, tokens, nh, hd).transpose(1, 2) for part in qkv_c.split(width, dim=2))
            q            = F.normalize(q, dim=-1) * hd**-0.5
            k            = F.normalize(k, dim=-1)
            a, b         = ab.split(nh, dim=2)
            g            = (-self.a_log.exp() * F.softplus(a + self.dt_bias)).transpose(1, 2)
            out          = gated_delta_chunk(q, k, v, g, torch.sigmoid(b).transpose(1, 2))
            out          = self.hnorm(out.transpose(1, 2)) * F.silu(z.view(batch, tokens, nh, hd))
            mixed        = self.proj(out.reshape(batch, tokens, width))
        
        # Attention: split q, k, v into heads, rotate q and k (RoPE), and attend to all tokens before, or to the last window tokens
        else:
            heads, groups, hd = spec.heads, spec.kv_heads, spec.head_dim
            cos, sin          = rope
            q                 = qkv[..., :heads * hd].view(batch, tokens, heads, hd).transpose(1, 2)
            q                 = torch.cat((q[..., 0::2] * cos - q[..., 1::2] * sin, q[..., 0::2] * sin + q[..., 1::2] * cos), dim=-1)
            k, v              = qkv[..., heads * hd:].view(batch, tokens, 2 * groups, hd).transpose(1, 2).split([groups, groups], dim=1)
            k                 = torch.cat((k[..., 0::2] * cos - k[..., 1::2] * sin, k[..., 0::2] * sin + k[..., 1::2] * cos), dim=-1)
            if self.mixer == "w":
                index    = torch.arange(tokens, device=x.device)
                mask     = (index[None, :] <= index[:, None]) & (index[None, :] > index[:, None] - spec.window)
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=groups != heads)
            else:
                attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=groups != heads)
            mixed = self.proj(attended.transpose(1, 2).reshape(batch, tokens, heads * hd))
        x = x + mixed
        
        # MLP
        h = self.fc(self.norm2(x))
        if spec.mlp == "swiglu":
            gate, up = h.chunk(2, dim=-1)
            h        = F.silu(gate) * up
        else:
            h = F.relu(h) ** 2 if spec.mlp == "relu2" else F.gelu(h)
        return x + self.out(h)
