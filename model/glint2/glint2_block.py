# @file glint2_block.py
# @brief Glint-2 の共有のブロック: RoPE と窓を使う attention と、そのあとの SwiGLU の MLP

from torch    import nn
from torch.nn import functional as F

import torch



class Glint2Block(nn.Module):
    '''
    #* Glint-2 がくり返し動かす 1 つのブロック。RMSNorm と attention、次に RMSNorm と SwiGLU の MLP で、それぞれの出力を入力に足す（pre-norm の残差）。
    #? glint2.py の Glint2 が、モデル全体で 1 つと、coda の分にもう 1 つ作る。層は Glint-Research/Glint-2 の generate.py に合わせ、名前だけ短くしている（attn.qkv -> qkv、ffn.gate_up -> gate_up など）。
    #! query、key、value は 1 つの行列（qkv）から作り、ループの LoRA の出力（qkv_delta）をそれに足す。どの head も自分の key と value を持つ（GQA ではない）。
    '''
    
    def __init__(self, dim: int, num_heads: int, ffn_hidden: int) -> None:
        '''
        #* norm と線形層を作る（バイアスはない）。
        #! Glint-2 の generate.py と同じく、norm には nn.RMSNorm の標準の eps を使う。
        '''
        
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        
        # attention の層
        self.attn_norm = nn.RMSNorm(dim)
        self.qkv       = nn.Linear(dim, 3 * dim, bias=False)
        self.out       = nn.Linear(dim, dim    , bias=False)
        
        # MLP の層（gate と up の層は 1 つの行列にまとめている）
        self.ffn_norm = nn.RMSNorm(dim)
        self.gate_up  = nn.Linear(dim       , 2 * ffn_hidden, bias=False)
        self.down     = nn.Linear(ffn_hidden, dim           , bias=False)
        return None
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, qkv_delta: torch.Tensor, window: torch.Tensor | None) -> torch.Tensor:
        '''
        #* attention と MLP を通したあとの x を返す。形は (batch, tokens, dim)。
        #! cos と sin は RoPE の表で、形は (tokens, head_dim / 2)。qkv_delta は qkv(x) と同じ形で、(batch, tokens, 3 * dim)。
        #! window は、各トークンが見てよいトークン（自分と、窓の中にある前のトークン）のマスク。前のトークンがすべて窓の中にあるときは None にする。
        '''
        
        # query、key、value を作り、ループの LoRA の出力を足す。形は (batch, heads, tokens, head_dim)
        batch, tokens, dim = x.shape
        q, k, v            = (self.qkv(self.attn_norm(x)) + qkv_delta).split(dim, dim=-1)
        q                  = q.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k                  = k.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v                  = v.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # query と key を、各トークンの位置に合わせて回す（RoPE。偶数番目の要素と、そのすぐあとの奇数番目の要素を組にする）
        q = torch.stack((q[..., 0::2] * cos - q[..., 1::2] * sin, q[..., 0::2] * sin + q[..., 1::2] * cos), dim=-1).flatten(-2)
        k = torch.stack((k[..., 0::2] * cos - k[..., 1::2] * sin, k[..., 0::2] * sin + k[..., 1::2] * cos), dim=-1).flatten(-2)
        
        # attention を動かし（causal で、窓があるときは窓の中だけ）、x に足す
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True) if window is None else F.scaled_dot_product_attention(q, k, v, attn_mask=window)
        x         = x + self.out(attention.transpose(1, 2).reshape(batch, tokens, dim))
        
        # SwiGLU の MLP を動かし、x に足す
        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)
