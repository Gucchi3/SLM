# @file llama_block.py
# @brief Llama のデコーダの層 1 つ: RoPE と GQA を使う attention と、そのあとの SwiGLU の MLP

from torch    import nn
from torch.nn import functional as F

import torch



class LlamaBlock(nn.Module):
    '''
    #* Llama のデコーダの層 1 つ。RMSNorm と attention、次に RMSNorm と SwiGLU の MLP で、それぞれの出力を入力に足す（pre-norm の残差）。
    #? supra_mini_v6.py の SupraMiniV6 が、層ごとに 1 つ作る。層の名前は transformers の LlamaDecoderLayer と同じで、"self_attn." と "mlp." を取ったもの。
    #! hidden_size は num_heads の倍数、num_heads は num_kv_heads の倍数にする。num_heads / num_kv_heads 個の query の head が、1 つの key の head と 1 つの value の head を共有する（GQA）。
    '''
    
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float) -> None:
        '''
        #* norm と線形層を作る（バイアスはない）。
        '''
        
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_size // num_heads
        
        # attention の層（key と value の head は query より少ない）
        kv_size              = num_kv_heads * self.head_dim
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.q_proj          = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj          = nn.Linear(hidden_size, kv_size    , bias=False)
        self.v_proj          = nn.Linear(hidden_size, kv_size    , bias=False)
        self.o_proj          = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # MLP の層
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=eps)
        self.gate_proj                = nn.Linear(hidden_size      , intermediate_size, bias=False)
        self.up_proj                  = nn.Linear(hidden_size      , intermediate_size, bias=False)
        self.down_proj                = nn.Linear(intermediate_size, hidden_size      , bias=False)
        return None
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        '''
        #* attention と MLP を通したあとの x を返す。形は (batch, tokens, hidden_size)。
        #! cos と sin は、トークンの位置の分の RoPE の表で、形は (tokens, head_dim)。
        #! 各トークンは自分と前のトークンだけを見る（causal な attention）。そのため、行の終わりのパディングは、その前のトークンの結果を変えない。
        '''
        
        # query、key、value を作る。形は (batch, heads, tokens, head_dim)
        batch, tokens, _ = x.shape
        h                = self.input_layernorm(x)
        q                = self.q_proj(h).view(batch, tokens, self.num_heads   , self.head_dim).transpose(1, 2)
        k                = self.k_proj(h).view(batch, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v                = self.v_proj(h).view(batch, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # query と key を、各トークンの位置に合わせて回す（RoPE。Llama と同じく、各 head の前半と後半を組にする）
        half = self.head_dim // 2
        q    = q * cos + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin
        k    = k * cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
        
        # attention を動かし（query の head の各グループは、key の head と value の head を 1 つずつ使う）、x に足す
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x         = x + self.o_proj(attention.transpose(1, 2).reshape(batch, tokens, -1))
        
        # SwiGLU の MLP を動かし、x に足す
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
