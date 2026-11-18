# @file supra_mini_v6.py
# @brief Tiny-ML Leaderboard にある、パラメータ 1.41M の Llama のモデル Supra-Mini-v6

from torch    import nn
from torch.nn import functional as F

import math
import torch

from .llama_block import LlamaBlock

VOCAB_SIZE        = 4096
HIDDEN_SIZE       = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS        = 6
NUM_HEADS         = 4
NUM_KV_HEADS      = 2
MAX_POSITIONS     = 1024
ROPE_THETA        = 10000.0
RMS_NORM_EPS      = 1e-5
INIT_STD          = 0.02



class SupraMiniV6(nn.Module):
    '''
    #* パラメータ 1,410,688 個の Llama のモデル Supra-Mini-v6。層 6 つ、hidden size 128、query の head 4 つ、key と value の head 2 つで、入力と出力で 1 つの埋め込みを共有する。
    #? model/build.py の build_model() が、model.name = "supra_mini_v6" のときに作る。大きさは Hugging Face Hub の SupraLabs/Supra-Mini-v6-1M の config.json に合わせている。tokenizer は python -m tool.data で作る（model/weight/hotaru_edu/）。
    #! tokenizer のトークンは VOCAB_SIZE 個（4096 個。tool/data/ のデータセットのスクリプトの --vocab_size の標準の値）にする。入力は MAX_POSITIONS 個（1024 個）までのトークンにできる。
    '''
    
    cos: torch.Tensor
    sin: torch.Tensor
    
    def __init__(self) -> None:
        '''
        #* 層と RoPE の表を作り、最初の重みを決める。
        #! 最初の重みは Supra-Mini-v6 の train_model.py に合わせる。平均 0、std 0.02 の正規分布で、残差に足す o_proj と down_proj だけ std 0.02 / sqrt(2 * layers) にする。
        '''
        
        super().__init__()
        
        # 埋め込み、層、最後の norm を作る。出力にも埋め込みの重みを使うので、出力の層は別に作らない
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers       = nn.ModuleList(LlamaBlock(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, INTERMEDIATE_SIZE, RMS_NORM_EPS) for _ in range(NUM_LAYERS))
        self.norm         = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
        
        # 各位置の RoPE の cos と sin の表を作る。ここで作り直せるので、重みと一緒には保存しない
        head_dim = HIDDEN_SIZE // NUM_HEADS
        inv_freq = 1.0 / ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles   = torch.outer(torch.arange(MAX_POSITIONS, dtype=torch.float32), inv_freq).repeat(1, 2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        
        # 行列の最初の重みを決める（norm の重みは 1 から始める）
        for name, parameter in self.named_parameters():
            if parameter.ndim == 2:
                residual = name.endswith(("o_proj.weight", "down_proj.weight"))
                nn.init.normal_(parameter, std=INIT_STD / math.sqrt(2 * NUM_LAYERS) if residual else INIT_STD)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* 各位置で、次のトークンの logits を返す。形は (batch, tokens, VOCAB_SIZE)。
        #? utils/trainer/ の train.py と evaluate.py、utils/validator/ の generate.py と score.py から呼ぶ。
        #! attention_mask は使わない。attention は causal なので、行の終わりのパディングは、その前のトークンの結果を変えない（パディングの target は -100）。
        #! CUDA graph のため、ここではテンソルの値を CPU 側で読まない。
        '''
        
        # 最初の位置の分の RoPE の表を使って、層を動かす
        tokens = input_ids.shape[1]
        x      = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, self.cos[:tokens], self.sin[:tokens])
        
        # 最後の隠れ状態を、埋め込みの重みで logits に変える
        return F.linear(self.norm(x), self.embed_tokens.weight)
