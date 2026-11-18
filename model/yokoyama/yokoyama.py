# @file yokoyama.py
# @brief EmbBERT の構造をもとにした causal な言語モデル Yokoyama

from torch import nn

import math
import torch



VOCAB_SIZE       = 4096
HIDDEN_SIZE      = 128
REDUCED_EMBEDDING = 16
NUM_LAYERS        = 4
FORWARD_EXPANSION = 1.0
KERNEL_SIZE       = 32
MAX_LENGTH        = 256
EPSILON           = 1e-5



class NanoCausalEmbedder(nn.Module):
    '''
    #* 小さくしたトークンの埋め込みと位置の埋め込みで、トークンを隠れ層のベクトルに変える。
    #? 参考にした EmbBERT の実装の Nano_Embedder に合わせている。ただし、双方向の文のセグメントは使わない。
    #! トークンの埋め込みと位置の埋め込みは小さい次元で持ち、そのあとモデルの隠れ層の次元に広げる。
    '''
    
    def __init__(self) -> None:
        '''
        #* 小さくしたトークンの埋め込みと位置の埋め込み、それを広げる層を作る。
        '''
        
        super().__init__()
        self.token        = nn.Embedding(VOCAB_SIZE, REDUCED_EMBEDDING, padding_idx=0)
        self.tok_expander = nn.Linear(REDUCED_EMBEDDING, HIDDEN_SIZE, bias=False)
        self.position     = nn.Embedding(MAX_LENGTH, REDUCED_EMBEDDING)
        self.pos_expander = nn.Linear(REDUCED_EMBEDDING, HIDDEN_SIZE, bias=False)
        return None
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
        #* 入力のトークンごとに、隠れ層のベクトルを 1 つ返す。
        '''
        
        tokens    = input_ids.shape[1]
        positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.tok_expander(self.token(input_ids)) + self.pos_expander(self.position(positions))



class EfficientDifferentialSkipAttention(nn.Module):
    '''
    #* 効率のよい causal な attention と、depthwise の causal な畳み込みの枝を組み合わせる。
    #? attention と畳み込みの式は、参考にしたリポジトリの EfficientDifferentialSkipAttention に合わせている。
    #! 畳み込みは nn.Conv1d の padding で両端に 0 を足し、出力の初めから入力と同じ長さの分だけを使う。そのため、未来のトークンを読まない。
    '''
    
    def __init__(self) -> None:
        '''
        #* 差分の attention、畳み込み、活性化、縮める層を作る。
        '''
        
        super().__init__()
        d_inner = int(HIDDEN_SIZE * FORWARD_EXPANSION)
        self.dropout    = nn.Dropout(0.1)
        self.act        = nn.SiLU()
        self.lambda_q1  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.lambda_k1  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.lambda_q2  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.lambda_k2  = nn.Parameter(torch.zeros(HIDDEN_SIZE).normal_(mean=0, std=0.1))
        self.query      = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.att_output = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv1d     = nn.Conv1d(HIDDEN_SIZE, d_inner, KERNEL_SIZE, groups=HIDDEN_SIZE, padding=KERNEL_SIZE - 1)
        self.contract   = nn.Linear(d_inner, HIDDEN_SIZE)
        return None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        #* attention の枝と畳み込みの枝を差分で組み合わせた結果を返す。
        '''
        
        lambda_1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        lambda_2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        
        scores  = torch.matmul(self.query(x), x.permute(0, 2, 1)) / math.sqrt(x.size(-1))
        scores  = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.nan_to_num(nn.functional.softmax(scores, dim=-1))
        context = torch.matmul(self.dropout(weights), x)
        out1    = self.att_output(context)
        
        tokens = x.shape[1]
        branch = self.conv1d(x.transpose(1, 2))[:, :, :tokens].transpose(1, 2)
        out2   = self.contract(self.act(branch))
        return lambda_1 * out1 - lambda_2 * out2



def make_score_mask(input_mask: torch.Tensor) -> torch.Tensor:
    '''
    #* パディングのマスクを、query の位置ごとのスコアのマスクに広げる。
    '''
    
    batch_size, tokens = input_mask.shape
    score_mask = torch.zeros((batch_size, tokens, tokens), dtype=input_mask.dtype, device=input_mask.device)
    for batch in range(batch_size):
        length = (input_mask[batch] != 0).sum()
        score_mask[batch, :length, :] = input_mask[batch]
    return score_mask



class Yokoyama(nn.Module):
    '''
    #* 小さくした埋め込みと差分の skip attention を使う、EmbGPT 風の causal な言語モデル。
    #? model/build.py の build_model() が、model.name = "yokoyama" のときに作る。
    #! 参考にしたプロジェクトの EmbBERT の基本の次元を使う。トークン 4096 個、hidden size 128、層 4 つ、小さくした埋め込み 16、expansion 1、カーネルの大きさ 32、文脈の長さ 256。
    '''
    
    def __init__(self) -> None:
        '''
        #* 埋め込み、pre-norm の attention のブロック、言語モデルの head（隠れ層のベクトルから語彙の logits を出す線形層）を作る。
        '''
        
        super().__init__()
        self.embedder  = NanoCausalEmbedder()
        self.norms     = nn.ModuleList(nn.RMSNorm(HIDDEN_SIZE, eps=EPSILON) for _ in range(NUM_LAYERS))
        self.attentions = nn.ModuleList(EfficientDifferentialSkipAttention() for _ in range(NUM_LAYERS))
        self.lm_head   = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)
        return None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        #* causal に、次のトークンの logits を返す。形は (batch, tokens, vocabulary)。
        #? trainer と validator から、model/build.py を通して呼ぶ。
        #! 各トークンは、自分の位置までにある、パディングでないトークンだけに attention する。
        '''
        
        tokens = input_ids.shape[1]
        padding_mask = make_score_mask(attention_mask)
        causal = torch.tril(torch.ones(tokens, tokens, dtype=padding_mask.dtype, device=input_ids.device))
        score_mask = padding_mask * causal
        
        x = self.embedder(input_ids)
        for norm, attention in zip(self.norms, self.attentions):
            normalized = norm(x)
            update = attention(normalized, score_mask)
            x = x + update
        return self.lm_head(x)
