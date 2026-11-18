# @file hotaru.py
# @brief Yokoyama をもとにして、構造の定数を固定した Hotaru

from torch import nn

import math
import torch

from utils.exceptions import Error

VOCAB_SIZE        = 4096
HIDDEN_SIZE       = 128
REDUCED_EMBEDDING = 16
NUM_LAYERS        = 4
FORWARD_EXPANSION = 1.0
KERNEL_SIZE       = 32
MAX_LENGTH        = 256
EPSILON           = 1e-5


class NanoCausalEmbedder(nn.Module):
    '''
    #* 小さくしたトークンの埋め込みと位置の埋め込みを、隠れ層の次元に広げる。
    '''

    def __init__(self) -> None:
        super().__init__()
        self.token        = nn.Embedding(VOCAB_SIZE, REDUCED_EMBEDDING, padding_idx=0)
        self.tok_expander = nn.Linear(REDUCED_EMBEDDING, HIDDEN_SIZE, bias=False)
        self.position     = nn.Embedding(MAX_LENGTH, REDUCED_EMBEDDING)
        self.pos_expander = nn.Linear(REDUCED_EMBEDDING, HIDDEN_SIZE, bias=False)
        return None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokens    = input_ids.shape[1]
        positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.tok_expander(self.token(input_ids)) + self.pos_expander(self.position(positions))


class EfficientDifferentialSkipAttention(nn.Module):
    '''
    #* Yokoyama と同じく、causal な attention と、depthwise の causal な畳み込みの枝を組み合わせる。
    '''

    def __init__(self) -> None:
        super().__init__()
        d_inner         = int(HIDDEN_SIZE * FORWARD_EXPANSION)
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
        lambda_1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        lambda_2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        scores   = torch.matmul(self.query(x), x.permute(0, 2, 1)) / math.sqrt(x.size(-1))
        scores   = scores.masked_fill(mask == 0, float("-inf"))
        weights  = torch.nan_to_num(nn.functional.softmax(scores, dim=-1))
        context  = torch.matmul(self.dropout(weights), x)
        out1     = self.att_output(context)
        tokens   = x.shape[1]
        branch   = self.conv1d(x.transpose(1, 2))[:, :, :tokens].transpose(1, 2)
        out2     = self.contract(self.act(branch))
        return lambda_1 * out1 - lambda_2 * out2


def make_score_mask(input_mask: torch.Tensor) -> torch.Tensor:
    '''
    #* Yokoyama のパディングのマスクの動きを、テンソルの値で決まる Python のスライスを使わずに再現する。
    '''

    tokens  = input_mask.shape[1]
    lengths = (input_mask != 0).sum(dim=1)
    queries = torch.arange(tokens, device=input_mask.device)[None, :] < lengths[:, None]
    return queries[:, :, None].to(input_mask.dtype) * input_mask[:, None, :]


class Hotaru(nn.Module):
    '''
    #* Yokoyama の構造で、位置を MAX_LENGTH 個にし、すべての RMSNorm の eps を EPSILON にしたモデル。
    #! ほかの構造の次元は Yokoyama と同じ。チェックポイントは、次元が合っているものしか読めない。
    '''

    def __init__(self) -> None:
        super().__init__()
        self.embedder   = NanoCausalEmbedder()
        self.norms      = nn.ModuleList(nn.RMSNorm(HIDDEN_SIZE, eps=EPSILON) for _ in range(NUM_LAYERS))
        self.attentions = nn.ModuleList(EfficientDifferentialSkipAttention() for _ in range(NUM_LAYERS))
        self.lm_head    = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)
        return None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = input_ids.shape[1]
        if tokens > self.embedder.position.num_embeddings:
            raise Error("Hotaru input exceeds its context length.", detail=f"Input: {tokens}, capacity: {self.embedder.position.num_embeddings}", hint="Set data.max_length at or below MAX_LENGTH in model/hotaru/hotaru.py.", code=30)
        padding_mask = make_score_mask(attention_mask)
        causal       = torch.tril(torch.ones(tokens, tokens, dtype=padding_mask.dtype, device=input_ids.device))
        score_mask   = padding_mask * causal
        x            = self.embedder(input_ids)
        for norm, attention in zip(self.norms, self.attentions):
            x = x + attention(norm(x), score_mask)
        return self.lm_head(x)
