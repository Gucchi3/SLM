# @file score.py
# @brief トークン列の対数確率 (Tiny-ML Leaderboard と Open SLM Leaderboard の決まり用)

from rich.progress import track
from torch         import nn

import math
import torch



def score_texts(model: nn.Module, ids_list: list[list[int]], device: torch.device, batch_size: int) -> list[float]:
    '''
    #* 各トークン列について、2 つ目から最後のトークンまでの log P(トークン | その前のトークン) の合計を返す。
    #? benchmark.py が BLiMP、ARC-Easy、WikiText-2 (Tiny-ML Leaderboard) に使う。
    #! 最初のトークンは前にトークンが無いので点を付けない (BOS トークンは足さない)。2 トークン未満の列は -inf になる。
    #! バッチの列は右側を埋める。パディングが前のトークンの点を変えないよう、モデルは因果的であること。
    #! バッチを動かしている間は進み具合のバーを表示し、終わると消す。
    '''
    
    # 2 トークン以上の列に、batch_size 列ずつ点を付ける
    scores = [-math.inf] * len(ids_list)
    usable = [index for index, ids in enumerate(ids_list) if len(ids) >= 2]
    with torch.no_grad():
        for start in track(range(0, len(usable), batch_size), description="Scoring", transient=True):
            # 入力、正解 (入力を 1 トークンずらしたもの)、パディングでないトークンのマスクを作る
            batch   = usable[start:start + batch_size]
            width   = max(len(ids_list[index]) for index in batch) - 1
            inputs  = torch.zeros((len(batch), width), dtype=torch.int64)
            targets = torch.zeros((len(batch), width), dtype=torch.int64)
            mask    = torch.zeros((len(batch), width), dtype=torch.bool)
            for row, index in enumerate(batch):
                ids                         = torch.tensor(ids_list[index])
                inputs[row, :len(ids) - 1]  = ids[:-1]
                targets[row, :len(ids) - 1] = ids[1:]
                mask[row, :len(ids) - 1]    = True
            
            # パディングでない正解の対数確率を足す
            logits    = model(inputs.to(device), torch.ones_like(inputs).to(device)).float()
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1)
            for index, total in zip(batch, (log_probs * mask.to(device)).sum(dim=-1).tolist()):
                scores[index] = total
    return scores


def score_continuations(model: nn.Module, requests: list[tuple[list[int], int]], device: torch.device, batch_size: int) -> list[float]:
    '''
    #* トークン列と数 n の組ごとに、列の最後の n トークンについての log P(トークン | その前のトークン) の合計を返す。
    #? benchmark.py が HellaSwag、ARC、PIQA、ArithMark-3 (Open SLM Leaderboard) に使う。最後の n トークンが選択肢になる。
    #! モデルは最後のトークンを除いた列を読むので、n は 1 以上で列の長さより小さくする。
    #! 長い列から先に動かす。バッチの中の列の長さがそろい (パディングが少ない)、メモリ不足も最初のバッチで分かる。点は頼まれた順に戻す。
    #! バッチの列は右側を埋めるので、モデルは因果的であること。バッチを動かしている間は進み具合のバーを表示し、終わると消す。
    '''
    
    # 頼まれたものを、列の長い順に並べる (同じ長さなら元の順を保つ)
    scores = [0.0] * len(requests)
    order  = sorted(range(len(requests)), key=lambda index: len(requests[index][0]), reverse=True)
    with torch.no_grad():
        for start in track(range(0, len(order), batch_size), description="Scoring", transient=True):
            # 入力、正解 (入力を 1 トークンずらしたもの)、各列の最後の n 個の正解のマスクを作る
            batch   = order[start:start + batch_size]
            width   = len(requests[batch[0]][0]) - 1
            inputs  = torch.zeros((len(batch), width), dtype=torch.int64)
            targets = torch.zeros((len(batch), width), dtype=torch.int64)
            mask    = torch.zeros((len(batch), width), dtype=torch.bool)
            for row, index in enumerate(batch):
                ids, count                 = torch.tensor(requests[index][0]), requests[index][1]
                end                        = len(ids) - 1
                inputs[row, :end]          = ids[:-1]
                targets[row, :end]         = ids[1:]
                mask[row, end - count:end] = True
            
            # マスクした正解の対数確率を足す
            logits    = model(inputs.to(device), torch.ones_like(inputs).to(device)).float()
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1)
            for index, total in zip(batch, torch.where(mask.to(device), log_probs, 0.0).sum(dim=-1).tolist()):
                scores[index] = total
    return scores
