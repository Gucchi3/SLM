# @file generate.py
# @brief 言語モデルで新しいトークンを 1 つずつ作る

from torch import nn

import torch



def generate_tokens(model: nn.Module, ids: list[int], device: torch.device, max_length: int, max_new_tokens: int, temperature: float, top_k: int, stop_id: int | None) -> list[int]:
    '''
    #* ids に続く新しいトークン ID を、最大 max_new_tokens 個まで 1 つずつ作り、stop_id の前で止める。
    #? chat.py の chat_model() が使う。
    #! モデルは最後の max_length トークンだけを見る。temperature が 0 なら必ず一番確率の高いトークンを選ぶ。top_k は選ぶ前に確率の高い k 個だけを残す (0 なら全部残す)。
    #! ランダムな選び方は PyTorch 全体の乱数の状態を使うので、setup_seed() をすれば毎回同じ答えになる。
    '''
    
    new_ids = list[int]()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 最後の max_length トークンから、次のトークンの点を求める
            context = torch.tensor([(ids + new_ids)[-max_length:]], device=device)
            logits  = model(context, torch.ones_like(context))[0, -1].float()
            
            # 一番確率の高いトークンを選ぶか、温度でやわらげた確率で top_k の中からランダムに選ぶ
            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                if top_k > 0:
                    logits = logits.masked_fill(logits < logits.topk(min(top_k, logits.numel())).values[-1], -torch.inf)
                next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))
            
            # 止めるトークンの前で止める
            if next_id == stop_id:
                break
            new_ids.append(next_id)
    return new_ids
