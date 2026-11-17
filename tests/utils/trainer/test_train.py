# @file test_train.py
# @brief 学習でループの周回数をランダムに選ぶ動きと、CUDA Graph との組み合わせのエラーを確かめる

from torch.utils.data import DataLoader
from types            import SimpleNamespace
from typing           import cast

import json
import pytest
import random
import torch

from model                      import build_model
from utils.exceptions           import Error
from utils.data                 import EpochSampler
from utils.trainer.optimization import build_loss_fn
from utils.trainer.pipeline     import pipeline_train
from utils.trainer.train        import train_steps
from utils.trainer.types        import TrainingComponents, TrainState



class _TestTrainSampler:
    '''
    #* train_steps() が呼ぶ set_position() だけを持つ、何もしないサンプラー。
    '''
    
    def set_position(self, epoch: int, start: int) -> None:
        '''
        #* 何もしない。
        '''
        
        return None


def _test_train_config(times_train: list[int], cuda_graph: bool) -> SimpleNamespace:
    '''
    #* 小さな hybrid モデル ((c)x1-a、dim 32) と、学習 6 回、蓄積 2 の設定を作る。
    '''
    
    # 小さな形 (config/hybrid.json と同じキー)
    attn = {"heads": 2, "head_dim": 16, "kv_heads": 1, "window": 0, "pos": {"type": "rope", "theta": 10000.0, "part": 1.0}, "qk_norm": False, "score": "softmax", "out_gate": "none", "kv": {"share": False, "mla": {"enable": False, "latent": 0}}}
    spec = {"mixers": "ca", "dim": 32, "vocab_size": 64, "max_length": 16, "embed": {"tie": False, "rank": 0, "pos": "none"}, "attn_global": attn, "attn_window": {**attn, "window": 4},
            "conv": {"kernel": 3}, "gdn": {"heads": 2, "head_dim": 8, "kernel": 4}, "ffn": {"type": "gelu", "hidden": 64}, "norm": {"type": "layer", "sandwich": False},
            "loop": {"start": 0, "end": 1, "times": 1, "embed": True, "lora_rank": 2, "times_train": times_train}}
    
    # 学習の設定
    data = {"run": {"seed": 5}, "data": {"batch_size": 2}, "model": {"name": "hybrid", "spec": spec},
            "train": {"steps": 6, "accumulation_steps": 2, "max_grad_norm": 1.0, "cuda_graph": cuda_graph, "compile": False, "log_interval": 100, "loss": {"name": "cross_entropy"}}}
    return json.loads(json.dumps(data), object_hook=lambda item: SimpleNamespace(**item))


def test_train_random_times() -> None:
    '''
    #* 重みの更新ごとに候補から周回数を選び、1 回の更新の中の micro-batch はすべて同じ周回数で通る。選び方は run.seed と更新の番号だけで決まる。
    '''
    
    # モデルの forward を包んで、渡された周回数を記録する
    config  = _test_train_config([1, 2, 4], cuda_graph=False)
    model   = build_model("hybrid", None, config.model.spec)
    seen    = []
    forward = model.forward
    model.forward = lambda input_ids, attention_mask, times=None: (seen.append(times), forward(input_ids, attention_mask, times))[1]
    
    # 学習 6 回 (micro-batch 12 個)
    loss_fn, kind = build_loss_fn(config)
    optimizer     = torch.optim.SGD(model.parameters(), lr=0.01)
    components    = TrainingComponents(loss_fn, kind, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0))
    ids           = torch.randint(0, 64, (2, 17))
    loader        = [{"input_ids": ids[:, :-1], "attention_mask": torch.ones(2, 16, dtype=torch.long), "targets": ids[:, 1:]}] * 12
    state         = TrainState(step=0, epoch=0, batch=0, best_ppl=float("inf"), history=[])
    metrics       = train_steps(config, torch.device("cpu"), model, None, cast(DataLoader, loader), cast(EpochSampler, _TestTrainSampler()), components, state, 6)
    
    # 更新 n の周回数は Random(seed * 1_000_003 + n).choice(候補)。同じ更新の 2 つの micro-batch は同じ数
    expected = [random.Random(5 * 1_000_003 + step).choice((1, 2, 4)) for step in range(6)]
    assert seen == [times for times in expected for _ in range(2)]
    assert len(set(expected)) >= 2
    assert state.step == 6 and metrics.loss > 0
    return None


def test_train_cuda_graph_error() -> None:
    '''
    #* 周回数の候補が 2 種類以上で train.cuda_graph が true なら、pipeline_train() がデータを読む前に番号 43 のエラーで止める。
    '''
    
    config        = _test_train_config([1, 3, 3], cuda_graph=True)
    config.train  = SimpleNamespace(**vars(config.train), method="normal", eval_interval=2)
    with pytest.raises(Error) as caught:
        pipeline_train(config)
    assert caught.value.code == 43
    return None
