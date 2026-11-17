# @file build.py
# @brief 名前からモデルを作る

##########################################################
'''
モデルの追加
1. model/xxx/ を作り、設定 (SimpleNamespace か None) を受け取ってモデルを返す build_xxx_model(settings) を書く。
2. MODEL_REGISTRY に "名前: build_xxx_model" を足す。
3. config.model.name (と config.model.teacher_name) をその名前にし、設定を config.model.spec (と config.model.teacher_spec) に書く。
'''
##########################################################

from collections.abc import Callable
from rich.console    import Console
from rich.text       import Text
from torch           import nn
from types           import SimpleNamespace

import torch

from utils.exceptions import Error
from .hybrid          import build_hybrid_model

# モデルの名前と、それを作る関数。hybrid のブロック列や各部の形は config.model.spec にすべて書く (config/hybrid.json を参照)
MODEL_REGISTRY: dict[str, Callable[[SimpleNamespace | None], nn.Module]] = {
    "hybrid": build_hybrid_model,
}



def build_model(name: str, checkpoint: str | None, settings: SimpleNamespace | None) -> nn.Module:
    '''
    #* 名前のモデルを設定から作り、チェックポイントがあれば重みを読み込む。
    #? utils/trainer/pipeline.py (生徒と教師) と utils/validator/load.py から呼ぶ。
    #! settings は config.model.spec (教師は config.model.teacher_spec)。書かれていなければ None を渡し、必要かどうかはモデルを作る関数が決める。
    #! checkpoint が null なら、各層の初期値から始める。
    #! 名前が合わない重み (別のタスク用に変えた出力層など) は読み飛ばして一覧を出す。名前が同じで形が違う層はエラーで止めるので、変えた層には新しい名前を付ける。
    '''
    
    # モデルを作る
    if name not in MODEL_REGISTRY:
        raise Error("Unknown model.", detail=f"model name: {name!r}\nAvailable: {', '.join(MODEL_REGISTRY)}", hint="Add the model to MODEL_REGISTRY in model/build.py.", code=6)
    model = MODEL_REGISTRY[name](settings)
    if checkpoint is None:
        return model
    
    # 学習済みの重みを読む
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except OSError as error:
        raise Error("Could not load the checkpoint.", detail=f"{checkpoint}: {error}", hint="Check model.checkpoint and model.teacher_checkpoint, or set them to null.", code=8)
    
    # 名前が合う重みを読み込む
    try:
        result = model.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        raise Error("The checkpoint does not fit the model.", detail=f"{checkpoint}: {error}", hint="A layer has the same name but another shape. Give the changed layer a new name, so its weights are skipped.", code=13)
    
    # 読み込まなかった重みを一覧にする
    console = Console()
    if result.missing_keys:
        console.print(Text(f"[Checkpoint] Not in {checkpoint}, so they keep the new initial values: {', '.join(result.missing_keys)}", style="yellow"))
    if result.unexpected_keys:
        console.print(Text(f"[Checkpoint] Not in the model {name}, so they are not used: {', '.join(result.unexpected_keys)}", style="yellow"))
    return model
