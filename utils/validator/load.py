# @file load.py
# @brief 検証用に、学習したモデル、その重み、トークナイザーを読み込む

from pathlib      import Path
from rich.console import Console
from rich.text    import Text
from tokenizers   import Tokenizer
from torch        import nn
from types        import SimpleNamespace

import torch

from model            import build_model
from utils.exceptions import Error
from utils.runtime    import setup_device, setup_seed



def load_model(config: SimpleNamespace) -> tuple[nn.Module, Tokenizer, torch.device]:
    '''
    #* シードとデバイスを設定し、model.checkpoint の重みを入れたモデル、data.tokenizer_dir のトークナイザー、デバイスを返す。
    #? chat.py の chat_model() と benchmark.py の benchmark_model() が使う。
    #! モデルは評価モードで float32。model.checkpoint が null なら、モデルはランダムな重みのままで、注意を表示する。
    '''
    
    # トークナイザーがあることを確かめる
    tokenizer_path = Path(config.data.tokenizer_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise Error("Tokenizer is missing.", detail=f"Missing: {tokenizer_path}", hint="Set data.tokenizer_dir to the folder of the tokenizer of the model.", code=33)
    
    # 重みを入れたモデルをデバイスに読み込む (形は model.spec)
    setup_seed(config.run.seed)
    device = setup_device(config.run.device)
    model  = build_model(config.model.name, config.model.checkpoint, getattr(config.model, "spec", None)).to(device=device, dtype=torch.float32).eval()
    if config.model.checkpoint is None:
        Console().print(Text("[Validator] model.checkpoint is null, so the model has random weights.", style="yellow"))
    return model, Tokenizer.from_file(str(tokenizer_path)), device
