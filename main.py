# @file main.py
# @brief プログラムの入口

##########################################################
'''
実行モードの追加
1. config を受け取る関数を書く (例: utils/validator/chat.py の chat_model(config))。
2. RUN_REGISTRY に "名前: 関数" を足す。
3. run.mode をその名前にするか、--mode で渡す。
'''
##########################################################

from collections.abc import Callable
from json            import JSONDecodeError
from pathlib         import Path
from types           import SimpleNamespace

import argparse
import json
import re

from utils.exceptions import Error
from utils.runtime    import remove_pycache
from utils.trainer    import pipeline_train
from utils.validator  import benchmark_model, chat_model

RUN_REGISTRY: dict[str, Callable[[SimpleNamespace], None]] = {
    "train"    : pipeline_train,
    "chat"     : chat_model,
    "benchmark": benchmark_model,
}



def main() -> None:
    '''
    #* config を読み、run.mode のモード (--mode があればそのモード) を実行する。
    #! JSON の設定ファイルを渡す (例: python main.py --config config/hybrid.json)。--mode chat や --mode benchmark を付けると、config を変えずに別のモードで動かせる。
    '''
    
    # 引数を読む
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path                 , required=True                                            )
    parser.add_argument("--mode"  , choices=list(RUN_REGISTRY), default=None , help="run mode to use instead of run.mode")
    args   = parser.parse_args()
    
    # 設定を読む (文字列の外の // から行末まではコメントとして消す)
    try:
        text   = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*', lambda match: match.group(1) or "", args.config.read_text(encoding="utf-8"))
        config = json.loads(text, object_hook=lambda data: SimpleNamespace(**data))
    except (OSError, JSONDecodeError) as error:
        raise Error("Failed to load the config file.", detail=f"{args.config}: {error}", hint="Check the path and the JSON format.", code=1)
    
    # 実行モードを選ぶ
    mode = args.mode if args.mode is not None else config.run.mode
    if mode not in RUN_REGISTRY:
        raise Error("Unknown run mode.", detail=f"run mode: {mode!r}\nAvailable: {', '.join(RUN_REGISTRY)}", hint="Set run.mode, or pass --mode with one of the available modes.", code=2)
    
    # モードを実行する。Ctrl+C では静かに止め、キャッシュは必ず消す
    try:
        RUN_REGISTRY[mode](config)
    except KeyboardInterrupt:
        Error.show_interrupted()
    finally:
        remove_pycache()
    return None


if __name__ == "__main__":
    main()
