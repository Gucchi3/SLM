# @file __main__.py
# @brief 複数のテキストデータを 1 つの共通トークナイザーで準備する

from functools       import partial
from pathlib         import Path

import argparse

from utils.exceptions import Error
from .corpus          import Corpora, run_corpus
from .sources         import SOURCE_REGISTRY, download_hf_text, read_hf_text, download_tinystories, read_tinystories


def choose_corpora(args: argparse.Namespace) -> Corpora:
    '''
    #* フラグでデータセットを丸ごと選ぶ。ダウンロードと読み込みには形式ごとの関数を使う。
    #! 選ばなかったデータセットは、トークナイザーの学習も含めてすべての段階から外れる。
    '''
    
    if not any(getattr(args, name) for name in ("tinystories", *SOURCE_REGISTRY)):
        raise Error("No datasets selected.", hint="Select at least one dataset, such as --tinystories or --finemath.", code=24)
    corpora: Corpora = {}
    if args.tinystories:
        corpora["tinystories"] = (download_tinystories, read_tinystories)
    for name, source in SOURCE_REGISTRY.items():
        if getattr(args, name):
            corpora[name] = (partial(download_hf_text, source=source, gb=None), read_hf_text)
    return corpora


def main() -> None:
    '''
    #* 選んだデータセットを、共通のコーパス処理で準備する。
    '''
    
    options = argparse.ArgumentParser(add_help=False)
    for name in ("tinystories", *SOURCE_REGISTRY):
        options.add_argument(f"--{name}", action="store_true", help="Use the entire dataset (omitted by default)")
    run_corpus("Prepare selected datasets with one shared tokenizer.", choose_corpora, Path("model/weight/hotaru_mix"), (options,))
    return None


if __name__ == "__main__":
    main()
