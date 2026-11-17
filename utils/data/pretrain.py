# @file pretrain.py
# @brief tool/data/ で作ったトークンファイルから、事前学習用の DataBundle を作る

from pathlib          import Path
from rich.console     import Console
from rich.table       import Table
from torch.utils.data import ConcatDataset, DataLoader
from types            import SimpleNamespace

import hashlib
import json
import torch

from utils.exceptions import Error
from .bin2dataset     import Bin2dataset
from .epoch_sampler   import EpochSampler
from .types           import DataBundle, DatasetMetadata, PreprocessingMetadata

SPLIT_NAMES = ("train", "validation", "test")



def build_pretrain_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* 事前学習 (普通のテキストで次のトークンを当てる) 用の train、validation、test のローダーを返す。data.sources のフォルダのトークンファイルを混ぜて使う。
    #? data.name が "pretrain" のとき bundle.py で選ばれる。各フォルダには、tool/data/ のデータセットのスクリプトが作った train.bin、validation.bin、test.bin、meta.json が入っている。
    #! ソースの repeat は、1 エポックでその train の窓を何回使うかを決める。1 なら各 1 回、2 なら 2 回、0.5 なら前半だけ。validation と test は、全ソースの全窓を 1 回ずつ使う。
    #! すべてのトークンが正解になるので、損失のマスクは要らない。train のローダーは EpochSampler の順 (シードとエポックから作る) で窓を渡し、最後の短いバッチは捨てる。だからバッチはすべて同じ形になる。
    '''
    
    # ソースを確かめる
    if not config.sources or min(source.repeat for source in config.sources) <= 0:
        raise Error("Invalid data sources.", detail=f"data.sources: {[vars(source) for source in config.sources]}", hint='List one or more token folders, each with a repeat above 0, such as [{"root": "data/tinystories", "repeat": 1.0}].', code=28)
    
    # 各ソースのトークンファイルがあり、data.tokenizer_dir のトークナイザーで作られたことを確かめる
    tokenizer_path = Path(config.tokenizer_dir) / "tokenizer.json"
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() if tokenizer_path.exists() else None
    metas          = []
    for source in config.sources:
        root  = Path(source.root)
        paths = [root / "meta.json", *(root / f"{split}.bin" for split in SPLIT_NAMES)]
        if not all(path.exists() for path in paths):
            raise Error("Token files are missing.", detail=f"Missing: {', '.join(str(path) for path in paths if not path.exists())}",
                        hint="Run python -m tool.data all with dataset flags, such as --tinystories, and list their folders in data.sources.", code=21)
        meta  = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if meta["tokenizer_sha256"] != tokenizer_hash:
            raise Error("The token files do not match the tokenizer.", detail=f"The tokenizer hash in {root / 'meta.json'} differs from {tokenizer_path}.",
                        hint="Make all the token files in data.sources with one tokenizer (run the binarize stage again), and set data.tokenizer_dir to it.", code=22)
        metas.append(meta)
    
    # 分割ごとに各ソースの窓を作り、ソースをつなぐ (ソースの train の窓は repeat 回使う)
    parts    = {split: [Bin2dataset(Path(source.root) / f"{split}.bin", config.max_length, source.repeat if split == "train" else 1.0) for source in config.sources] for split in SPLIT_NAMES}
    datasets = {split: ConcatDataset(windows) for split, windows in parts.items()}
    
    # 1 エポックの train の窓の混ぜ方を表示する
    table = Table(title="Train data mix (one epoch)")
    for column in ("Source", "Train tokens", "Repeat", "Tokens per epoch", "Share"):
        table.add_column(column, justify="right")
    for source, meta, windows in zip(config.sources, metas, parts["train"]):
        table.add_row(source.root, f"{meta['tokens']['train']:,}", f"{source.repeat:g}", f"{len(windows) * config.max_length:,}", f"{len(windows) / len(datasets['train']):.1%}")
    Console().print(table)
    
    # ローダーを作る。train のローダーだけが、順番を混ぜ (シードとエポックで)、最後の短いバッチを捨て、エポックをまたいでワーカーを残す
    workers           = config.num_workers
    pin_memory        = torch.cuda.is_available()
    train_sampler     = EpochSampler(len(datasets["train"]), seed)
    train_loader      = DataLoader(datasets["train"]     , batch_size=config.batch_size, sampler=train_sampler, num_workers=workers, pin_memory=pin_memory, drop_last=True , persistent_workers=workers > 0)
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size, sampler=None         , num_workers=workers, pin_memory=pin_memory, drop_last=False, persistent_workers=False      )
    test_loader       = DataLoader(datasets["test"]      , batch_size=config.batch_size, sampler=None         , num_workers=workers, pin_memory=pin_memory, drop_last=False, persistent_workers=False      )
    
    # データセットの詳細を metadata.json 用に記録する (分割のシードは最初のソースのもの)
    preprocessing = PreprocessingMetadata(name="tokenizer", parameters={"tokenizer": metas[0]["tokenizer"], "vocab_size": metas[0]["vocab_size"], "max_length": config.max_length}, output_dtype="int64")
    metadata      = DatasetMetadata(name="+".join(meta["name"] for meta in metas), input_shape=(config.max_length,), num_classes=metas[0]["vocab_size"], class_names=None, train_size=len(datasets["train"]), validation_size=len(datasets["validation"]),
                                    test_size=len(datasets["test"]), split_seed=metas[0]["seed"], preprocessing=preprocessing)
    return DataBundle(train_loader, train_sampler, validation_loader, test_loader, metadata)
