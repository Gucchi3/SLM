# @file corpus.py
# @brief テキストや会話のコーパスを分割し、トークナイザーを学習して、トークンファイルを書く

from collections.abc    import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools          import partial
from itertools          import batched
from pathlib            import Path
from rich.console       import Console
from rich.table         import Table
from tokenizers         import Tokenizer
from typing             import Literal

import argparse
import hashlib
import json
import multiprocessing
import numpy as np
import os

from utils.exceptions import Error
from utils.runtime    import remove_pycache
from .tokenizer       import TOKENIZER_REGISTRY, TokenizerType, build_tokenizer

Role           = Literal["text", "user", "bot"]
CorpusItem     = list[tuple[Role, str]]
CorpusReader   = Callable[[Path], Iterator[CorpusItem]]
CorpusDownload = Callable[[Path], None]
Corpora        = dict[str, tuple[CorpusDownload, CorpusReader]]

SPECIAL_TOKENS         = ("<pad>", "<eos>", "<user>", "<bot>")
EOS_ID                 = 1
USER_ID                = 2
BOT_ID                 = 3
SPLIT_NAMES            = ("train", "validation", "test")
STAGES                 = ("download", "split", "tokenizer", "binarize", "all")
ENCODE_BATCH           = 1000
ENCODE_WORKERS         = min(os.cpu_count() or 1, 16)
ENCODE_TASKS_PER_CHILD = 100
CHARS_PER_GB           = 10**9

# 役割ごとに、テキストの ID の前後に置く ID
ROLE_TOKENS: dict[Role, tuple[list[int], list[int]]] = {
    "text": ([]       , [EOS_ID]),
    "user": ([USER_ID], []      ),
    "bot" : ([BOT_ID] , [EOS_ID]),
}



def split_corpus(data_dir: Path, read: CorpusReader, split_ratio: tuple[float, float, float], seed: int) -> None:
    '''
    #* 項目を train、validation、test に分け、その分け方を、各項目の大きさ、比率、分割のシードと一緒に split.npz に保存する。
    #? split とすべての段階で run_corpus() から呼ぶ。train_corpus_tokenizer() と binarize_corpus() は split.npz を読むので、どの段階も同じ分け方を使う。
    #! 項目の大きさは、その項目のテキストの文字数。train_corpus_tokenizer() はこれを使って train の項目の一部を選ぶ。
    #! test の比率が 0 なら test の項目は無く、binarize_corpus() の test のファイルは空になる。
    '''
    
    # 項目の数と、各項目の文字数を数える
    console = Console()
    console.print("[Split] Counting the items...")
    sizes   = np.fromiter((sum(len(text) for _, text in item) for item in read(data_dir / "raw")), dtype=np.int64)
    count   = len(sizes)
    
    # 分割ごとの項目数を決める
    validation_count = round(count * split_ratio[1])
    test_count       = round(count * split_ratio[2])
    train_count      = count - validation_count - test_count
    if min(train_count, validation_count) < 1 or (split_ratio[2] > 0 and test_count < 1):
        raise Error("Invalid split ratio.", detail=f"items: {count}, train: {train_count}, validation: {validation_count}, test: {test_count}", hint="Make train, validation, and a test above 0 large enough to hold one item each.", code=15)
    
    # 項目の番号をシードで混ぜ、各項目に分割を割り当てる (0 が train、1 が validation、2 が test)
    order = np.random.default_rng(seed).permutation(count)
    split = np.zeros(count, dtype=np.uint8)
    split[order[train_count:train_count + validation_count]] = 1
    split[order[train_count + validation_count:]]            = 2
    np.savez(data_dir / "split.npz", split=split, sizes=sizes, ratio=np.array(split_ratio), seed=np.array(seed))
    console.print(f"[Split] Split {count} items of {data_dir.name} into train {train_count}, validation {validation_count}, test {test_count}.")
    return None


def train_corpus_tokenizer(corpora: dict[Path, CorpusReader], tokenizer_dir: Path, tokenizer_type: TokenizerType, vocab_size: int, max_chars: int) -> None:
    '''
    #* すべてのコーパスの train の項目で、tokenizer_type のトークナイザーを 1 つ学習し、tokenizer_dir/tokenizer.json に保存する。
    #? tokenizer とすべての段階で run_corpus() から呼ぶ。corpora は、コーパスのフォルダごとの読み込み関数を渡す。先に各コーパスで split_corpus() を実行しておく。
    #! train の項目の文字数の合計が max_chars を超えるときは、各項目を max_chars / (全文字数) の確率で使う。だから使う文字数はおよそ max_chars になり、各コーパスの割合も保たれる。
    #! 特殊トークンには、SPECIAL_TOKENS の順に ID 0、1、2、3 を振る。binarize_corpus() が ID として足すので、テキストの中には入れない。
    #! この段階は tokenizer_dir のトークナイザーを上書きする。
    '''
    
    # 使う train の項目を選ぶ。選び方は各分割のシードで決まる (シードに 1 を足し、選び方が分け方に引きずられないようにする)
    splits      = {data_dir: _corpus_load_split(data_dir) for data_dir in corpora}
    train_chars = sum(int(sizes[split == 0].sum()) for split, sizes, _, _ in splits.values())
    chosen      = {data_dir: (split == 0) & (np.random.default_rng(seed + 1).random(len(split)) < max_chars / train_chars) for data_dir, (split, _, _, seed) in splits.items()}
    used_chars  = sum(int(sizes[chosen[data_dir]].sum()) for data_dir, (_, sizes, _, _) in splits.items())
    Console().print(f"[Tokenizer] Training on {used_chars / CHARS_PER_GB:.2f} GB of the {train_chars / CHARS_PER_GB:.2f} GB of train text (1 GB = 10^9 characters)...")
    
    # 選んだ項目でトークナイザーを学習する (1 つの項目のテキストは改行でつなぐ)
    tokenizer, trainer = build_tokenizer(tokenizer_type, vocab_size, list(SPECIAL_TOKENS))
    train_texts        = ("\n".join(text for _, text in item) for data_dir, read in corpora.items() for item, keep in zip(read(data_dir / "raw"), chosen[data_dir], strict=True) if keep)
    try:
        tokenizer.train_from_iterator(train_texts, trainer, length=sum(int(keep.sum()) for keep in chosen.values()))
    except ValueError:
        raise Error("Run the split stage again.", detail=f"A split.npz in {', '.join(str(data_dir) for data_dir in corpora)} has another number of items than its raw files.",
                    hint="The raw files changed after the split stage. Run the split stage, then this stage again.", code=18)
    
    # トークナイザーを保存する
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    Console().print(f"[Tokenizer] Saved {tokenizer_dir / 'tokenizer.json'} with {tokenizer.get_vocab_size()} tokens.")
    return None


def binarize_corpus(name: str, data_dir: Path, tokenizer_dir: Path, read: CorpusReader) -> None:
    '''
    #* すべての項目を、役割のトークン付きのトークン ID にし、uint16 で train.bin、validation.bin、test.bin に書く。各項目の始まりは .idx と meta.json に書く。
    #? binarize とすべての段階で run_corpus() から呼ぶ。項目のエンコードは _corpus_encode() がワーカープロセスで行う。.bin ファイルは utils/data/bin2dataset.py の Bin2dataset が読む。
    #! テキストの後ろには <eos> を付ける。ユーザーの発話の前には <user> を付ける。ボットの発話の前には <bot>、後ろには <eos> を付ける (ROLE_TOKENS を参照)。
    #! 各テキストは単独でエンコードし、役割のトークンとは ID でつなぐ。だから結果は、トークナイザーの空白の扱いや言語に左右されない。
    #! 先に split_corpus() を実行しておく。tokenizer_dir のトークナイザーをそのまま使うので、既存のトークナイザーでもよい。ファイルは、すべての項目を書き終えてから名前を付ける。
    #! ENCODE_WORKERS 個のプロセスが、ENCODE_BATCH 個ずつ項目をエンコードする。各プロセスは 1 スレッドで動く (TOKENIZERS_PARALLELISM=false。新しいプロセスにも引き継がれる)。1 プロセスで多くのスレッドを使うと、Windows では CPU コアを 1、2 個しか使わなかった。
    #! tokenizers のエンコードは、ワーカーの中でバッチごとに約 7 MB を確保したまま返さない。そこで ENCODE_TASKS_PER_CHILD バッチごとにワーカーを新しく作り直し、ワーカー 1 つのメモリを 1.5 GB 程度に抑える。
    #! max_tasks_per_child を使うと、ProcessPoolExecutor は開始方法を渡さない限り spawn を使う。そこで既定の開始方法 (Linux は forkserver、Windows は spawn) を渡し、新しいワーカーがモジュールを読み直さずにすばやく始まるようにする。
    #! ENCODE_WORKERS は最大 16 にする。CPU コアの多いマシンで、ワーカーがメモリを使い切らないようにするため。
    '''
    
    # 分け方とトークナイザーを読む
    split, _, ratio, seed = _corpus_load_split(data_dir)
    tokenizer_path        = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise Error("Tokenizer is missing.", detail=f"Missing: {tokenizer_path}", hint="Run the tokenizer stage, or set --tokenizer_dir to the folder of an existing tokenizer.", code=20)
    tokenizer             = Tokenizer.from_file(str(tokenizer_path))
    
    # 分割ごとに .part ファイルを開く: トークン ID (.bin、uint16) と各項目の始まり (.idx、uint64)
    bin_parts = [data_dir / f"{split_name}.bin.part" for split_name in SPLIT_NAMES]
    idx_parts = [data_dir / f"{split_name}.idx.part" for split_name in SPLIT_NAMES]
    bin_files = [path.open("wb") for path in bin_parts]
    idx_files = [path.open("wb") for path in idx_parts]
    items     = [0, 0, 0]
    tokens    = [0, 0, 0]
    roles     = set[Role]()
    
    # ワーカープロセスでバッチをエンコードし、結果を項目の順に書く。map() は buffersize バッチ先までしか項目を読まないので、メモリは小さいまま
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        with ProcessPoolExecutor(ENCODE_WORKERS, mp_context=multiprocessing.get_context(), max_tasks_per_child=ENCODE_TASKS_PER_CHILD) as pool:
            batches = batched(zip(read(data_dir / "raw"), split, strict=True), ENCODE_BATCH)
            for ids_by_split, starts_by_split, batch_roles in pool.map(partial(_corpus_encode, str(tokenizer_path)), batches, buffersize=2 * ENCODE_WORKERS):
                # 分割ごとに、ID と項目の始まりを書く。バッチの中の始まりは、それまでに書いたトークン数だけずらす
                for part, (ids, starts) in enumerate(zip(ids_by_split, starts_by_split)):
                    bin_files[part].write(ids.tobytes())
                    idx_files[part].write((starts + tokens[part]).tobytes())
                    items[part]  += len(starts)
                    tokens[part] += len(ids)
                roles |= batch_roles
        
        # 各 .idx の最後にトークン数を書き、項目 i が ids[idx[i]:idx[i + 1]] になるようにする
        for part, idx_file in enumerate(idx_files):
            idx_file.write(np.array([tokens[part]], dtype=np.uint64).tobytes())
    except ValueError:
        raise Error("Run the split stage again.", detail=f"split.npz has {len(split)} items, but the raw files have another number.", hint="The raw files changed after the split stage. Run the split stage again.", code=18)
    finally:
        for file in (*bin_files, *idx_files):
            file.close()
    
    # 書き終えたファイルに名前を付ける
    for path in (*bin_parts, *idx_parts):
        path.replace(path.with_suffix(""))
    
    # トークンファイルの詳細を保存する。後で違うトークナイザーに気付けるよう、トークナイザーのハッシュも入れる
    meta = {
        "name"            : name,
        "kind"            : "chat" if roles & {"user", "bot"} else "text",
        "tokenizer"       : tokenizer_path.as_posix(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size"      : tokenizer.get_vocab_size(),
        "special_tokens"  : {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "dtype"           : "uint16",
        "split_ratio"     : ratio,
        "seed"            : seed,
        "items"           : dict(zip(SPLIT_NAMES, items)),
        "tokens"          : dict(zip(SPLIT_NAMES, tokens)),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")
    
    # 分割ごとの数を表示する
    table = Table(title=f"{name} token files")
    for column in ("Split", "Items", "Tokens", "Tokens per item"):
        table.add_column(column, justify="right")
    for split_name, item_count, token_count in zip(SPLIT_NAMES, items, tokens):
        table.add_row(split_name, f"{item_count:,}", f"{token_count:,}", f"{token_count / item_count:.1f}" if item_count else "-")
    Console().print(table)
    return None


def _corpus_encode(tokenizer_path: str, batch: tuple[tuple[CorpusItem, int], ...]) -> tuple[list[np.ndarray], list[np.ndarray], set[Role]]:
    '''
    #* 1 バッチの項目をエンコードし、分割ごとに、つないだトークン ID (uint16) と各項目の始まり (uint64) を、バッチの中の役割と一緒に返す。
    #? binarize_corpus() が使い、ワーカープロセスで ENCODE_BATCH 個ずつ実行する。batch には各項目とその分割 (0 が train、1 が validation、2 が test) が入る。
    #! トークナイザーはバッチごとにファイルから読む。エンコードに比べれば短い時間で済む。
    '''
    
    # バッチのテキストをまとめてエンコードする。トークンの位置 (offsets) は使わないので作らない
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encodings = iter(tokenizer.encode_batch_fast([text for item, _ in batch for _, text in item]))
    
    # 各項目の ID を役割のトークンとつなぎ、その分割のリストに入れる
    joined = [list[int]() for _ in SPLIT_NAMES]
    starts = [list[int]() for _ in SPLIT_NAMES]
    roles  = set[Role]()
    for item, part in batch:
        starts[part].append(len(joined[part]))
        for role, _ in item:
            prefix, suffix = ROLE_TOKENS[role]
            joined[part].extend(prefix + next(encodings).ids + suffix)
            roles.add(role)
    return [np.array(ids, dtype=np.uint16) for ids in joined], [np.array(item_starts, dtype=np.uint64) for item_starts in starts], roles


def _corpus_load_split(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    '''
    #* split.npz を読み、各項目の分割、各項目の大きさ、分割の比率、シードを返す。
    #? train_corpus_tokenizer() と binarize_corpus() が使う。
    '''
    
    split_path = data_dir / "split.npz"
    if not split_path.exists():
        raise Error("Run the split stage first.", detail=f"Missing: {split_path}", hint="Run python -m tool.data split with the same dataset flags.", code=18)
    with np.load(split_path) as saved:
        return saved["split"], saved["sizes"], saved["ratio"].tolist(), int(saved["seed"])


def run_corpus(description: str, choose: Callable[[argparse.Namespace], Corpora], tokenizer_dir: Path, parents: tuple[argparse.ArgumentParser, ...] = ()) -> None:
    '''
    #* コマンドラインを読んで値を確かめ、choose() が返すコーパスについて、選んだ段階を実行する。
    #? tool/data/__main__.py から呼ぶ。choose() は読んだコマンドラインを受け取り、各コーパスのダウンロードと読み込みの関数を名前ごとに返す。
    #! 各コーパスは自分のフォルダ data_root/name を持つ。トークナイザーは全コーパスで 1 つを共有する。tokenizer の段階で全コーパスの train の項目で学習し、binarize の段階で各コーパスに使う。
    #! tokenizer_dir は --tokenizer_dir の既定値。parents はデータセットを選ぶフラグを足す。
    #! プロジェクトのフォルダから実行する (例: python -m tool.data all --tinystories)。既存のトークナイザーを残すときは、download、split、binarize を実行し、tokenizer の段階は飛ばす。
    '''
    
    # 引数のパーサーを作る
    parser = argparse.ArgumentParser(description=description, parents=parents)
    parser.add_argument("stage"           , choices=STAGES                                                                        )
    parser.add_argument("--data_root"     , type=Path , default=Path("data")                                                      )
    parser.add_argument("--tokenizer_dir" , type=Path , default=tokenizer_dir                                                     )
    parser.add_argument("--tokenizer_type", choices=list(TOKENIZER_REGISTRY), default="bytelevel_bpe"                             )
    parser.add_argument("--vocab_size"    , type=int  , default=4096                                                              )
    parser.add_argument("--tokenizer_gb"  , type=float, default=2.0, metavar="GB"                                                 )
    parser.add_argument("--split"         , type=float, default=[0.995, 0.005, 0.000], nargs=3, metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument("--seed"          , type=int  , default=0                                                                 )
    
    # Ctrl+C では静かに止め、エラーの後でもキャッシュは必ず消す
    try:
        # 値を読んで確かめる
        args = parser.parse_args()
        if not 256 + len(SPECIAL_TOKENS) <= args.vocab_size <= 65536:
            raise Error("Invalid vocab size.", detail=f"--vocab_size: {args.vocab_size}", hint=f"Use a number from {256 + len(SPECIAL_TOKENS)} to 65536.", code=14)
        if args.tokenizer_gb <= 0:
            raise Error("Invalid size of the tokenizer text.", detail=f"--tokenizer_gb: {args.tokenizer_gb}", hint="Use a number above 0, such as --tokenizer_gb 2.", code=23)
        if min(args.split[:2]) <= 0 or args.split[2] < 0 or abs(sum(args.split) - 1) > 1e-6:
            raise Error("Invalid split ratio.", detail=f"--split: {args.split}", hint="Use train and validation above 0 and a test of 0 or more that add up to 1, such as --split 0.99 0.005 0.005, or --split 0.995 0.005 0 for no test.", code=15)
        if args.seed < 0:
            raise Error("Invalid seed.", detail=f"--seed: {args.seed}", hint="Use a number of 0 or more.", code=19)
        corpora = choose(args)
        
        # 段階を順に実行する。tokenizer の段階では、全コーパス用のトークナイザーを 1 つ作る
        if args.stage in ("download", "all"):
            for name, (download, _) in corpora.items():
                download(args.data_root / name / "raw")
        if args.stage in ("split", "all"):
            for name, (_, read) in corpora.items():
                split_corpus(args.data_root / name, read, tuple(args.split), args.seed)
        if args.stage in ("tokenizer", "all"):
            train_corpus_tokenizer({args.data_root / name: read for name, (_, read) in corpora.items()}, args.tokenizer_dir, args.tokenizer_type, args.vocab_size, int(args.tokenizer_gb * CHARS_PER_GB))
        if args.stage in ("binarize", "all"):
            for name, (_, read) in corpora.items():
                binarize_corpus(name, args.data_root / name, args.tokenizer_dir, read)
    except KeyboardInterrupt:
        Error.show_interrupted()
    finally:
        remove_pycache()
    return None
