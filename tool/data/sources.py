# @file sources.py
# @brief 共通のコーパス処理で使うテキストと parquet のデータセットを、ダウンロードして読む

##########################################################
'''
ソースの追加
1. SOURCE_REGISTRY に "名前: HFTextSource(repo, folder, text_column)" を足す。
   folder には、テキストの列を持つ parquet ファイルを置く。
2. オプション --名前 でそのデータセットを選び、トークンファイルは data/名前 に入る。
'''
##########################################################

from collections.abc import Iterator
from dataclasses     import dataclass
from huggingface_hub import HfApi, hf_hub_download
from pathlib         import Path
from rich.console    import Console
from rich.progress   import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from urllib.request  import urlopen

import json
import pyarrow.parquet as pq
import random

from utils.exceptions import Error
from .corpus          import CHARS_PER_GB, CorpusItem


@dataclass(frozen=True)
class HFTextSource:
    '''
    #* Hugging Face Hub 上の parquet ファイルと、そのテキストの列の場所を持つ。
    '''
    
    repo       : str
    folder     : str
    text_column: str

SOURCE_REGISTRY: dict[str, HFTextSource] = {
    "fineweb_edu": HFTextSource("HuggingFaceFW/fineweb-edu"  , "sample/10BT"   , "text"),
    "cosmopedia" : HFTextSource("HuggingFaceTB/smollm-corpus", "cosmopedia-v2" , "text"),
    "finemath"   : HFTextSource("HuggingFaceTB/finemath"     , "finemath-4plus", "text"),
}
FILE_ORDER_SEED = 0
READ_BATCH      = 1024
SOURCE_URL      = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
SOURCE_FILES    = ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt")
STORY_END       = "<|endoftext|>"
READ_CHUNK      = 1 << 20



def download_hf_text(raw_dir: Path, source: HFTextSource, gb: float | None) -> None:
    '''
    #* source の parquet ファイルを、テキストが gb GB に達するまで raw_dir にダウンロードし、ファイルの一覧と量を raw_dir/files.json に保存する。
    #? 共通のコーパス処理から呼ぶ。files.json は read_hf_text() が読む。
    #! ファイルは FILE_ORDER_SEED で混ぜた順に取る。だから同じ量なら同じファイルになり、量を増やしても少ない量のときのファイルは残る。
    #! raw_dir にすでにあるファイルはダウンロードし直さない。
    #! gb=None なら、設定したソースのフォルダの parquet ファイルをすべてダウンロードし、テキストを全部使う。
    '''
    
    # フォルダの parquet ファイルを一覧にし、固定のシードで順番を混ぜる
    try:
        paths = sorted(entry.path for entry in HfApi().list_repo_tree(source.repo, source.folder, recursive=True, repo_type="dataset") if entry.path.endswith(".parquet"))
    except OSError as error:
        raise Error("Failed to download from Hugging Face.", detail=f"{source.repo}/{source.folder}: {error}", hint="Check the network and SOURCE_REGISTRY, then run the download stage again.", code=25)
    random.Random(FILE_ORDER_SEED).shuffle(paths)
    
    # 1 ファイルずつダウンロードしてテキストの文字数を数え、量に達するまで続ける
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = int(gb * CHARS_PER_GB) if gb is not None else None
    used   = list[str]()
    chars  = 0
    size   = 0
    for path in paths:
        if target is not None and chars >= target:
            break
        local = raw_dir / path
        if not local.exists():
            try:
                hf_hub_download(source.repo, path, repo_type="dataset", local_dir=raw_dir)
            except OSError as error:
                raise Error("Failed to download from Hugging Face.", detail=f"{source.repo}/{path}: {error}", hint="Check the network, then run the download stage again.", code=25)
        file = pq.ParquetFile(local)
        if source.text_column not in file.schema_arrow.names:
            raise Error("The text column is missing.", detail=f"{source.repo}/{path} has no column {source.text_column!r}.", hint="Fix text_column of the source in SOURCE_REGISTRY.", code=27)
        chars += sum(len(text) for batch in file.iter_batches(batch_size=READ_BATCH, columns=[source.text_column]) for text in batch.column(0).to_pylist() if text)
        size  += local.stat().st_size
        used.append(path)
    
    # ファイルと量を保存し、ダウンロードしたものを表示する
    files = {"repo": source.repo, "folder": source.folder, "text_column": source.text_column, "max_chars": target, "files": used, "chars_in_files": chars}
    (raw_dir / "files.json").write_text(json.dumps(files, indent=4), encoding="utf-8")
    Console().print(f"[Download] {source.repo}/{source.folder}: {len(used)} of {len(paths)} files ({size / 1e9:.2f} GB), with {chars / CHARS_PER_GB:.2f} GB of text. {(chars if target is None else min(chars, target)) / CHARS_PER_GB:.2f} GB of it is used.")
    if target is not None and chars < target:
        Console().print(f"[yellow][Download] {source.repo}/{source.folder} has only {chars / CHARS_PER_GB:.2f} GB of text, so all of it is used.[/yellow]")
    return None


def read_hf_text(raw_dir: Path) -> Iterator[CorpusItem]:
    '''
    #* raw_dir/files.json のファイルのテキストを、1 つずつテキストの項目として渡す。ダウンロードの段階の量に達したら終わる。
    #? corpus.py の split、tokenizer、binarize の段階から呼ぶ。どの段階も、同じテキストを同じ順で見る必要がある。
    #! 両端の空白は取り除き、空のテキストは飛ばす。
    '''
    
    # ファイルの一覧を読む
    files_path = raw_dir / "files.json"
    if not files_path.exists():
        raise Error("Downloaded files are missing.", detail=f"Missing: {files_path}", hint="Run python -m tool.data download with the same dataset flags.", code=26)
    files      = json.loads(files_path.read_text(encoding="utf-8"))
    
    # 量に達するまでテキストを渡す
    texts = (text for path in files["files"] for batch in pq.ParquetFile(raw_dir / path).iter_batches(batch_size=READ_BATCH, columns=[files["text_column"]]) for text in batch.column(0).to_pylist())
    chars = 0
    for text in texts:
        if files["max_chars"] is not None and chars >= files["max_chars"]:
            break
        if text and (text := text.strip()):
            chars += len(text)
            yield [("text", text)]


def download_tinystories(raw_dir: Path) -> None:
    '''
    #* TinyStories の GPT-4 版テキストファイル 2 つをダウンロードする。すでにあるファイルは飛ばす。
    #! ダウンロードはまず .part ファイルに書くので、途中で止まったダウンロードが完了したように見えることはない。
    '''
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        path = raw_dir / name
        if path.exists():
            Console().print(f"[Download] {path} is already there, so it is skipped.")
            continue
        part = path.with_name(path.name + ".part")
        try:
            with urlopen(f"{SOURCE_URL}/{name}") as response, part.open("wb") as file, Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn()) as progress:
                task = progress.add_task(name, total=int(response.headers.get("Content-Length", 0)) or None)
                while chunk := response.read(READ_CHUNK):
                    file.write(chunk)
                    progress.advance(task, len(chunk))
        except OSError as error:
            raise Error("Failed to download TinyStories.", detail=f"{SOURCE_URL}/{name}: {error}", hint="Check the network, then run the download stage again.", code=16)
        part.replace(path)
    return None


def read_tinystories(raw_dir: Path) -> Iterator[CorpusItem]:
    '''
    #* 空でない物語を、ソースファイルの順に読む。物語の区切りは end-of-text の印。
    #! どの段階も、同じ物語を同じ順で読む必要がある。
    '''
    
    paths = [raw_dir / name for name in SOURCE_FILES]
    if not all(path.exists() for path in paths):
        raise Error("TinyStories files are missing.", detail=f"Missing: {', '.join(str(path) for path in paths if not path.exists())}", hint="Run python -m tool.data download --tinystories.", code=17)
    for path in paths:
        with path.open(encoding="utf-8") as file:
            rest = ""
            while chunk := file.read(READ_CHUNK):
                pieces = (rest + chunk).split(STORY_END)
                rest = pieces.pop()
                for piece in pieces:
                    if story := piece.strip():
                        yield [("text", story)]
            if story := rest.strip():
                yield [("text", story)]
