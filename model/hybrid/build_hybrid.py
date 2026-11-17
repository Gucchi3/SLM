# @file build_hybrid.py
# @brief config.model.spec を読んで確かめ、hybrid モデルを作る

from dataclasses import is_dataclass
from types       import SimpleNamespace
from typing      import Any, Literal, get_args, get_origin, get_type_hints

from utils.exceptions import Error
from .hybrid_lm       import HybridLm
from .types           import MIXER_LETTERS, HybridSpec



def build_hybrid_model(settings: SimpleNamespace | None) -> HybridLm:
    '''
    #* config.model.spec (HybridSpec のすべての項目) を読み、値を確かめてから HybridLm を作る。
    #? model/build.py の MODEL_REGISTRY から build_model() が呼ぶ。
    #! 知らないキー、書かれていないキー、型の違う値、選べない値、範囲の外の値は、すべてエラーにする。省略したときの値は無い。
    '''
    
    # 設定を読む (キーと型と選べる値は _build_hybrid_spec() が確かめる)
    if settings is None:
        raise Error("model.spec is missing.", detail="The hybrid model needs all its settings in model.spec.", hint="Copy model.spec from config/hybrid.json.", code=40)
    spec = _build_hybrid_spec(HybridSpec, settings, "model.spec")
    
    # 全体とブロック列の値を確かめる (問題は全部集めてから 1 回で知らせる)
    problems = []
    letters  = sorted(set(spec.mixers) - set(MIXER_LETTERS))
    if not spec.mixers or letters:
        problems.append(f"model.spec.mixers: {spec.mixers!r} (use only {', '.join(MIXER_LETTERS)}, at least one letter; unknown: {', '.join(letters)})")
    for key, value in (("dim", spec.dim), ("vocab_size", spec.vocab_size), ("max_length", spec.max_length), ("conv.kernel", spec.conv.kernel), ("gdn.heads", spec.gdn.heads),
                       ("gdn.head_dim", spec.gdn.head_dim), ("gdn.kernel", spec.gdn.kernel), ("ffn.hidden", spec.ffn.hidden)):
        if value < 1:
            problems.append(f"model.spec.{key}: {value} (use 1 or more)")
    if spec.embed.rank < 0:
        problems.append(f"model.spec.embed.rank: {spec.embed.rank} (use 0 for one stage, or 1 or more)")
    
    # 注意の値を確かめる ("a" の window は使わないので 0、"w" の window は 1 以上)
    for name, attn in (("attn_global", spec.attn_global), ("attn_window", spec.attn_window)):
        where = f"model.spec.{name}"
        if attn.heads < 1 or attn.kv_heads < 1 or attn.heads % attn.kv_heads:
            problems.append(f"{where}.heads, kv_heads: {attn.heads}, {attn.kv_heads} (use 1 or more, and kv_heads must divide heads)")
        if attn.head_dim < 2 or attn.head_dim % 2:
            problems.append(f"{where}.head_dim: {attn.head_dim} (use an even number of 2 or more, for RoPE)")
        if (name == "attn_global" and attn.window != 0) or (name == "attn_window" and attn.window < 1):
            problems.append(f"{where}.window: {attn.window} (use 0 for attn_global, and 1 or more for attn_window)")
        if attn.pos.theta <= 0 or not 0.0 <= attn.pos.part <= 1.0:
            problems.append(f"{where}.pos: theta {attn.pos.theta}, part {attn.pos.part} (use theta above 0 and part from 0 to 1)")
        if attn.kv.mla.latent < (1 if attn.kv.mla.enable else 0):
            problems.append(f"{where}.kv.mla.latent: {attn.kv.mla.latent} (use 1 or more when mla.enable is true, and 0 or more when it is false)")
    
    # ループの値を確かめる
    loop = spec.loop
    if not (0 <= loop.start <= loop.end <= len(spec.mixers) and loop.times >= 1):
        problems.append(f"model.spec.loop: start {loop.start}, end {loop.end}, times {loop.times} (use 0 <= start <= end <= {len(spec.mixers)} blocks, and times >= 1)")
    if loop.lora_rank < 0 or ((loop.embed or loop.lora_rank) and loop.start == loop.end):
        problems.append(f"model.spec.loop: embed {loop.embed}, lora_rank {loop.lora_rank} (use lora_rank 0 or more, and use embed or lora_rank only with blocks in the loop, start < end)")
    if any(times < 1 for times in loop.times_train) or (loop.times_train and loop.start == loop.end):
        problems.append(f"model.spec.loop.times_train: {list(loop.times_train)} (use numbers of 1 or more, and only with blocks in the loop, start < end)")
    if problems:
        raise Error("Invalid value in model.spec.", detail="\n".join(problems), hint="Fix the values. config/hybrid.json lists every key with its choices.", code=41)
    return HybridLm(spec)


def _build_hybrid_spec[T](kind: type[T], data: Any, where: str) -> T:
    '''
    #* data (SimpleNamespace) を kind のデータクラスにする。中のデータクラスは自分を呼んで作る。
    #? build_hybrid_model() が使う。
    #! キーは kind の項目とちょうど同じにする。bool、int、float、str、Literal (選べる値)、tuple[int, ...] (整数のリスト) の型を確かめる。float の項目には int も書ける。
    '''
    
    # キーが過不足なく書かれているかを確かめる
    if not isinstance(data, SimpleNamespace):
        raise Error("Wrong type in model.spec.", detail=f"{where}: {data!r} (write an object {{...}})", hint="Fix the value. config/hybrid.json lists every key with its type.", code=42)
    hints   = get_type_hints(kind)
    given   = vars(data)
    unknown = sorted(set(given) - set(hints))
    missing = [key for key in hints if key not in given]
    if unknown or missing:
        raise Error("Unknown or missing key in model.spec.", detail=f"{where}\nunknown: {', '.join(unknown) or '-'}\nmissing: {', '.join(missing) or '-'}\nAvailable: {', '.join(hints)}", hint="Write every key, and no other key. config/hybrid.json lists them all.", code=40)
    
    # 項目ごとに型と選べる値を確かめる
    values = {}
    for key, hint in hints.items():
        value = given[key]
        path  = f"{where}.{key}"
        if isinstance(hint, type) and is_dataclass(hint):
            values[key] = _build_hybrid_spec(hint, value, path)
            continue
        if get_origin(hint) is tuple:
            if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise Error("Wrong type in model.spec.", detail=f"{path}: {value!r} (write a list of integers, such as [1, 3])", hint="Fix the value. config/hybrid.json lists every key with its type.", code=42)
            values[key] = tuple(value)
            continue
        if get_origin(hint) is Literal:
            if not isinstance(value, str) or value not in get_args(hint):
                raise Error("Invalid choice in model.spec.", detail=f"{path}: {value!r}\nAvailable: {', '.join(get_args(hint))}", hint="Use one of the available values.", code=41)
        elif (hint is bool and not isinstance(value, bool)) or (hint is str and not isinstance(value, str)) or (hint in (int, float) and (isinstance(value, bool) or not isinstance(value, (int, float) if hint is float else int))):
            raise Error("Wrong type in model.spec.", detail=f"{path}: {value!r} (write a value of type {hint.__name__})", hint="Fix the value. config/hybrid.json lists every key with its type.", code=42)
        values[key] = float(value) if hint is float else value
    return kind(**values)
