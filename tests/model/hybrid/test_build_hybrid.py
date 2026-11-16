# @file test_build_hybrid.py
# @brief 命名規則で書けるモデルを config から組み立て、forward と backward が通り、パラメータ数が式と合うことを確かめる

from types  import SimpleNamespace
from typing import cast

import copy
import json
import pytest
import re
import torch

from model                     import build_model
from model.hybrid              import HybridLm
from model.hybrid.hybrid_block import HybridBlock
from utils.exceptions          import Error

# 命名規則のブロック列 (ループは "(c3a)x2" の形)
MODEL_NAMES = ["ccca", "cgga", "ccga", "cgca", "gcga", "c3a", "g3a", "c3wg3a", "c3ag3a", "c3wc3w", "g3wg3w", "cwga", "cgwa", "(c3a)x2", "(c3a)x2-kvs", "c-(c3w)x2-a", "a4-kvh4", "w", "g",
               "(c3a)x2-lora8", "(c3a)x2-lemb", "(c3a)x2-lora8-lemb", "c-(cgwa)x3-a-lora4-lemb-kvs", "(g3a)x2-lora8", "(c3w)x2-a-lemb",
               "(c3a)x2-rand1_3_8", "(c3a)x2-lora8-lemb-rand1_3_8", "c-(cgw)x1-a-lora4-lemb-kvs-rand2_4"]

# タグを試すときのブロック列 (c、g、a、w をすべて含む)
TAG_BASE = "c3wg3a"

# 命名規則のタグ
TAG_NAMES = ["relu2", "swiglu", "rms", "rmsp2", "dyt", "tie", "emb64", "tie-emb64", "gate", "qknorm", "sgate", "egate", "sandwich", "prope", "sima", "lpos", "mla12", "win128", "kvh4", "k3", "h8", "d48", "h8ad32", "ad32", "l512", "v8192"]

# 入力のトークン数 (Gated DeltaNet の 1 チャンク 64 より長くして、2 チャンク目と端の埋めも通す)
TOKENS = 70



def _test_build_hybrid_template() -> dict:
    '''
    #* config/hybrid.json の model.spec を辞書で返す (// のコメントは main.py と同じように消す)。
    '''
    
    text = open("config/hybrid.json", encoding="utf-8").read()
    text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*', lambda match: match.group(1) or "", text)
    return json.loads(text)["model"]["spec"]


def _test_build_hybrid_spec(name: str) -> dict:
    '''
    #* 命名規則の名前 (<ブロック列>-<タグ>...) から model.spec の辞書を作る。
    #! ブロック列は文字と繰り返しの数 (c3a = ccca)、ループは (ブロック列)x回数 で、ループの前後にも "-" でブロック列を置ける (c-(c3w)x2-a)。
    '''
    
    # ブロック列とループを読む (タグでない部分はブロック列)
    spec   = _test_build_hybrid_template()
    parts  = name.split("-")
    tags   = [part for part in parts if not re.fullmatch(r"\(?[cgaw0-9]+\)?(x[0-9]+)?", part)]
    blocks = [part for part in parts if part not in tags]
    mixers = ""
    for part in blocks:
        loop = re.fullmatch(r"\(([cgaw0-9]+)\)x([0-9]+)", part)
        text = "".join(letter * int(count or 1) for letter, count in re.findall(r"([cgaw])([0-9]*)", loop.group(1) if loop else part))
        if loop:
            spec["loop"].update(start=len(mixers), end=len(mixers) + len(text), times=int(loop.group(2)))
        mixers += text
    spec["mixers"] = mixers
    
    # タグごとに値を変える (注意の設定は a と w の両方に入れる。prope は LFM3 と同じく a だけ)
    attns = (spec["attn_global"], spec["attn_window"])
    for tag in tags:
        number = int(re.sub(r"[^0-9]", "", tag) or 0)
        if tag in ("relu2", "swiglu"):
            spec["ffn"]["type"] = tag
        elif tag in ("rms", "rmsp2", "dyt"):
            spec["norm"]["type"] = tag
        elif tag == "sandwich":
            spec["norm"]["sandwich"] = True
        elif tag == "tie":
            spec["embed"]["tie"] = True
        elif tag.startswith("emb"):
            spec["embed"]["rank"] = number
        elif tag == "lpos":
            spec["embed"]["pos"] = "learned"
            for attn in attns:
                attn["pos"]["type"] = "none"
        elif tag == "prope":
            spec["attn_global"]["pos"].update(part=0.5, theta=1000000.0)
        elif tag.startswith("win"):
            spec["attn_window"]["window"] = number
        elif tag.startswith("k") and tag[1:].isdigit():
            spec["conv"]["kernel"] = number
        elif tag == "h8":
            spec["gdn"]["heads"] = 8
        elif tag == "d48":
            spec["gdn"]["head_dim"] = 48
        elif tag.startswith("lora"):
            spec["loop"]["lora_rank"] = number
        elif tag == "lemb":
            spec["loop"]["embed"] = True
        elif tag.startswith("rand"):
            spec["loop"]["times_train"] = [int(item) for item in tag[4:].split("_")]
        elif tag in ("l512", "v8192"):
            spec["max_length" if tag[0] == "l" else "vocab_size"] = number
        else:
            for attn in attns:
                if tag in ("gate", "qknorm"):
                    attn["qk_norm"] = True
                if tag in ("gate", "sgate", "egate"):
                    attn["out_gate"] = "element" if tag == "egate" else "head"
                if tag == "sima":
                    attn["score"] = "sima"
                if tag == "kvs":
                    attn["kv"]["share"] = True
                if tag.startswith("mla"):
                    attn["kv"]["mla"] = {"enable": True, "latent": number}
                if tag.startswith("kvh"):
                    attn["kv_heads"] = number
                if tag in ("h8ad32", "ad32"):
                    attn.update(heads=8 if tag == "h8ad32" else attn["heads"], head_dim=32)
                assert tag in ("gate", "qknorm", "sgate", "egate", "sima", "kvs", "h8ad32", "ad32") or tag.startswith(("mla", "kvh")), f"unknown tag {tag}"
    return spec


def _test_build_hybrid_namespace(data: dict) -> SimpleNamespace:
    '''
    #* 辞書を、main.py が config を読んだときと同じ SimpleNamespace の入れ子にする。
    '''
    
    return json.loads(json.dumps(data), object_hook=lambda item: SimpleNamespace(**item))


def _test_build_hybrid_count(spec: dict) -> int:
    '''
    #* model.spec の辞書から、モデルのパラメータ数を式で求める (モデルのコードは使わない)。
    '''
    
    # 正規化 1 つの数 (LayerNorm は weight と bias、dyt は weight、bias、alpha)
    d     = spec["dim"]
    norm  = {"layer": 2 * d, "rms": d, "rmsp2": d, "dyt": 2 * d + 1}[spec["norm"]["type"]]
    ffn   = spec["ffn"]
    total = 0
    
    # ブロックごとの数
    for mixer in spec["mixers"]:
        total += norm * (4 if spec["norm"]["sandwich"] else 2) + d * ffn["hidden"] * (2 if ffn["type"] == "swiglu" else 1) + ffn["hidden"] * d
        if mixer == "c":
            total += 3 * d * d + d * spec["conv"]["kernel"] + d * d
        elif mixer == "g":
            gdn    = spec["gdn"]
            width  = gdn["heads"] * gdn["head_dim"]
            total += d * (4 * width + 2 * gdn["heads"]) + 3 * width * gdn["kernel"] + width * d + gdn["head_dim"] + 2 * gdn["heads"]
        else:
            attn   = spec["attn_global" if mixer == "a" else "attn_window"]
            q      = attn["heads"] * attn["head_dim"]
            kv     = 2 * attn["kv_heads"] * attn["head_dim"]
            mla    = attn["kv"]["mla"]
            gate   = {"none": 0, "head": attn["heads"], "element": q}[attn["out_gate"]]
            total += d * (q + (mla["latent"] if mla["enable"] else kv)) + (mla["latent"] * kv if mla["enable"] else 0) + q * d + (2 * attn["head_dim"] if attn["qk_norm"] else 0) + d * gate
    
    # ループ埋め込み (周回ごとに dim) と LoRA (周回 x ループのブロックごとに dim x rank + rank x qkv の出力の幅)
    loop   = spec["loop"]
    widths = {"c": 3 * d, "g": 4 * spec["gdn"]["heads"] * spec["gdn"]["head_dim"] + 2 * spec["gdn"]["heads"]}
    for mixer in "aw":
        attn          = spec["attn_global" if mixer == "a" else "attn_window"]
        widths[mixer] = attn["heads"] * attn["head_dim"] + (attn["kv"]["mla"]["latent"] if attn["kv"]["mla"]["enable"] else 2 * attn["kv_heads"] * attn["head_dim"])
    turns  = max([loop["times"], *loop["times_train"]])
    total += turns * d if loop["embed"] else 0
    total += turns * sum(d * loop["lora_rank"] + loop["lora_rank"] * widths[mixer] for mixer in spec["mixers"][loop["start"]:loop["end"]])
    
    # 埋め込み、位置の表、最後の正規化、出力層
    embed  = spec["embed"]
    rank   = embed["rank"]
    total += spec["vocab_size"] * (rank or d) + rank * d + (spec["max_length"] * d if embed["pos"] == "learned" else 0) + norm
    total += 0 if embed["tie"] else (rank or d) * spec["vocab_size"] + rank * d
    return total


def _test_build_hybrid_run(spec: dict) -> HybridLm:
    '''
    #* spec からモデルを作り、forward と backward を通して、出力の形、値、勾配を確かめる。
    #! 周回数は times と times_train の一番大きい数にして、周回ごとの重みがすべて使われるようにする。
    '''
    
    # forward
    torch.manual_seed(0)
    model  = build_model("hybrid", None, _test_build_hybrid_namespace(spec))
    ids    = torch.randint(0, spec["vocab_size"], (2, TOKENS))
    turns  = max([spec["loop"]["times"], *spec["loop"]["times_train"]])
    logits = model(ids, torch.ones_like(ids), times=turns)
    assert isinstance(model, HybridLm)
    assert logits.shape == (2, TOKENS, spec["vocab_size"])
    assert torch.isfinite(logits).all()
    
    # backward (すべてのパラメータに 0 でない勾配が来る。LoRA の down は up の初期値が 0 なので勾配も 0)
    torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, spec["vocab_size"]), ids[:, 1:].reshape(-1)).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0 or name.startswith("lora_down"), name
    return model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_hybrid_models(name: str) -> None:
    '''
    #* 命名規則のブロック列のモデルが組み立てられ、forward と backward が通り、パラメータ数が式と合う。
    '''
    
    spec  = _test_build_hybrid_spec(name)
    model = _test_build_hybrid_run(spec)
    assert len(model.blocks) == len(spec["mixers"])
    assert [block.mixer for block in model.blocks] == list(spec["mixers"])
    assert sum(parameter.numel() for parameter in model.parameters()) == _test_build_hybrid_count(spec)
    return None


@pytest.mark.parametrize("tag", TAG_NAMES)
def test_build_hybrid_tags(tag: str) -> None:
    '''
    #* 各タグを付けたモデルが組み立てられ、forward と backward が通り、パラメータ数が式と合う。タグは標準からの変更として本当に入っている。
    '''
    
    spec  = _test_build_hybrid_spec(f"{TAG_BASE}-{tag}")
    model = _test_build_hybrid_run(spec)
    assert spec != _test_build_hybrid_spec(TAG_BASE)
    assert sum(parameter.numel() for parameter in model.parameters()) == _test_build_hybrid_count(spec)
    return None


def test_build_hybrid_count_c3a() -> None:
    '''
    #* 標準モデル c3a のパラメータ数を、手で数えた値と比べる (式そのものの確認)。
    '''
    
    # ブロック: 正規化 2 x 512、MLP 2 x 256 x 1024、c は 4 x 256^2 + 256 x 7、a は 256 x (256 + 128) + 256^2
    block_c = 2 * 512 + 2 * 256 * 1024 + 4 * 256 * 256 + 256 * 7
    block_a = 2 * 512 + 2 * 256 * 1024 + 256 * 384 + 256 * 256
    lm      = 2 * 4096 * 256 + 512
    model   = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("c3a")))
    assert sum(parameter.numel() for parameter in model.parameters()) == 3 * block_c + block_a + lm == 5_154_560
    return None


def test_build_hybrid_window() -> None:
    '''
    #* "w" のブロックは直近 window トークンだけを、"a" のブロックは前の全トークンを見る (入力を 1 つ変えて、出力が変わる位置を調べる)。
    '''
    
    for mixer, score in (("w", "softmax"), ("w", "sima"), ("a", "softmax"), ("a", "sima")):
        # window 5 の 1 ブロックのモデル
        spec                         = _test_build_hybrid_spec(mixer)
        spec["attn_window"]["window"] = 5
        spec["attn_window"]["score"]  = score
        spec["attn_global"]["score"]  = score
        torch.manual_seed(0)
        model = build_model("hybrid", None, _test_build_hybrid_namespace(spec)).eval()
        ids   = torch.randint(0, 4096, (1, 20))
        
        # 位置 8 のトークンを変えて、出力が変わる位置を集める
        changed      = ids.clone()
        changed[0, 8] = (changed[0, 8] + 1) % 4096
        with torch.no_grad():
            diff = (model(ids, ids) - model(changed, changed)).abs().amax(dim=-1)[0]
        moved    = [position for position in range(20) if diff[position] > 1e-6]
        expected = list(range(8, 13)) if mixer == "w" else list(range(8, 20))
        assert moved == expected, (mixer, score, moved)
    return None


def test_build_hybrid_parts() -> None:
    '''
    #* tie、lpos、prope、kvs、ループ埋め込み、LoRA が、モデルの中に本当に入っている。
    '''
    
    # tie: 出力層と埋め込みが同じ重み
    model = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("c3a-tie-emb64")))
    assert isinstance(model, HybridLm) and model.head.weight is model.embed.weight and model.head_down is None
    
    # prope: a の表は後ろ半分が回らず (cos 1、sin 0)、w の表は全部回る
    model = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("c3wg3a-prope")))
    assert isinstance(model, HybridLm)
    assert torch.equal(model.rope_global_cos[:, 16:], torch.ones(1024, 16)) and torch.equal(model.rope_global_sin[:, 16:], torch.zeros(1024, 16))
    assert not torch.allclose(model.rope_global_cos[1:, :16], torch.ones(1023, 16))
    assert torch.allclose(model.rope_global_sin[5, 1], torch.tensor(5 * 1000000.0 ** (-1 / 32)).sin())
    assert torch.allclose(model.rope_window_sin[5, 31], torch.tensor(5 * 10000.0 ** (-31 / 32)).sin())
    
    # kvs: times 1 では kvs の有無で同じ出力、times 2 では違う出力になる
    ids     = torch.randint(0, 4096, (1, 30))
    outputs = {}
    for name in ("(c3a)x1", "(c3a)x1-kvs", "(c3a)x2", "(c3a)x2-kvs"):
        torch.manual_seed(0)
        with torch.no_grad():
            outputs[name] = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec(name)))(ids, ids)
    assert torch.equal(outputs["(c3a)x1"], outputs["(c3a)x1-kvs"])
    assert not torch.allclose(outputs["(c3a)x2"], outputs["(c3a)x2-kvs"])
    assert not torch.allclose(outputs["(c3a)x1"], outputs["(c3a)x2"])
    
    # ループ埋め込みと LoRA: 同じ重みを読み込むと、初期値では出力を変えず、重みを変えると出力が変わる。周回ごとに別の重みを持つ
    plain = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("(c3a)x2")))
    model = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("(c3a)x2-lora8-lemb")))
    assert isinstance(model, HybridLm) and model.loop_embed is not None
    assert len(model.lora_down) == 8 and model.loop_embed.weight.shape == (2, 256)
    assert sorted(model.load_state_dict(plain.state_dict(), strict=False).missing_keys) == sorted(name for name in model.state_dict() if name.startswith(("lora", "loop_embed")))
    with torch.no_grad():
        assert torch.allclose(plain(ids, ids), model(ids, ids))
        cast_up = model.lora_up[5]
        assert isinstance(cast_up, torch.nn.Linear)
        cast_up.weight.normal_()
        changed = model(ids, ids)
        assert not torch.allclose(plain(ids, ids), changed)
        model.loop_embed.weight[1].add_(1.0)
        assert not torch.allclose(changed, model(ids, ids))
    
    # lpos: 位置の表を変えると出力が変わり、RoPE は使わない
    spec  = _test_build_hybrid_spec("c3a-lpos")
    model = build_model("hybrid", None, _test_build_hybrid_namespace(spec))
    assert isinstance(model, HybridLm) and model.position is not None
    with torch.no_grad():
        before = model(ids, ids)
        model.position.weight.add_(1.0)
        assert not torch.allclose(before, model(ids, ids))
    return None


@pytest.mark.parametrize("change, code", [
    (lambda spec: spec.update(extra=1)                               , 40),
    (lambda spec: spec["attn_window"].pop("window")                  , 40),
    (lambda spec: spec["embed"].update(tie=1)                        , 42),
    (lambda spec: spec.update(dim="256")                             , 42),
    (lambda spec: spec.update(dim=256.0)                             , 42),
    (lambda spec: spec.update(conv=7)                                , 42),
    (lambda spec: spec["ffn"].update(type="silu")                    , 41),
    (lambda spec: spec["norm"].update(type="batch")                  , 41),
    (lambda spec: spec.update(mixers="ccxa")                         , 41),
    (lambda spec: spec.update(mixers="")                             , 41),
    (lambda spec: spec["attn_global"].update(heads=4, kv_heads=3)    , 41),
    (lambda spec: spec["attn_global"].update(head_dim=33)            , 41),
    (lambda spec: spec["attn_global"].update(window=256)             , 41),
    (lambda spec: spec["attn_window"].update(window=0)               , 41),
    (lambda spec: spec["attn_window"]["pos"].update(part=1.5)        , 41),
    (lambda spec: spec["attn_global"]["kv"]["mla"].update(enable=True), 41),
    (lambda spec: spec["loop"].update(start=2, end=5, times=2)       , 41),
    (lambda spec: spec["loop"].update(start=0, end=4, times=0)       , 41),
    (lambda spec: spec["gdn"].update(heads=0)                        , 41),
    (lambda spec: spec["loop"].update(embed=True)                    , 41),
    (lambda spec: spec["loop"].update(lora_rank=4)                   , 41),
    (lambda spec: spec["loop"].update(end=4, lora_rank=-1)           , 41),
    (lambda spec: spec["loop"].update(embed="true")                  , 42),
    (lambda spec: spec["loop"].update(times_train=[1, 3])            , 41),
    (lambda spec: spec["loop"].update(end=4, times_train=[0, 3])     , 41),
    (lambda spec: spec["loop"].update(end=4, times_train=3)          , 42),
    (lambda spec: spec["loop"].update(end=4, times_train=[1, 2.0])   , 42),
    (lambda spec: spec["loop"].update(end=4, times_train=[True])     , 42),
])
def test_build_hybrid_errors(change, code: int) -> None:
    '''
    #* 知らないキー、無いキー、型の違い、選べない値、範囲の外の値は、決まった番号のエラーになる。
    '''
    
    spec = copy.deepcopy(_test_build_hybrid_spec("c3a"))
    change(spec)
    with pytest.raises(Error) as caught:
        build_model("hybrid", None, _test_build_hybrid_namespace(spec))
    assert caught.value.code == code
    return None


def test_build_hybrid_float_from_int() -> None:
    '''
    #* float の項目 (theta) には int も書ける。spec が無いときと知らないモデル名はエラーになる。
    '''
    
    spec                                = _test_build_hybrid_spec("c3a")
    spec["attn_global"]["pos"]["theta"] = 10000
    model                               = build_model("hybrid", None, _test_build_hybrid_namespace(spec))
    assert isinstance(model, HybridLm) and model.spec.attn_global.pos.theta == 10000.0 and isinstance(model.spec.attn_global.pos.theta, float)
    with pytest.raises(Error) as caught:
        build_model("hybrid", None, None)
    assert caught.value.code == 40
    with pytest.raises(Error) as caught:
        build_model("c3a", None, _test_build_hybrid_namespace(spec))
    assert caught.value.code == 6
    return None


def test_build_hybrid_sima() -> None:
    '''
    #* SimA の注意ブロックの出力を、トークンごとの足し算で素直に求めた値と比べる (window 3 の w と、全体の a)。
    '''
    
    for mixer in ("a", "w"):
        # RoPE なし、kv_heads 2 の SimA の 1 ブロック
        spec = _test_build_hybrid_spec(f"{mixer}-sima-kvh2-win3")
        for attn in (spec["attn_global"], spec["attn_window"]):
            attn["pos"]["type"] = "none"
        torch.manual_seed(0)
        model = build_model("hybrid", None, _test_build_hybrid_namespace(spec))
        assert isinstance(model, HybridLm)
        block = cast(HybridBlock, model.blocks[0])
        x     = torch.randn(1, 9, 256)
        
        # q、k、v を作り、トークンごとに L1 正規化する
        with torch.no_grad():
            qkv = block.qkv(block.norm1(x))[0]
            q   = qkv[:, :256].view(9, 4, 64)
            k   = qkv[:, 256:384].view(9, 2, 64)
            v   = qkv[:, 384:].view(9, 2, 64)
            q   = q / q.abs().sum(dim=-1, keepdim=True)
            k   = k / k.abs().sum(dim=-1, keepdim=True)
            
            # 位置 t の出力 = 見えるトークン j について (q_t . k_j / 8) v_j を足し、見えるトークンの数で割る (頭 0, 1 は kv 0、頭 2, 3 は kv 1)
            out = torch.zeros(9, 4, 64)
            for t in range(9):
                seen = [j for j in range(t + 1) if mixer == "a" or j > t - 3]
                for head in range(4):
                    for j in seen:
                        out[t, head] += float(q[t, head] @ k[j, head // 2]) / 8 * v[j, head // 2]
                    out[t, head] /= len(seen)
            
            # 残差と MLP を足して、ブロックの出力と比べる
            y        = x[0] + block.proj(out.reshape(9, 256))
            expected = y + block.out(torch.nn.functional.gelu(block.fc(block.norm2(y))))
            assert torch.allclose(block(x, None, {}, None)[0], expected, atol=1e-5)
    return None


def test_build_hybrid_times() -> None:
    '''
    #* forward の times で周回数を変えられる。times を渡さなければ loop.times で通り、周回ごとのループ埋め込みと LoRA は周回の番号で選ばれる。
    '''
    
    # ループ埋め込みと LoRA なし: times 3 で通すと、同じ重みの (c3a)x3 と同じ出力。times なしは (c3a)x2 と同じ
    ids   = torch.randint(0, 4096, (1, 30))
    model = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("(c3a)x2-rand1_3")))
    three = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("(c3a)x3")))
    two   = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("(c3a)x2")))
    three.load_state_dict(model.state_dict())
    two.load_state_dict(model.state_dict())
    assert isinstance(model, HybridLm) and model.loop_times_train == (1, 3)
    with torch.no_grad():
        assert torch.equal(model(ids, ids, times=3), three(ids, ids))
        assert torch.equal(model(ids, ids), two(ids, ids))
        assert not torch.allclose(model(ids, ids, times=1), two(ids, ids))
    
    # ループ埋め込みと LoRA あり: 一番大きい 8 周まで用意し、times 3 の出力は 4 周目以降の重みを変えても変わらず、3 周目の重みを変えると変わる
    model = build_model("hybrid", None, _test_build_hybrid_namespace(_test_build_hybrid_spec("(c3a)x2-lora4-lemb-rand1_3_8")))
    assert isinstance(model, HybridLm) and model.loop_embed is not None
    assert model.loop_embed.weight.shape == (8, 256) and len(model.lora_down) == 8 * 4
    with torch.no_grad():
        for up in model.lora_up:
            assert isinstance(up, torch.nn.Linear)
            up.weight.normal_(std=0.1)
        before = model(ids, ids, times=3)
        model.loop_embed.weight[3:].add_(1.0)
        for up in model.lora_up[12:]:
            assert isinstance(up, torch.nn.Linear)
            up.weight.add_(1.0)
        assert torch.equal(before, model(ids, ids, times=3))
        model.loop_embed.weight[2].add_(1.0)
        assert not torch.allclose(before, model(ids, ids, times=3))
        assert model(ids, ids, times=8).shape == before.shape
    return None
