# @file types.py
# @brief hybrid モデルの形を表す設定のまとまり

from dataclasses import dataclass
from typing      import Literal

# ブロックのトークンミキサー: c = ゲート付き短畳み込み (LFM2)、g = Gated DeltaNet、a = 全トークンへの因果注意、w = 直近 window トークンへの因果注意
type MixerName = Literal["c", "g", "a", "w"]

# 使えるミキサーの文字
MIXER_LETTERS = "cgaw"



@dataclass(frozen=True)
class EmbedSpec:
    '''
    #* 埋め込みと出力層の設定を持つ。
    #! rank が 0 より大きいと、埋め込みと出力層を 2 段 (vocab x rank と rank x dim) に分ける (EmbBERT、Yokoyama と同じ)。0 なら 1 段。
    #! tie が true だと、出力層は埋め込みの重みを使い回す (rank があるときは 2 段とも使い回す)。
    #! pos が "learned" だと、学習する位置の表 (max_length x dim) を埋め込みに足す。"none" なら足さない。
    '''
    
    tie : bool
    rank: int
    pos : Literal["none", "learned"]


@dataclass(frozen=True)
class PosSpec:
    '''
    #* 注意で q と k に掛ける位置の情報 (RoPE) の設定を持つ。
    #! type が "none" だと回転しない (NoPE)。theta は RoPE の底。
    #! part は回す次元の割合 (1.0 で全部、LFM3 の部分 RoPE は 0.5)。残りの次元は角度 0 のまま回さない。
    '''
    
    type : Literal["rope", "none"]
    theta: float
    part : float


@dataclass(frozen=True)
class MlaSpec:
    '''
    #* MLA (k と v を小さな潜在ベクトル 1 本から作る方式) の設定を持つ。
    #! enable が true のとき、入力から latent 次元のベクトルを作り、それを広げて k と v にする。KV キャッシュには潜在ベクトルだけを置けばよい。
    '''
    
    enable: bool
    latent: int


@dataclass(frozen=True)
class KvSpec:
    '''
    #* k と v の作り方と持ち方の設定を持つ。
    #! share が true だと、ループの中の注意ブロックは 1 周目の k と v を 2 周目以降も使う。KV キャッシュがループの回数で増えない (kvs)。
    '''
    
    share: bool
    mla  : MlaSpec


@dataclass(frozen=True)
class AttnSpec:
    '''
    #* 注意ブロック ("a" か "w") の設定を持つ。
    #! kv_heads は heads を割り切る数にする (GQA)。head_dim は RoPE のために偶数にする。
    #! window は "w" の注意だけが使う (直近 window トークンだけを見る)。"a" の設定では 0。
    #! qk_norm は q と k に頭ごとの RMSNorm を掛ける。score は "softmax" か "sima" (softmax を使わない SimA)。
    #! out_gate は注意の出力に掛ける sigmoid のゲート。"head" は頭ごとに 1 つ、"element" は出力の値ごとに 1 つ、"none" は無し。
    '''
    
    heads   : int
    head_dim: int
    kv_heads: int
    window  : int
    pos     : PosSpec
    qk_norm : bool
    score   : Literal["softmax", "sima"]
    out_gate: Literal["none", "head", "element"]
    kv      : KvSpec


@dataclass(frozen=True)
class ConvSpec:
    '''
    #* "c" ブロック (ゲート付き短畳み込み) の設定を持つ。
    '''
    
    kernel: int


@dataclass(frozen=True)
class GdnSpec:
    '''
    #* "g" ブロック (Gated DeltaNet) の設定を持つ。
    #! 状態の大きさは heads x head_dim x head_dim。kernel は q、k、v に掛ける短い畳み込みの幅。
    '''
    
    heads   : int
    head_dim: int
    kernel  : int


@dataclass(frozen=True)
class FfnSpec:
    '''
    #* 各ブロックの MLP の設定を持つ。
    #! swiglu はゲートと up を 1 つの行列にまとめるので、最初の層の出力は 2 x hidden になる。
    '''
    
    type  : Literal["gelu", "relu2", "swiglu"]
    hidden: int


@dataclass(frozen=True)
class NormSpec:
    '''
    #* 正規化の設定を持つ。
    #! type は layer (LayerNorm)、rms (RMSNorm)、rmsp2 (1 / rms を 2 の累乗に丸めた RMSNorm)、dyt (Dynamic Tanh)。
    #! sandwich が true だと、ミキサーと MLP の出力にも正規化を掛ける (Peri-LN)。
    '''
    
    type    : Literal["layer", "rms", "rmsp2", "dyt"]
    sandwich: bool


@dataclass(frozen=True)
class LoopSpec:
    '''
    #* ループの設定を持つ。start から end - 1 までのブロックを、同じ重みのまま times 回くり返す (学習でも評価でも、回数は固定)。
    #! times が 1 で embed が false、lora_rank が 0 なら、各ブロックを 1 回ずつ通すだけで、ループなしと同じ。
    #! embed が true だと、各周回に入る前に、その周回の学習するベクトル (幅 dim、初期値 0) を全トークンに足す (ループ埋め込み、Glint-2)。
    #! lora_rank が 0 より大きいと、各周回の各ブロックの qkv の出力に、その周回だけの低ランクの差分 (rank lora_rank、up の初期値 0) を足す (Glint-2)。
    #! times_train が空でなければ、学習では重みの更新ごとに、この候補から周回数をランダムに選ぶ (同じ数を何度も書くと選ばれやすくなる)。評価は times で行い、候補の数ごとの PPL も測る。空なら学習も times で行う。
    #! ループ埋め込みと LoRA は、times と times_train の一番大きい数の周回まで用意する。
    '''
    
    start      : int
    end        : int
    times      : int
    embed      : bool
    lora_rank  : int
    times_train: tuple[int, ...]


@dataclass(frozen=True)
class HybridSpec:
    '''
    #* hybrid モデル 1 つの形をすべて持つ。mixers の 1 文字が 1 ブロックになり、その順に通す。
    #? config.model.spec から build_hybrid_model() (build_hybrid.py) が作り、HybridLm (hybrid_lm.py) に渡す。
    #! 値はすべて設定ファイルに書く。省略したときの値は持たない。
    '''
    
    mixers     : str
    dim        : int
    vocab_size : int
    max_length : int
    embed      : EmbedSpec
    attn_global: AttnSpec
    attn_window: AttnSpec
    conv       : ConvSpec
    gdn        : GdnSpec
    ffn        : FfnSpec
    norm       : NormSpec
    loop       : LoopSpec
