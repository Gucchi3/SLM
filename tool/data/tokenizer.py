# @file tokenizer.py
# @brief トークナイザーの種類から、学習前のトークナイザーとその学習器を作る

##########################################################
'''
トークナイザーの種類の追加
1. 学習前の Tokenizer とその学習器を返す build_xxx_tokenizer(vocab_size, special_tokens) を書く。
2. TOKENIZER_REGISTRY に "名前: build_xxx_tokenizer" を足し、TokenizerType にその名前を足す。
3. データセットのスクリプトの tokenizer の段階に --tokenizer_type 名前 を渡す。
'''
##########################################################

from collections.abc import Callable
from tokenizers      import Tokenizer, decoders, models, pre_tokenizers, trainers
from typing          import Literal

TokenizerType    = Literal["bytelevel_bpe"]
TokenizerBuilder = Callable[[int, list[str]], tuple[Tokenizer, trainers.Trainer]]



def build_bytelevel_bpe_tokenizer(vocab_size: int, special_tokens: list[str]) -> tuple[Tokenizer, trainers.Trainer]:
    '''
    #* バイト単位の BPE トークナイザーとその学習器を作る。英語のように単語の間に空白がある言語向け。
    #! 基本の語彙に 256 バイトすべてがあるので、どんなテキストもエンコードできる。
    #! BPE の前に数字を 1 桁ずつに分けるので、数は必ず 1 桁ずつになる (123 -> 1 2 3)。算数のために桁の位置をそろえるため。
    #! 日本語には別の種類を足す。この種類は空白の無い長いテキストを BPE の前に 1 かたまりにしてしまい、うまく働かない。
    '''
    
    tokenizer               = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tokenizer.decoder       = decoders.ByteLevel()
    trainer                 = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    return tokenizer, trainer


TOKENIZER_REGISTRY: dict[str, TokenizerBuilder] = {
    "bytelevel_bpe": build_bytelevel_bpe_tokenizer,
}


def build_tokenizer(tokenizer_type: TokenizerType, vocab_size: int, special_tokens: list[str]) -> tuple[Tokenizer, trainers.Trainer]:
    '''
    #* tokenizer_type の学習前のトークナイザーと学習器を返す。
    #? corpus.py の train_corpus_tokenizer() から呼ぶ。同じファイルの run_corpus() が、TOKENIZER_REGISTRY にある名前だけを通す。
    '''
    
    return TOKENIZER_REGISTRY[tokenizer_type](vocab_size, special_tokens)
