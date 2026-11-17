# @file benchmark.py
# @brief Tiny-ML Leaderboard (BLiMP、ARC-Easy、WikiText-2) と Open SLM Leaderboard (HellaSwag、ARC-Easy、ARC-Challenge、PIQA、ArithMark-3、Int Index) のテストで、各リーダーボードの決まりどおりにモデルを測る

##########################################################
'''
ベンチマークの追加
1. スコアの辞書を返す benchmark_xxx(model, tokenizer, device, data_dir, max_length, batch_size) を書く。
2. BENCHMARK_REGISTRY に "名前: (リーダーボード, benchmark_xxx)" を足す。スコアは表の中で、そのリーダーボードの名前の下に表示される。
3. validator.benchmark.tasks にその名前を足す。
'''
##########################################################

from collections.abc import Callable
from datetime        import datetime
from huggingface_hub import hf_hub_download
from pathlib         import Path
from rich.console    import Console
from rich.table      import Table
from rich.text       import Text
from tokenizers      import Tokenizer
from torch           import nn
from types           import SimpleNamespace

import json
import math
import pyarrow.parquet as pq
import re
import torch

from utils.exceptions import Error
from .load            import load_model
from .score           import score_continuations, score_texts

BenchmarkFunction = Callable[[nn.Module, Tokenizer, torch.device, Path, int, int], dict[str, float]]

# BLiMP の 67 のテスト (それぞれ 1,000 の文の組)。Glint-1.3 の benchmark.py と同じ
BLIMP_CONFIGS = (
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement", "animate_subject_passive", "animate_subject_trans", "causative", "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch", "coordinate_structure_constraint_object_extraction", "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2", "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1", "distractor_agreement_relational_noun", "distractor_agreement_relative_clause", "drop_argument",
    "ellipsis_n_bar_1", "ellipsis_n_bar_2", "existential_there_object_raising", "existential_there_quantifiers_1", "existential_there_quantifiers_2", "existential_there_subject_raising",
    "expletive_it_object_raising", "inchoative", "intransitive", "irregular_past_participle_adjectives", "irregular_past_participle_verbs", "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2", "left_branch_island_echo_question", "left_branch_island_simple_question", "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope", "passive_1", "passive_2", "principle_A_c_command", "principle_A_case_1", "principle_A_case_2", "principle_A_domain_1", "principle_A_domain_2",
    "principle_A_domain_3", "principle_A_reconstruction", "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2", "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope", "sentential_subject_island", "superlative_quantifiers_1", "superlative_quantifiers_2", "tough_vs_raising_1", "tough_vs_raising_2", "transitive", "wh_island",
    "wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
)

# Hugging Face Hub の AxiomicLabs/Arithmark-3.0 のコミット。Open SLM Leaderboard が使う arithmark-3.jsonl (1,000 問) を含む
ARITHMARK3_REVISION = "6f6e59dd9b7e2c63455f7af7f838f9ecc3d0a746"

# Open SLM Leaderboard の Int Index (Intelligence Index) の項目: (名前, でたらめに選んだときのスコア (%), 重み)。順番はリーダーボードのページの getIntelligenceIndex() と同じ
# "arc" は arc_easy と arc_challenge の平均
INT_INDEX_PARTS = (
    ("hellaswag" , 25.0, 1.0 ),
    ("arc"       , 25.0, 1.0 ),
    ("piqa"      , 50.0, 1.0 ),
    ("arithmark3", 25.0, 0.65),
)



def _benchmark_download(repo: str, paths: list[str], folder: Path, revision: str | None = None) -> list[Path]:
    '''
    #* Hugging Face Hub のデータセットのファイルを folder にダウンロードし、手元のパスを返す。
    #? benchmark_xxx() の関数が使う。
    #! folder にすでにあるファイルはダウンロードし直さないので、2 回目からはネットワーク無しでテストできる。
    #! revision はデータセットのコミットを固定する。None なら、lm-evaluation-harness と同じく一番新しいファイルをダウンロードする。
    '''
    
    for path in paths:
        if not (folder / path).exists():
            try:
                hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=folder)
            except OSError as error:
                raise Error("Failed to download the benchmark data.", detail=f"{repo}/{path}: {error}", hint="Check the network, then run the benchmark again.", code=36)
    return [folder / path for path in paths]


def _benchmark_score(model: nn.Module, device: torch.device, batch_size: int, name: str, questions: list[list[tuple[list[int], list[int], str]]], size: int) -> tuple[list[list[float]], list[int]]:
    '''
    #* 評価した各問題の選択肢の対数尤度と、元の番号を返す。
    #? _benchmark_choices() と benchmark_arithmark3() が使う。size は lm-evaluation-harness では data.max_length + 1、bencharithmark-3.py では data.max_length。
    #! どれかの選択肢が size - 1 トークンを超えると、その問題は丸ごと飛ばす。そうでなければ、文脈の先頭のトークンだけを切る。どちらの数も表示する。
    #! 文脈や選択肢が空、size < 2、評価できる問題が無いときは、エラー 38 で止める。
    '''
    
    requests = list[tuple[list[int], int]]()
    indices  = list[int]()
    counts   = list[int]()
    cut      = 0
    for index, pairs in enumerate(questions):
        for context_ids, choice_ids, choice in pairs:
            if not context_ids or not choice_ids or size < 2:
                raise Error("A choice cannot be scored.", detail=f"{name}: the context has {len(context_ids)} tokens, and the choice {choice!r} has {len(choice_ids)} tokens (requests of up to {size} tokens).",
                            hint="The context and the choice need 1 or more tokens each. Check the data and data.max_length, or remove the test from validator.benchmark.tasks.", code=38)
        if any(len(choice_ids) >= size for _, choice_ids, _ in pairs):
            continue
        indices.append(index)
        counts.append(len(pairs))
        for context_ids, choice_ids, _ in pairs:
            cut += len(context_ids) + len(choice_ids) > size
            requests.append(((context_ids + choice_ids)[-size:], len(choice_ids)))
    skipped = len(questions) - len(indices)
    if skipped > 0:
        Console().print(Text(f"[Benchmark] {name}: skipped {skipped:,} of {len(questions):,} questions because a choice exceeds {size - 1:,} tokens; evaluated {len(indices):,} questions.", style="yellow"))
    if not indices:
        raise Error("No benchmark questions can be evaluated.", detail=f"{name}: {skipped:,} of {len(questions):,} questions were skipped.",
                    hint="Increase data.max_length or check the benchmark data.", code=38)
    if cut > 0:
        Console().print(Text(f"[Benchmark] {name}: {cut:,} of {len(requests):,} texts have more than {size:,} tokens, so their first tokens were cut.", style="yellow"))
    
    scores = iter(score_continuations(model, requests, device, batch_size))
    return [[next(scores) for _ in range(count)] for count in counts], indices


def _benchmark_choices(model: nn.Module, tokenizer: Tokenizer, device: torch.device, max_length: int, batch_size: int, name: str, questions: list[tuple[str, list[str], int]]) -> dict[str, float]:
    '''
    #* (文脈, 選択肢, 答え) の各問題に lm-evaluation-harness の決まり (multiple_choice、zero-shot) で答え、長さで正規化した正解率と、しない正解率 (%) を返す。
    #? benchmark_hellaswag()、benchmark_arc_easy()、benchmark_arc_challenge()、benchmark_piqa() が使う。決まりは lm-evaluation-harness の _encode_pair() と _loglikelihood_tokens() に従う。
    #! 各選択肢は、文脈の後の " " + 選択肢 として点を付ける。文脈の最後の空白は選択肢の先頭へ移す。選択肢のトークンは、文脈 + " " + 選択肢 のトークンのうち、文脈だけのトークン数より後ろの部分。
    #! 文脈と選択肢が max_length + 1 トークンを超えるときは、文脈の先頭のトークンを切る。どれかの選択肢が max_length トークンを超えると、その問題は丸ごと飛ばし、正解率の分母からも外す。
    #! name には acc_norm (リーダーボードの数字。対数尤度 / 選択肢の文字数 が一番大きいもの) を、name_acc には acc (対数尤度が一番大きいもの) を入れる。同点なら最初の選択肢を選ぶ。
    '''
    
    # 各選択肢のトークンを、その文脈の後ろに作る
    pairs = list[list[tuple[list[int], list[int], str]]]()
    for context, choices, _ in questions:
        context_ids  = tokenizer.encode(context.rstrip()).ids
        pairs.append([(context_ids, tokenizer.encode(context + " " + choice).ids[len(context_ids):], choice) for choice in choices])
    
    # 対数尤度が一番大きい選択肢を、選択肢の文字数で割る場合と割らない場合で選ぶ
    scores, indices = _benchmark_score(model, device, batch_size, name, pairs, max_length + 1)
    acc, acc_norm   = 0, 0
    for index, totals in zip(indices, scores):
        _, choices, answer = questions[index]
        means     = [total / len(choice) for total, choice in zip(totals, choices)]
        acc      += max(range(len(totals)), key=totals.__getitem__) == answer
        acc_norm += max(range(len(means)), key=means.__getitem__) == answer
    return {name: 100 * acc_norm / len(indices), f"{name}_acc": 100 * acc / len(indices), f"{name}_questions": len(indices), f"{name}_skipped": len(questions) - len(indices)}


def _benchmark_hellaswag_text(text: str) -> str:
    '''
    #* HellaSwag のテキストを、lm-evaluation-harness の tasks/hellaswag/utils.py の preprocess() と同じように整える。
    #? benchmark_hellaswag() が使う。
    #! " [title]" は ". " にし、ほかの角かっこの語 (WikiHow の部分の印) は消し、2 つ続いた空白は 1 つにする (1 回だけ)。
    '''
    
    text = text.strip().replace(" [title]", ". ")
    return re.sub("\\[.*?\\]", "", text).replace("  ", " ")


def benchmark_blimp(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* BLiMP の正解率 (%) を返す: 67,000 の文の組のうち、文法的に正しい文の対数確率の合計が大きかった組の割合。
    #! 各文は max_length トークンで切る。同点は不正解として数える。
    '''
    
    # 67 のテストの文の組を読む
    files = _benchmark_download("nyu-mll/blimp", [f"{name}/train-00000-of-00001.parquet" for name in BLIMP_CONFIGS], data_dir / "blimp")
    pairs = [(row["sentence_good"], row["sentence_bad"]) for path in files for row in pq.read_table(path, columns=["sentence_good", "sentence_bad"]).to_pylist()]
    
    # 各組の 2 つの文に点を付け、文法的に正しい文の方が確率が高い組を数える
    good = score_texts(model, [tokenizer.encode(sentence).ids[:max_length] for sentence, _ in pairs], device, batch_size)
    bad  = score_texts(model, [tokenizer.encode(sentence).ids[:max_length] for _, sentence in pairs], device, batch_size)
    return {"blimp": 100 * sum(good_score > bad_score for good_score, bad_score in zip(good, bad)) / len(pairs), "blimp_pairs": len(pairs)}


def benchmark_arc_easy_tinyml(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Tiny-ML Leaderboard の決まりで、ARC-Easy の 2,376 のテスト問題の正解率 (%) を返す。各問題では log P(問題 + " " + 選択肢) - log P(問題) が一番大きい選択肢を選ぶ。
    #! 点は長さで割らず、"Question:" のようなプロンプトの形も足さない。各テキストは max_length トークンで切る。同点なら最初の選択肢を選ぶ。
    #! Open SLM Leaderboard は ARC-Easy に別の決まりを使うので、そのスコアは benchmark_arc_easy() で求める。
    '''
    
    # 各問題を単独で、また各選択肢付きで、すべて同じバッチで点を付ける
    files  = _benchmark_download("allenai/ai2_arc", ["ARC-Easy/test-00000-of-00001.parquet"], data_dir / "arc")
    rows   = pq.read_table(files[0]).to_pylist()
    texts  = [text for row in rows for text in (row["question"], *(row["question"] + " " + choice for choice in row["choices"]["text"]))]
    scores = iter(score_texts(model, [tokenizer.encode(text).ids[:max_length] for text in texts], device, batch_size))
    
    # 問題だけのときから対数確率が一番大きく増える選択肢を選ぶ
    correct = 0
    for row in rows:
        question = next(scores)
        choices  = [next(scores) - question for _ in row["choices"]["text"]]
        correct += row["choices"]["label"][max(range(len(choices)), key=choices.__getitem__)] == row["answerKey"]
    return {"arc_easy_tinyml": 100 * correct / len(rows), "arc_easy_tinyml_questions": len(rows)}


def benchmark_wikitext2(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* WikiText-2 (raw、test) の perplexity を、バイトごと (byte_ppl、リーダーボードの数字) とトークンごとに返す。
    #! 行を空白でつないで 1 つのテキストにする。そのトークンを重ならない max_length + 1 トークンの窓に切り、各窓に単独で点を付ける。
    #! byte_ppl = exp(点を付けた全トークンの負の対数尤度 / テキストの UTF-8 のバイト数)。だからトークナイザーの違うモデルどうしでも比べられる。
    '''
    
    # 行をつないで 1 つのテキストにし、トークンを max_length トークンずつずれる窓に切る
    files   = _benchmark_download("Salesforce/wikitext", ["wikitext-2-raw-v1/test-00000-of-00001.parquet"], data_dir / "wikitext")
    text    = " ".join(pq.read_table(files[0], columns=["text"]).column(0).to_pylist()).strip()
    ids     = tokenizer.encode(text).ids
    windows = [ids[start:start + max_length + 1] for start in range(0, len(ids) - 1, max_length)]
    
    # 全部の窓の負の対数尤度を足す
    nll    = -sum(score_texts(model, windows, device, batch_size))
    tokens = sum(len(window) - 1 for window in windows)
    return {"wikitext2_byte_ppl": math.exp(nll / len(text.encode("utf-8"))), "wikitext2_token_ppl": math.exp(nll / tokens), "wikitext2_tokens": tokens}


def benchmark_hellaswag(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Open SLM Leaderboard が使う lm-evaluation-harness の決まりで、HellaSwag の 10,042 の validation 問題の正解率 (%) を返す。
    #! 文脈は activity_label + ": " + ctx_a + " " + ctx_b (str.capitalize() を使うので、ほかの文字は小文字になる)、4 つの選択肢は endings。すべて _benchmark_hellaswag_text() で整える。
    #! "hellaswag" は acc_norm (リーダーボードの数字)。点の付け方は _benchmark_choices() を参照。
    '''
    
    files     = _benchmark_download("Rowan/hellaswag", ["data/validation-00000-of-00001.parquet"], data_dir / "hellaswag")
    rows      = pq.read_table(files[0], columns=["activity_label", "ctx_a", "ctx_b", "endings", "label"]).to_pylist()
    questions = [(_benchmark_hellaswag_text(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()), [_benchmark_hellaswag_text(ending) for ending in row["endings"]], int(row["label"])) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "hellaswag", questions)


def benchmark_arc_easy(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Open SLM Leaderboard が使う lm-evaluation-harness の決まりで、ARC-Easy の 2,376 のテスト問題の正解率 (%) を返す。
    #! 文脈は "Question: " + question + "\\nAnswer:"。データは benchmark_arc_easy_tinyml() (Tiny-ML Leaderboard) と同じだが、決まりが違うので 2 つのスコアは一致しない。
    #! "arc_easy" は acc_norm (リーダーボードの数字)。点の付け方は _benchmark_choices() を参照。
    '''
    
    files     = _benchmark_download("allenai/ai2_arc", ["ARC-Easy/test-00000-of-00001.parquet"], data_dir / "arc")
    rows      = pq.read_table(files[0]).to_pylist()
    questions = [(f"Question: {row['question']}\nAnswer:", row["choices"]["text"], row["choices"]["label"].index(row["answerKey"])) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "arc_easy", questions)


def benchmark_arc_challenge(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Open SLM Leaderboard が使う lm-evaluation-harness の決まりで、ARC-Challenge の 1,172 のテスト問題の正解率 (%) を返す。
    #! 文脈は "Question: " + question + "\\nAnswer:"。"arc_challenge" は acc_norm (リーダーボードの数字)。点の付け方は _benchmark_choices() を参照。
    '''
    
    files     = _benchmark_download("allenai/ai2_arc", ["ARC-Challenge/test-00000-of-00001.parquet"], data_dir / "arc")
    rows      = pq.read_table(files[0]).to_pylist()
    questions = [(f"Question: {row['question']}\nAnswer:", row["choices"]["text"], row["choices"]["label"].index(row["answerKey"])) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "arc_challenge", questions)


def benchmark_piqa(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Open SLM Leaderboard が使う lm-evaluation-harness の決まりで、PIQA の 1,838 の validation 問題の正解率 (%) を返す。
    #! 文脈は "Question: " + goal + "\\nAnswer:"、2 つの選択肢は sol1 と sol2。"piqa" は acc_norm (リーダーボードの数字)。点の付け方は _benchmark_choices() を参照。
    '''
    
    files     = _benchmark_download("baber/piqa", ["piqa_validation.parquet"], data_dir / "piqa")
    rows      = pq.read_table(files[0]).to_pylist()
    questions = [(f"Question: {row['goal']}\nAnswer:", [row["sol1"], row["sol2"]], row["label"]) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "piqa", questions)


def benchmark_arithmark3(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Open SLM Leaderboard が使う公式のスクリプト bencharithmark-3.py の決まりで、ArithMark-3 の 1,000 の算数の文章題の正解率 (%) を返す。
    #! lm-evaluation-harness と違い、文脈と各結末は別々に (特殊トークン無しで) エンコードしてからつなぐ。両方が max_length トークンに収まるよう、文脈の先頭のトークンを切る。
    #! どれかの結末が max_length トークン以上なら、その問題は丸ごと飛ばし、正解率の分母からも外す (公式のスクリプトはそこで止まる)。
    #! "arithmark3" は acc_norm (リーダーボードの数字。対数尤度 / 結末のトークン数 が一番大きいもの)、"arithmark3_acc" は acc (対数尤度が一番大きいもの)。同点なら最初の結末を選ぶ。
    #! 公式のスクリプトの既定は bfloat16。ここではほかのテストと同じく、その --dtype float32 の設定にあたる float32 で動かす。
    '''
    
    # 問題を読む (空でない各行に JSON のオブジェクトが 1 つ)
    files = _benchmark_download("AxiomicLabs/Arithmark-3.0", ["arithmark-3.jsonl"], data_dir / "arithmark3", ARITHMARK3_REVISION)
    with files[0].open(encoding="utf-8-sig") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    
    # 各結末のトークンを、その文脈の後ろに作る
    pairs = list[list[tuple[list[int], list[int], str]]]()
    for item in items:
        context_ids  = tokenizer.encode(item["ctx"], add_special_tokens=False).ids
        pairs.append([(context_ids, tokenizer.encode(ending, add_special_tokens=False).ids, ending) for ending in item["endings"]])
    
    # 対数尤度が一番大きい結末を、結末のトークン数で割る場合と割らない場合で選ぶ
    scores, indices = _benchmark_score(model, device, batch_size, "arithmark3", pairs, max_length)
    acc, acc_norm   = 0, 0
    for index, totals in zip(indices, scores):
        item      = items[index]
        means     = [total / len(ending_ids) for total, (_, ending_ids, _) in zip(totals, pairs[index])]
        acc      += max(range(len(totals)), key=totals.__getitem__) == int(item["label"])
        acc_norm += max(range(len(means)), key=means.__getitem__) == int(item["label"])
    return {"arithmark3": 100 * acc_norm / len(indices), "arithmark3_acc": 100 * acc / len(indices), "arithmark3_questions": len(indices), "arithmark3_skipped": len(items) - len(indices)}


BENCHMARK_REGISTRY: dict[str, tuple[str, BenchmarkFunction]] = {
    "blimp"          : ("Tiny-ML" , benchmark_blimp          ),
    "arc_easy_tinyml": ("Tiny-ML" , benchmark_arc_easy_tinyml),
    "wikitext2"      : ("Tiny-ML" , benchmark_wikitext2      ),
    "hellaswag"      : ("Open SLM", benchmark_hellaswag      ),
    "arc_easy"       : ("Open SLM", benchmark_arc_easy       ),
    "arc_challenge"  : ("Open SLM", benchmark_arc_challenge  ),
    "piqa"           : ("Open SLM", benchmark_piqa           ),
    "arithmark3"     : ("Open SLM", benchmark_arithmark3     ),
}


def benchmark_int_index(scores: dict[str, float]) -> dict[str, float]:
    '''
    #* HellaSwag、ARC、PIQA、ArithMark-3 のスコア (%) から、Open SLM Leaderboard の Int Index ("int_index") と Avg ("open_slm_avg") を返す。
    #? benchmark_tasks() から呼ぶ。計算はリーダーボードのページの getIntelligenceIndex() と getAvg() と同じ。
    #! Int Index: 各スコア s を 100 * (s - でたらめのスコア) / (100 - でたらめのスコア) にし、でたらめで 0、全問正解で 100 になるようにして (でたらめより低いと 0 未満)、INT_INDEX_PARTS の重みで平均する。
    #! ARC の項目は、変換前の arc_easy と arc_challenge の平均 (片方だけ実行したときはその値)。Avg は変換しない項目の単純な平均。
    #! 実行していない項目は、どちらの平均からも外す。項目が 2 つ未満なら空の辞書を返す。値は丸めない (リーダーボードは小数 2 桁で表示する)。
    '''
    
    # 実行した項目を取り出す。ARC の項目は 2 つの ARC のスコアの平均か、実行した方
    values = {name: scores[name] for name in ("hellaswag", "piqa", "arithmark3") if name in scores}
    arcs   = [scores[name] for name in ("arc_easy", "arc_challenge") if name in scores]
    if arcs:
        values["arc"] = arcs[0] if len(arcs) == 1 else (arcs[0] + arcs[1]) / 2
    parts = [(values[name], chance, weight) for name, chance, weight in INT_INDEX_PARTS if name in values]
    if len(parts) < 2:
        return {}
    
    # リーダーボードのページと同じく、左から 1 つずつ足す (Python 3.12 以降の sum() はより精密に足すので、最後の桁が変わることがある)
    index_total, index_weight, avg_total = 0.0, 0.0, 0.0
    for score, chance, weight in parts:
        index_total  += (score - chance) / (100 - chance) * 100 * weight
        index_weight += weight
        avg_total    += score
    return {"int_index": index_total / index_weight, "open_slm_avg": avg_total / len(parts)}


def benchmark_check(config: SimpleNamespace) -> None:
    '''
    #* validator.benchmark.tasks に BENCHMARK_REGISTRY に無い名前があれば、エラーで止める。
    #? benchmark_model() と、test データの無い学習の前の pipeline_train() (utils/trainer/pipeline.py) から呼ぶ。名前が間違っていれば、学習の前に止まる。
    '''
    
    unknown = [name for name in config.validator.benchmark.tasks if name not in BENCHMARK_REGISTRY]
    if unknown:
        raise Error("Unknown benchmark.", detail=f"validator.benchmark.tasks: {unknown}\nAvailable: {', '.join(BENCHMARK_REGISTRY)}", hint="Add the benchmark to BENCHMARK_REGISTRY in utils/validator/benchmark.py.", code=35)
    return None


def benchmark_tasks(model: nn.Module, tokenizer: Tokenizer, device: torch.device, config: SimpleNamespace, folder: Path, weights: str | None) -> dict[str, float]:
    '''
    #* validator.benchmark.tasks のテストをモデルで実行し、Open SLM Leaderboard の Int Index と Avg を足して、スコアを表で表示し、重みの名前と一緒に folder/benchmark.json に保存する。
    #? benchmark_model() と、test データの無い学習の後の pipeline_train() (utils/trainer/pipeline.py) から呼ぶ。先に benchmark_check() を呼んでおく。
    #! Tiny-ML Leaderboard のテストは Glint-1.3 の benchmark.py に、Open SLM Leaderboard のテストは lm-evaluation-harness と bencharithmark-3.py に従う。すべて BOS トークン無しの float32 で動き、モデルは data.max_length トークンまで読む。
    #! データは初回に validator.benchmark.data_dir にダウンロードする。
    '''
    
    # 各テストを実行し、リーダーボードごとのスコアの名前を覚える
    settings = config.validator.benchmark
    console  = Console()
    scores   = dict[str, float]()
    groups   = dict[str, list[str]]()
    for name in settings.tasks:
        console.print(f"[Benchmark] Running {name}...")
        leaderboard, function = BENCHMARK_REGISTRY[name]
        result                = function(model, tokenizer, device, Path(settings.data_dir), config.data.max_length, settings.batch_size)
        scores               |= result
        groups.setdefault(leaderboard, []).extend(result)
    
    # 項目が 2 つ以上実行されていれば、Int Index と Avg を足す
    index = benchmark_int_index(scores)
    if index:
        scores |= index
        groups.setdefault("Open SLM", []).extend(index)
    
    # 各リーダーボードのスコアを表のそれぞれの部分に表示する (Int Index と Avg は太字)
    table = Table(title="Benchmark", min_width=40)
    table.add_column("Leaderboard", style="bold magenta")
    table.add_column("Score"      , style="bold cyan"   )
    table.add_column("Value"      , justify="right"     )
    for leaderboard, names in groups.items():
        for row, name in enumerate(names):
            value = scores[name]
            table.add_row(leaderboard if row == 0 else "", name, f"{value:,}" if isinstance(value, int) else f"{value:.4f}", style="bold green" if name in index else None)
        table.add_section()
    console.print(table)
    
    # スコアを、それを出した重みの名前と一緒に保存する
    result = {"weights": weights, "tokenizer_dir": config.data.tokenizer_dir, "max_length": config.data.max_length, "date": datetime.now().isoformat(timespec="seconds"), "scores": scores}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "benchmark.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
    console.print(f"[Benchmark] Saved {folder / 'benchmark.json'}.")
    return scores


def benchmark_model(config: SimpleNamespace) -> None:
    '''
    #* model.checkpoint の重みでモデルとそのトークナイザーを読み込み、benchmark_tasks() で validator.benchmark.tasks のテストを実行する。
    #? main.py で run.mode (または --mode) が benchmark のとき選ばれる。load.py の load_model() を使う。
    #! benchmark.json は重みの隣に置く。model.checkpoint が null なら run.log_dir に置く。
    '''
    
    benchmark_check(config)
    model, tokenizer, device = load_model(config)
    folder                   = Path(config.model.checkpoint).parent if config.model.checkpoint is not None else Path(config.run.log_dir)
    benchmark_tasks(model, tokenizer, device, config, folder, config.model.checkpoint)
    return None
