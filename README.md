# SLM

## 環境構築

```powershell
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r pyproject.toml
```

続いて、[PyTorch 公式](https://pytorch.org/get-started/locally/)で OS・CUDA に合うコマンドを選び、`pip3 install...` を `uv pip install...` に置き換えて実行します。

## 実行方法

### 初回のみ：データ・トークナイザの準備

使うデータを選び、ダウンロード → 分割 → トークナイザ作成 → トークン化を一括実行。  
指定したデータを全件取得・使用。未指定のデータは除外。

```powershell
python -m tool.data all --tinystories --fineweb_edu --finemath -- cosmopedia --tokenizer_dir .\model\weight\hotaru
```

> [!WARNING]
> 指定先の既存トークナイザは上書きされます。
### 学習

```powershell
python main.py --config .\config\hybrid.json
```

## 使用可能なデータセット
>[!NOTE]
>`./tool/data/sources.py`を参照
- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)**
- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**
- **[Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/cosmopedia-v2)**
- **[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)**

## 実装されているモデル
>[!NOTE]
>`./model/hybrid/` と `./config/hybrid.json` を参照

- **hybrid**: ブロック列 `c`（ゲート付き短畳み込み）、`g`（Gated DeltaNet）、`a`（全体注意）、`w`（窓注意）を自由に並べるモデル。
  - 形はすべて `config.model.spec` に書きます（省略不可。知らないキー、型の違い、選べない値はエラー）。各項目の意味と選べる値は `config/hybrid.json` のコメントにあります。
  - 注意: GQA、qk-norm、出力ゲート（頭ごと / 値ごと）、RoPE（部分 RoPE 可）/ 学習する位置の表 / なし、softmax / SimA、MLA、ループ内の KV 共有
  - 埋め込み: 出力層との共有（tie）、2 段埋め込み
  - FFN: GELU / ReLU² / SwiGLU、正規化: LayerNorm / RMSNorm / rmsp2 / DyT、サンドイッチ正規化
  - ループ: 固定回数、周回ごとのループ埋め込みと LoRA、学習での周回数のランダム選択（`loop.times_train`）

## 測定可能なベンチマーク
>[!NOTE]
>`./utils/validator/benchmark.py`を参照

`--mode benchmark` またはテストデータがない場合の学習後に実行。データは初回に自動ダウンロード。

### [Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard)

- **[BLiMP](https://huggingface.co/datasets/nyu-mll/blimp)** (`blimp`)
- **[ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc)** (`arc_easy_tinyml`)
- **[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)** (`wikitext2`)

### [Open SLM Leaderboard](https://axiomiclabs-open-slm-leaderboard.static.hf.space/index.html)

- **[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)** (`hellaswag`)
- **[ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc)** (`arc_easy`)
- **[ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)** (`arc_challenge`)
- **[PIQA](https://huggingface.co/datasets/baber/piqa)** (`piqa`)
- **[ArithMark-3](https://huggingface.co/datasets/AxiomicLabs/Arithmark-3.0)** (`arithmark3`)

> [!WARNING]
> `arc_easy_tinyml` と `arc_easy` は測定方法が異なります。Int Index の ARC-Easy には `arc_easy` が使用されます。

- すべて `float32` で測定します。
- Open SLM は、選択肢が1つでも `data.max_length`（ArithMark-3 は `data.max_length - 1`）トークンを超える問題をスキップし、精度の分母から除外します。
### Int Index と Avg

$$
\mathrm{Int\ Index} =
\frac{
\widehat{\mathrm{HellaSwag}} + \widehat{\mathrm{ARC}} + \widehat{\mathrm{PIQA}} + 0.65 \times \widehat{\mathrm{ArithMark\text{-}3}}
}{
1 + 1 + 1 + 0.65
}
$$

$$
\widehat{s} = 100 \times \frac{s - c}{100 - c}
$$

- $`s`$: スコア（%）、$`c`$: チャンスレート（PIQA は 50、ほかは 25）。
- ARC = `arc_easy` と `arc_challenge` の平均（片方だけなら、その値）。
- 未実行の項目は分子・分母から除外。4項目中2項目以上で `int_index` と `open_slm_avg` を算出。
- Avg = 変換前のスコアの単純平均。

## Q＆A
  Use AI.
