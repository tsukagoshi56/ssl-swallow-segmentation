# 🍽️ 食行動音声データによる嚥下音検出実験

本プロジェクトは、**皮膚接触型マイクで収録した咀嚼・嚥下音**を用いた音響イベント検出モデルの構築と評価を行います。

---

## 🔁 全体の構成

```
share/
├── data_prepare/                # データ準備に関するノートブック
│   ├── 0_Eat_behavior_dataset.ipynb  # データセット展開・前処理・分割
│   ├── 1_eating_json.ipynb           # 音声ファイルとラベルを対応付けてJSON作成
│   └── 2_split_json.ipynb            # 訓練 / 検証 / テスト用にデータを分割
│
├── analysys/
│   └── show_all_results.ipynb   # モデルの出力結果の可視化・集計
│
├── experiment.py                # 実験の実行スクリプト（学習・推論・評価）
└── README.md                    # このファイル
```

---

## 📦 1. データ準備：`data_prepare/`

### 🔹 ステップ概要

| ステップ | 内容 |
|---------|------|
| `0_Eat_behavior_dataset.ipynb` | データダウンロード・分割・セグメント化・HPF適用までの全処理 |
| `1_eating_json.ipynb`          | `.wav` と `.txt` を対応付けた JSON ファイルを作成 |
| `2_split_json.ipynb`           | 訓練・検証・テストセットにデータを分割（比率で指定可） |

### 🔹 実行前に必要なライブラリ

```bash
pip install gdown librosa soundfile tqdm scipy
```

### 🔹 実行の流れ（代表: 0_Eat_behavior_dataset）

1. Google Drive から `.tar` ファイルをダウンロード
2. 音声ファイルのリサンプリング（16kHz）
3. ラベルテキストの整形と統合
4. 10秒以下のセグメントを自動生成
5. 合成音声保存 + 対応テキストラベル出力
6. 長時間ファイル削除（>17秒）
7. 全 `.wav` に High-Pass Filter（HPF）を適用
8. 元のフルファイルを削除し、必要最小限に整理

> 🎯 出力される音声 + ラベルは `dataset/old_wav_aug/conbined/`, `old_text_aug/` に格納されます。

---

## 🧠 2. モデル学習・評価：`experiment.py`

### 🔹 基本的な使い方

```bash
python experiment.py --model wavlm --task train --json_dir ./dataset/json --output_dir ./result
```

### 🔹 モデル選択肢

- `--model`：`wavlm`, `wav2vec2`, `hubert`, `mel`, `mfcc` など
- `--frontend` で手法変更も可能（詳細はスクリプト中に記載）

### 🔹 推論・評価

```bash
python experiment.py --task inference
```

---

## 📊 3. 結果の可視化：`analysys/`

- `show_all_results.ipynb` で実験ごとの出力を比較可能（IoU, F1スコアなどをプロット）

---

## 🔚 最終出力構成例

```
dataset/
├── old_wav_aug/
│   └── conbined/
│       ├── eat_*.wav
├── old_text_aug/
│   └── eat_*.txt
├── json/
│   ├── train.json
│   ├── valid.json
│   └── test.json

result/
├── model/
│   ├── checkpoint.pt
├── evaluation/
│   ├── eval_metrics.json
```
