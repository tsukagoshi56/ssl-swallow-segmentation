### 食行動音声データによる嚥下音検出実験

本プロジェクトでは、**皮膚接触型マイクで収録した咀嚼・嚥下音**に対し、自己教師あり音響特徴（SSL features）を用いてイベント検出（chewing, swallowing, noise）を行うモデルを構築・評価します。

#### 1. 全体構成

```plaintext
share/
├── data_prepare/
│ ├── 0_Eat_behavior_dataset.ipynb
│ ├── 1_eating_json.ipynb
│ └── 2_split_json.ipynb
│
├── analysys/
│ └── show_all_results.ipynb
│
├── experiment.py
└── README.md
```

#### 2. データ準備 (`data_prepare/`)

##### ステップ概要

| ステップ | 内容 |
|---------|------|
| `0_Eat_behavior_dataset.ipynb` | 音声/ラベルのダウンロード・整形・HPF適用 |
| `1_eating_json.ipynb` | .wav と .txt を対応付けた JSON 作成 |
| `2_split_json.ipynb` | JSON を学習・検証・テスト用に分割 |

##### 実行前に必要なライブラリ

```bash
pip install gdown librosa soundfile tqdm scipy
```

**処理の流れ（例：`0_Eat_behavior_dataset`）**

1. Google Drive から .tar ファイルをダウンロード
2. 16kHz へのリサンプリング
3. テキストラベルの整形・統合
4. 10秒単位でセグメント化・音声生成
5. 長時間 (>17秒) のファイルを削除
6. High-Pass Filter（HPF）の適用
7. 最小限ファイルのみを保存し構造を整理

**出力先：**

- 音声: `dataset/old_wav_aug/conbined/`
- ラベル: `old_text_aug/`

#### 3. 学習・推論・評価 (`experiment.py`)

##### 基本コマンド

```bash
# 学習 + 評価（検証・テスト）
python experiment.py

# テストのみ実行
python experiment.py --test

# フレーム単位の確率出力（inference）
python experiment.py --inference
```

##### モデル設定の指定方法

本スクリプトでは、実験条件はすべて `experiment.py` 冒頭部の以下のような定義で切り替えます。

```python
# 実験定義
EXPERIMENT_GROUP = "wavlm_gru"

EXPERIMENTS = {
    "wavlm_gru": [
        {"name": "wavlm_gru", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"}
    ],
    "exp": [
        {"name": "gru_mel", "feature_type": "mel", "architecture": "gru", "ssl_model_name": None},
        # ...
    ]
}
```

`EXPERIMENT_GROUP` に指定したキーに対応するリスト内の設定が自動的に読み込まれます。

**各設定のパラメータ:**

- `name`: 実験名
- `ssl_model_name`: 使用するSSLモデル（例: `"microsoft/wavlm-base-plus"`）
- `feature_type`: `"raw"`, `"mel"`, `"mfcc"` など
- `architecture`: `"gru"`, `"lstm"`, `"fc"` など
- （任意）`dataset_frac`: 使用するデータ割合（`1.0` = 全体）

**実験切り替え方法**: `EXPERIMENT_GROUP` を変更します。

```python
EXPERIMENT_GROUP = "exp"
```

#### 4. 結果の可視化 (`analysys/`)

`show_all_results.ipynb` にて、各実験の出力結果（IoU、F1スコアなど）をプロットし比較可能です。

---

## English Version: Swallowing Sound Detection Experiment Using Eating Behavior Audio Data

This project builds and evaluates models for event detection (chewing, swallowing, noise) using **chewing/swallowing sounds recorded with skin-attached microphones** and self-supervised acoustic features (SSL features).

### 1. Project Structure

```plaintext
share/
├── data_prepare/
│ ├── 0_Eat_behavior_dataset.ipynb
│ ├── 1_eating_json.ipynb
│ └── 2_split_json.ipynb
│
├── analysys/
│ └── show_all_results.ipynb
│
├── experiment.py
└── README.md
```

### 2. Data Preparation (`data_prepare/`)

#### Step Overview

| Step | Description |
|---------|------|
|`0_Eat_behavior_dataset.ipynb` | Downloads/processes audio/labels; applies HPF |
| `1_eating_json.ipynb` | Creates JSON matching .wav and .txt |
| `2_split_json.ipynb` | Splits JSON into train/val/test |

#### Required Libraries

```bash
pip install gdown librosa soundfile tqdm scipy
```

**Processing Flow (Example: `0_Eat_behavior_dataset`)**

1. Download .tar file from Google Drive.
2. Resample to 16 kHz.
3. Process and merge text labels.
4. Segment into 10-s chunks, generate audio.
5. Remove long files (>17 seconds).
6. Apply High-Pass Filter (HPF).
7. Save necessary files and organize structure.

**Outputs:**

- Audio: `dataset/old_wav_aug/conbined/`
- Labels: `old_text_aug/`

### 3. Train/Inference/Evaluation (`experiment.py`)

#### Basic Commands

```bash
# Train + evaluate (validation/test)
python experiment.py

# Run only test
python experiment.py --test

# Output frame-level probabilities (inference)
python experiment.py --inference
```

#### Model Configuration

Experiment conditions are set in the `experiment.py` script as follows:

```python
# --- Experiment Definitions ---
EXPERIMENT_GROUP = "wavlm_gru"

EXPERIMENTS = {
    "wavlm_gru": [
        {"name": "wavlm_gru", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"}
    ],
    "exp": [
        {"name": "gru_mel", "feature_type": "mel", "architecture": "gru", "ssl_model_name": None},
        # ...
    ]
}
```

The list of configurations is automatically loaded based on the key specified in `EXPERIMENT_GROUP`.

**Configuration Parameters:**

- `name`: Experiment name
- `ssl_model_name`: The SSL model to use (e.g., `"microsoft/wavlm-base-plus"`)
- `feature_type`: `"raw"`, `"mel"`, `"mfcc"`, etc.
- `architecture`: `"gru"`, `"lstm"`, `"fc"`, etc.
- (Optional) `dataset_frac`: Fraction of the data to use (`1.0` = entire dataset)

**How to Change Experiments:** Modify `EXPERIMENT_GROUP`.

```python
EXPERIMENT_GROUP = "exp"
```

### 4. Visualization of Results (`analysys/`)

`show_all_results.ipynb` can plot metrics (IoU, F1-score, etc.) for all experiments to facilitate comparison.