# -*- coding: utf-8 -*-
"""
Sound‑Event Detection Experiment Script (SSL + RNN/FC/CRNN)
-----------------------------------------------------
* Unified version combining training and event‑based evaluation (IoU‑F1).
* Supports WavLM, Wav2Vec2, HuBERT front‑ends or handcrafted features.
* Includes checkpointing, early stopping, and experiment management.
* Evaluates on validation set each epoch and test set at the end.
* Includes a smoke test option to quickly check basic functionality for all experiments,
  including event metric calculation on the test set for the limited run.

Usage:
  $ python sound_event_ssl_experiment.py
  $ python sound_event_ssl_experiment.py --test
  $ python sound_event_ssl_experiment.py --test --model model_name
"""

import os
import re
from pathlib import Path
import json
import math
import random
import datetime
from pathlib import Path
import sys
import logging
import time
import itertools
import argparse
from typing import List, Dict, Tuple, Optional, Union, Any
from functools import partial

import torch
import torchaudio
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2Model, HubertModel, PreTrainedModel

random.seed(42)

LABEL_SHORT = {"chewing": "ch", "swallowing": "sw", "noise": "ns"}

# --- Configuration ---
BASE_CONFIG: Dict[str, Any] = {
    # Data paths
    "train_json": "./json/ssl_gru_only_eat_not_aug/train.json",
    "val_json":   "./json/ssl_gru_only_eat_not_aug/val.json",
    "test_json":  "./json/ssl_gru_only_eat_not_aug/test.json",
    "base_save_dir": f"./result",
    # Processing
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "sr": 16000,
    "ssl_hop_length": 320,    # Corresponds to 20 ms @ 16 kHz
    # Training
    "batch_size": 8,
    "num_epochs": 150,
    "lr": 1e-5,
    "optimizer": "adam",
    "momentum": 0.9,
    "early_stopping_patience": 10,
    # Model / Task
    "classes": ["chewing", "swallowing", "noise"],
    # Default SSL settings
    "default_ssl_model_name": "microsoft/wavlm-base-plus",
    "default_freeze_fe": True,
    "default_freeze_transformer": False,
    # Inference
    "threshold": 0.5,
}

# --- Constants ---
EPS = sys.float_info.epsilon
IOU_THRESHOLDS: List[float] = [EPS] + [i * 0.1 for i in range(1, 11)]

# --- Experiment Definitions ---
EXPERIMENT_GROUP = "wavlm_gru"
EXPERIMENTS: Dict[str, List[Dict[str, Any]]] = {
    "wavlm_gru": [
           {"name": "wavlm_gru",  "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
  ],

    "exp": [ 
        {"name": "wavlm_gru_75pct", "dataset_frac": 0.75, "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
        {"name": "wavlm_gru_50pct",  "dataset_frac": 0.5, "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
        {"name": "wavlm_gru_25pct",  "dataset_frac": 0.25,"ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
        {"name": "wavlm_gru_10pct",  "dataset_frac": 0.1, "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
        {"name": "gru_raw",  "feature_type": "raw",  "architecture": "gru", "ssl_model_name": "microsoft/wavlm-base-plus"}, # Needs an SSL model if raw
        {"name": "gru_mfcc", "feature_type": "mfcc", "architecture": "gru", "ssl_model_name": None},
        {"name": "gru_mel",  "feature_type": "mel",  "architecture": "gru", "ssl_model_name": None},
        {"name": "wavlm_gru",  "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "gru"},
        {"name": "wavlm_lstm", "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "lstm"},
        {"name": "wavlm_fc",   "ssl_model_name": "microsoft/wavlm-base-plus", "feature_type": "raw", "architecture": "fc"},
        {"name": "wavlm-base-plus", "ssl_model_name": "microsoft/wavlm-base-plus", "freeze_fe": True, "freeze_transformer": False, "feature_type": "raw", "architecture": "gru"},
        {"name": "wav2vec2-base",   "ssl_model_name": "facebook/wav2vec2-base",      "freeze_fe": True, "freeze_transformer": False, "feature_type": "raw", "architecture": "gru"},
        {"name": "wav2vec2-large",  "ssl_model_name": "facebook/wav2vec2-large",     "freeze_fe": True, "freeze_transformer": False, "feature_type": "raw", "architecture": "gru"},
        {"name": "hubert-base",     "ssl_model_name": "facebook/hubert-base-ls960",  "freeze_fe": True, "freeze_transformer": False, "feature_type": "raw", "architecture": "gru"},
        {"name": "hubert-large",    "ssl_model_name": "facebook/hubert-large-ls960-ft","freeze_fe": True,"freeze_transformer": False, "feature_type": "raw", "architecture": "gru"},
        {"name": "wavlm-large",    "ssl_model_name": "microsoft/wavlm-large","freeze_fe": True,"freeze_transformer": False, "feature_type": "raw", "architecture": "gru"},
    ]
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class SoundEventDataset(Dataset):
    def __init__(self,
                 annotations: List[Dict[str, Any]],
                 config: Dict[str, Any],
                 feature_type: str = "raw",
                 dataset_frac: float = 1.0):
        super().__init__()
        self.sr = config["sr"]
        self.hop_length = config["ssl_hop_length"]
        self.hop_time = self.hop_length / self.sr
        self.classes = config["classes"]
        self.num_classes = len(self.classes)
        self.feature_type = feature_type

        if not (0.0 < dataset_frac <= 1.0):
            raise ValueError("dataset_frac must be between 0 (exclusive) and 1 (inclusive)")

        # Select a fraction of the dataset if needed
        ann = annotations.copy()
        if dataset_frac < 1.0:
            num_samples = int(len(ann) * dataset_frac)
            random.shuffle(ann)
            self.annotations = ann[:num_samples]
            logger.info(f"Using {num_samples}/{len(annotations)} samples ({dataset_frac:.1%})")
        else:
            self.annotations = ann

        # Initialize feature transforms if not using raw audio
        self.feature_transform = None
        if feature_type == "mfcc":
            self.feature_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sr, n_mfcc=40, log_mels=True,
                melkwargs={'n_fft': 400, 'hop_length': self.hop_length, 'n_mels': 64, 'center': False}
            )
            self.feature_dim = 40
        elif feature_type == "mel":
            self.feature_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_mels=128, n_fft=1024, hop_length=self.hop_length, center=False
            )
            self.feature_dim = 128
        elif feature_type == "raw":
            self.feature_dim = 1
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

    def __len__(self) -> int:
        return len(self.annotations)

    def _make_label_matrix(self,
                           timestamps: Dict[str, List[Tuple[float, float]]],
                           num_frames: int) -> torch.Tensor:
        label_matrix = torch.zeros((self.num_classes, num_frames), dtype=torch.float32)
        for class_idx, class_name in enumerate(self.classes):
            for start_time, end_time in timestamps.get(class_name, []):
                start_frame = max(0, math.floor(start_time / self.hop_time))
                end_frame = min(num_frames, math.ceil(end_time / self.hop_time))
                end_frame = max(start_frame, end_frame)
                if start_frame < num_frames:
                   label_matrix[class_idx, start_frame:end_frame] = 1.0
        return label_matrix

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.annotations[idx]
        audio_path = item["path"]

        try:
            wav, sr0 = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise e

        # Resample if necessary
        if sr0 != self.sr:
            resampler = torchaudio.transforms.Resample(sr0, self.sr)
            wav = resampler(wav)

        # Convert to mono
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0)
        wav = wav.squeeze()

        num_samples = wav.shape[-1]
        num_frames = num_samples // self.hop_length

        # Create label matrix
        labels = self._make_label_matrix(item["timestamps"], num_frames)

        # Prepare features or return raw audio
        if self.feature_transform:
            if wav.dim() == 0:
                 wav = wav.unsqueeze(0)

            features = self.feature_transform(wav.unsqueeze(0))
            features = features.squeeze(0).permute(1, 0)

            feat_len = features.shape[0]
            label_len = labels.shape[1]
            if feat_len < label_len:
                 labels = labels[:, :feat_len]
            elif label_len < feat_len:
                 features = features[:label_len, :]
                 num_frames = label_len

            min_len = min(feat_len, labels.shape[1])
            features = features[:min_len, :]
            labels = labels[:, :min_len]

            return features, labels, audio_path
        else:
            return wav, labels, audio_path

# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------
class EventDetector(nn.Module):
    def __init__(self,
                 config: Dict[str, Any],
                 ssl_model_name: Optional[str],
                 freeze_feature_extractor: bool,
                 freeze_transformer_layers: bool,
                 architecture: str,
                 feature_type: str):
        super().__init__()
        self.feature_type = feature_type
        self.architecture = architecture
        self.ssl_model: Optional[PreTrainedModel] = None
        input_dimension: int

        # SSL モデル利用（feature_type="raw" & ssl_model_name ありの場合）
        if feature_type == "raw" and ssl_model_name:
            logger.info(f"Loading SSL model: {ssl_model_name}")
            if "wavlm" in ssl_model_name.lower():
                self.ssl_model = WavLMModel.from_pretrained(ssl_model_name)
            elif "wav2vec2" in ssl_model_name.lower():
                self.ssl_model = Wav2Vec2Model.from_pretrained(ssl_model_name)
            elif "hubert" in ssl_model_name.lower():
                self.ssl_model = HubertModel.from_pretrained(ssl_model_name)
            else:
                raise ValueError(f"Unsupported SSL model type in name: {ssl_model_name}")

            if freeze_feature_extractor:
                logger.info("Freezing SSL feature extractor.")
                if hasattr(self.ssl_model, 'feature_extractor'):
                    for param in self.ssl_model.feature_extractor.parameters():
                        param.requires_grad = False
                if hasattr(self.ssl_model, 'feature_projection'):
                    for param in self.ssl_model.feature_projection.parameters():
                        param.requires_grad = False

            if freeze_transformer_layers:
                if hasattr(self.ssl_model, 'encoder'):
                     logger.info("Freezing SSL transformer encoder layers.")
                     for param in self.ssl_model.encoder.parameters():
                         param.requires_grad = False
                else:
                     logger.warning(f"Model {ssl_model_name} does not seem to have an 'encoder' attribute for freezing.")

            input_dimension = self.ssl_model.config.hidden_size

        # 手加工特徴量やraw直接利用の場合
        elif feature_type == "mfcc":
            input_dimension = 40
            logger.info("Using MFCC features (dim=40).")
        elif feature_type == "mel":
            input_dimension = 128
            logger.info("Using Mel Spectrogram features (dim=128).")
        elif feature_type == "raw" and ssl_model_name is None:
            # Raw audio direct processing (1D CNN to process raw audio)
            logger.info("Using raw audio directly without SSL frontend.")
            input_dimension = 1  # Raw audio has 1 channel
        else:
             raise ValueError(f"Invalid feature_type: {feature_type}")

        # --- Backend Layers ---
        # For GRU/LSTM/FC architectures
        if architecture in ["gru", "lstm", "fc"]:
            # Projection layer to map features to a common dimension
            self.projection = nn.Linear(input_dimension, 256)
            self.activation = nn.ReLU()

            # Recurrent or Fully Connected layer
            self.rnn_or_fc: Optional[nn.Module] = None
            rnn_hidden_size = 128
            if architecture == "gru":
                logger.info("Using GRU backend.")
                self.rnn_or_fc = nn.GRU(
                    input_size=256,
                    hidden_size=rnn_hidden_size,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3
                )
                output_dimension = rnn_hidden_size * 2
            elif architecture == "lstm":
                logger.info("Using LSTM backend.")
                self.rnn_or_fc = nn.LSTM(
                    input_size=256,
                    hidden_size=rnn_hidden_size,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3
                )
                output_dimension = rnn_hidden_size * 2
            elif architecture == "fc":
                logger.info("Using FC backend (applied frame-wise).")
                self.rnn_or_fc = None
                output_dimension = 256
            
            self.dropout = nn.Dropout(0.3)
            self.output_layer = nn.Linear(output_dimension, len(config["classes"]))
        
        # For CRNN architecture with SSL input
        elif architecture == "crnn":
            logger.info("Using CRNN backend (with SSL features).")
            # CNN層: SSL出力をさらに時間方向に畳み込む
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dimension, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            
            # RNN層
            rnn_input_size = 128
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            
            # 分類層
            self.dropout = nn.Dropout(0.3)
            self.output_layer = nn.Linear(256, len(config["classes"]))

            # アップサンプリング層
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # For CRNN architecture with direct raw audio input (without SSL)
        elif architecture == "crnn_direct":
            logger.info(f"Using CRNN_DIRECT backend (directly from {feature_type}).")
            
            if feature_type == "raw":
                # Raw audio direct processing with 1D convolution
                self.raw_frontend = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=40),  # ~5ms stride
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4),  # Further downsample
                    nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4),  # Final downsample to ~20ms frames
                )
                cnn_input_dim = 128
            else:
                # MFCC/Mel input
                cnn_input_dim = input_dimension
            
            # CNN層
            self.cnn = nn.Sequential(
                nn.Conv1d(cnn_input_dim, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            
            # RNN層
            self.rnn = nn.GRU(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            
            # 分類層
            self.dropout = nn.Dropout(0.3)
            self.output_layer = nn.Linear(256, len(config["classes"]))

            # アップサンプリング層
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 前処理部分（SSL or 手加工特徴量 or raw処理）
        if self.ssl_model and self.feature_type == "raw":
            # SSL利用
            x = x.to(torch.float32)
            output = self.ssl_model(x)
            h = output.last_hidden_state  # (B, T_feat, hidden_size)
        elif self.feature_type == "raw" and self.architecture == "crnn_direct":
            # Raw音声を直接処理する場合（SSLなし）
            x = x.unsqueeze(1)  # (B, 1, T_wav) - Add channel dimension for Conv1D
            h = self.raw_frontend(x)  # (B, 128, T_feat)
        else:
            # mfcc/mel または raw から CRNN_direct への場合
            if self.architecture == "crnn_direct" and self.feature_type in ["mfcc", "mel"]:
                # 既に (B, T, F) の場合、(B, F, T) に変換
                h = x.transpose(1, 2)  # (B, F, T)
            else:
                # 通常の入力
                h = x  # (B, T, F)

        # 2. アーキテクチャに応じた処理
        if self.architecture in ["crnn", "crnn_direct"]:
            # 標準のCRNN処理（通常のSSL-CRNNとSSLなしCRNN_directで共通）
            if hasattr(self, "cnn"):
                if h.dim() == 3 and h.size(1) != h.size(2):
                    # 既に (B, F, T) 形式になっている場合はそのまま
                    pass
                else:
                    # まだ (B, T, F) の場合は変換
                    h = h.transpose(1, 2)  # (B, F, T)
                
                h = self.cnn(h)  # (B, 128, T/2)
                h = h.transpose(1, 2)  # (B, T/2, 128)
            
            # RNN適用
            self.rnn.flatten_parameters()
            h, _ = self.rnn(h)  # (B, T/2, 256)
            
            # ドロップアウト
            h = self.dropout(h)
            
            # 分類層
            logits = self.output_layer(h)  # (B, T/2, C)
            
            # 時間方向をアップサンプリング
            logits = logits.transpose(1, 2)  # (B, C, T/2)
            logits = self.upsample(logits)  # (B, C, T)
            
            return logits
        else:
            # GRU/LSTM/FC用の処理
            if hasattr(self, "projection"):
                h = self.activation(self.projection(h))  # (B, T, 256)

            if isinstance(self.rnn_or_fc, (nn.GRU, nn.LSTM)):
                self.rnn_or_fc.flatten_parameters()
                h, _ = self.rnn_or_fc(h)          # (B, T, hidden*2)

            h = self.dropout(h)                   # (B, T, output_dim)
            logits = self.output_layer(h)         # (B, T, C)

            # (B, C, T) に並べ替え
            return logits.permute(0, 2, 1)

# -----------------------------------------------------------------------------
# Collate Function for DataLoader
# -----------------------------------------------------------------------------
def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, str]],
                config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    features_list, labels_list, paths = zip(*batch)

    # Determine max lengths based on actual data dimensions
    is_raw_audio = features_list[0].dim() == 1
    if is_raw_audio:
        max_len_feat = max(x.size(0) for x in features_list)
    else:
        max_len_feat = max(x.size(0) for x in features_list)

    max_len_label = max(y.size(1) for y in labels_list)

    # Pad features/waveforms
    if is_raw_audio:
        padded_features = torch.stack(
            [nn.functional.pad(x, (0, max_len_feat - x.size(0))) for x in features_list]
        )
    else:
        feature_dim = features_list[0].size(1)
        padded_features_list = []
        for x in features_list:
            padding_size = max_len_feat - x.size(0)
            padded_x = nn.functional.pad(x, (0, 0, 0, padding_size))
            padded_features_list.append(padded_x)
        padded_features = torch.stack(padded_features_list)

    # Pad labels (C, T) -> (B, C, T_max)
    padded_labels = torch.stack(
        [nn.functional.pad(y, (0, max_len_label - y.size(1))) for y in labels_list]
    )

    return padded_features, padded_labels, list(paths)


# -----------------------------------------------------------------------------
# Event-based Metrics Utilities
# -----------------------------------------------------------------------------
def interval_iou(interval_a: Tuple[float, float], interval_b: Tuple[float, float]) -> float:
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)

    intersection_duration = max(0.0, intersection_end - intersection_start)
    union_duration = (end_a - start_a) + (end_b - start_b) - intersection_duration

    return intersection_duration / (union_duration + EPS) if union_duration >= 0 else 0.0

def sequence_to_intervals(binary_sequence: np.ndarray, hop_time: float) -> List[Tuple[float, float]]:
    intervals = []
    in_event = False
    start_frame = 0
    padded_sequence = np.pad(binary_sequence, (0, 1), mode='constant', constant_values=0)

    for frame_index, value in enumerate(padded_sequence):
        if value > 0 and not in_event:
            in_event = True
            start_frame = frame_index
        elif value == 0 and in_event:
            in_event = False
            end_frame = frame_index
            intervals.append((start_frame * hop_time, end_frame * hop_time))
            if not (end_frame * hop_time >= start_frame * hop_time):
                logger.warning(f"Detected interval with end time < start time: "
                               f"({start_frame * hop_time:.3f}, {end_frame * hop_time:.3f}). "
                               f"Start frame: {start_frame}, End frame: {end_frame}")

    return intervals

def model_output_to_intervals(
    output_tensor: torch.Tensor, threshold: float, hop_time: float, class_names: List[str]
    ) -> Dict[str, List[Tuple[float, float]]]:
    if output_tensor.dim() != 2:
        raise ValueError(f"output_tensor must have shape (C, T), but got {output_tensor.shape}")
    if len(class_names) != output_tensor.shape[0]:
        raise ValueError(f"Number of class_names ({len(class_names)}) must match C dimension ({output_tensor.shape[0]}) of output_tensor")

    output_tensor = output_tensor.detach().cpu()
    binary_predictions = (output_tensor.numpy() >= threshold).astype(np.int8)
    predicted_intervals = {}
    for i, class_name in enumerate(class_names):
        predicted_intervals[class_name] = sequence_to_intervals(binary_predictions[i], hop_time)
    return predicted_intervals

def extract_food_type(file_path: str) -> str:
    food_types = ["cbg","gum", "rtz", "w20"]
    
    file_name = os.path.basename(file_path).lower()
    
    for food in food_types:
        if food in file_name:
            return food
    
    return "unknown"

def calculate_class_specific_metrics(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str],
    iou_thresholds: List[float] = IOU_THRESHOLDS
    ) -> Dict[float, Tuple[float, float, float]]:
    results = {}
    
    if not ground_truth_list or not prediction_list:
        return {thr: (0.0, 0.0, 0.0) for thr in iou_thresholds}

    for iou_thr in iou_thresholds:
        total_tp = 0
        total_n_pred = 0
        total_n_gt = 0

        for gt_intervals_dict, pred_intervals_dict in zip(ground_truth_list, prediction_list):
            for class_name in class_names:
                gt_intervals = gt_intervals_dict.get(class_name, [])
                pred_intervals = pred_intervals_dict.get(class_name, [])

                n_gt = len(gt_intervals)
                n_pred = len(pred_intervals)
                total_n_gt += n_gt
                total_n_pred += n_pred

                if n_gt == 0 or n_pred == 0:
                    continue

                iou_matrix = np.zeros((n_gt, n_pred))
                for gt_idx in range(n_gt):
                    for pred_idx in range(n_pred):
                         iou_matrix[gt_idx, pred_idx] = interval_iou(gt_intervals[gt_idx], pred_intervals[pred_idx])

                matched_gt = np.zeros(n_gt, dtype=bool)
                matched_pred = np.zeros(n_pred, dtype=bool)

                tp_class_file = 0
                for gt_idx in range(n_gt):
                     best_iou = -1.0
                     best_pred_idx = -1
                     for pred_idx in range(n_pred):
                         if not matched_pred[pred_idx] and iou_matrix[gt_idx, pred_idx] >= iou_thr:
                             if iou_matrix[gt_idx, pred_idx] > best_iou:
                                 best_iou = iou_matrix[gt_idx, pred_idx]
                                 best_pred_idx = pred_idx

                     if best_pred_idx != -1:
                         matched_gt[gt_idx] = True
                         matched_pred[best_pred_idx] = True
                         tp_class_file += 1

                total_tp += tp_class_file

        precision = total_tp / (total_n_pred + EPS)
        recall = total_tp / (total_n_gt + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)

        precision = np.clip(precision, 0.0, 1.0)
        recall = np.clip(recall, 0.0, 1.0)
        f1_score = np.clip(f1_score, 0.0, 1.0)

        results[iou_thr] = (float(precision), float(recall), float(f1_score))

    return results

def load_patience_from_log(log_path: Path, patience_total: int) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    pattern = re.compile(r'Patience:\s*(\d+)\s*/\s*{}'.format(patience_total))
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else 0

def collect_matched_iou(
    gt_list: List[Dict[str, List[Tuple[float, float]]]],
    pred_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str],
    iou_threshold: float = EPS
) -> Dict[str, List[float]]:
    matched_ious = {cn: [] for cn in class_names}
    for gt_dict, pred_dict in zip(gt_list, pred_list):
        for cn in class_names:
            gt_intervals = gt_dict.get(cn, [])
            pred_intervals = pred_dict.get(cn, [])
            iou_matrix = [[interval_iou(a, b) for b in pred_intervals] for a in gt_intervals]
            matched_pred = set()
            for i, row in enumerate(iou_matrix):
                best_j, best_iou = -1, -1.0
                for j, val in enumerate(row):
                    if j in matched_pred:  continue
                    if val >= iou_threshold and val > best_iou:
                        best_iou = val
                        best_j = j
                if best_j >= 0:
                    matched_pred.add(best_j)
                    matched_ious[cn].append(best_iou)
    return matched_ious

def has_smoke_test_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    return "Smoke test epoch finished." in text

def calculate_event_metrics_by_class_and_food(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    file_paths: List[str],
    class_names: List[str],
    iou_thresholds: List[float] = IOU_THRESHOLDS
    ) -> Dict[str, Dict[str, Dict[float, Tuple[float, float, float]]]]:
    results = {"all": {}}
    food_files = {}
    
    for i, path in enumerate(file_paths):
        food_type = extract_food_type(path)
        if food_type not in food_files:
            food_files[food_type] = []
        food_files[food_type].append(i)
    
    for class_name in class_names:
        class_metrics = calculate_class_specific_metrics(
            ground_truth_list, prediction_list, [class_name], iou_thresholds
        )
        results["all"][class_name] = class_metrics
    
    for food_type, indices in food_files.items():
        if food_type not in results:
            results[food_type] = {}
        
        food_gt = [ground_truth_list[i] for i in indices]
        food_pred = [prediction_list[i] for i in indices]
        
        for class_name in class_names:
            food_class_metrics = calculate_class_specific_metrics(
                food_gt, food_pred, [class_name], iou_thresholds
            )
            results[food_type][class_name] = food_class_metrics
    
    return results

def print_event_statistics(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    file_paths: List[str],
    class_names: List[str]
    ):
    food_counts = {}
    total_counts = {class_name: {"gt": 0, "pred": 0} for class_name in class_names}
    food_type_counts = {}
    
    print("\n=== イベント統計情報 ===")
    print(f"テストサンプル数: {len(file_paths)}")
    
    for i, (gt, pred, path) in enumerate(zip(ground_truth_list, prediction_list, file_paths)):
        food_type = extract_food_type(path)
        
        if food_type not in food_counts:
            food_counts[food_type] = 0
            food_type_counts[food_type] = {class_name: {"gt": 0, "pred": 0} for class_name in class_names}
        
        food_counts[food_type] += 1
        
        for class_name in class_names:
            gt_events = len(gt.get(class_name, []))
            pred_events = len(pred.get(class_name, []))
            
            total_counts[class_name]["gt"] += gt_events
            total_counts[class_name]["pred"] += pred_events
            
            food_type_counts[food_type][class_name]["gt"] += gt_events
            food_type_counts[food_type][class_name]["pred"] += pred_events
    
    print("\n--- クラス別イベント数 (全体) ---")
    for class_name, counts in total_counts.items():
        print(f"{class_name}: GT={counts['gt']}, 予測={counts['pred']}")
    
    print("\n--- 食材別サンプル数 ---")
    for food_type, count in food_counts.items():
        print(f"{food_type}: {count}サンプル")
    
    print("\n--- 食材・クラス別イベント数 ---")
    for food_type, class_counts in food_type_counts.items():
        print(f"\n{food_type}:")
        for class_name, counts in class_counts.items():
            print(f"  {class_name}: GT={counts['gt']}, 予測={counts['pred']}")

def collect_raw_iou(
    gt_list: List[Dict[str, List[Tuple[float, float]]]],
    pred_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str]
) -> Dict[str, List[float]]:
    raw = {cn: [] for cn in class_names}
    for gt, pred in zip(gt_list, pred_list):
        for cn in class_names:
            for a in gt.get(cn, []):
                for b in pred.get(cn, []):
                    raw[cn].append(interval_iou(a, b))
    return raw

def collect_raw_iou_by_food(
    gt_list: List[Dict[str, List[Tuple[float, float]]]],
    pred_list: List[Dict[str, List[Tuple[float, float]]]],
    file_paths: List[str],
    class_names: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    food_iou = {}
    for gt, pred, path in zip(gt_list, pred_list, file_paths):
        ft = extract_food_type(path)
        if ft not in food_iou:
            food_iou[ft] = {cn: [] for cn in class_names}
        for cn in class_names:
            for a in gt.get(cn, []):
                for b in pred.get(cn, []):
                    food_iou[ft][cn].append(interval_iou(a, b))
    return food_iou

def calculate_event_metrics(
    ground_truth_list: List[Dict[str, List[Tuple[float, float]]]],
    prediction_list: List[Dict[str, List[Tuple[float, float]]]],
    class_names: List[str],
    iou_thresholds: List[float] = IOU_THRESHOLDS
    ) -> Dict[float, Tuple[float, float, float]]:
    results = {}
    if not ground_truth_list or not prediction_list:
        logger.warning("Ground truth or prediction list is empty, cannot calculate metrics.")
        return {thr: (0.0, 0.0, 0.0) for thr in iou_thresholds}

    for iou_thr in iou_thresholds:
        total_tp = 0
        total_n_pred = 0
        total_n_gt = 0

        for gt_intervals_dict, pred_intervals_dict in zip(ground_truth_list, prediction_list):
            for class_name in class_names:
                gt_intervals = gt_intervals_dict.get(class_name, [])
                pred_intervals = pred_intervals_dict.get(class_name, [])

                n_gt = len(gt_intervals)
                n_pred = len(pred_intervals)
                total_n_gt += n_gt
                total_n_pred += n_pred

                if n_gt == 0 or n_pred == 0:
                    continue

                iou_matrix = np.zeros((n_gt, n_pred))
                for gt_idx in range(n_gt):
                    for pred_idx in range(n_pred):
                         iou_matrix[gt_idx, pred_idx] = interval_iou(gt_intervals[gt_idx], pred_intervals[pred_idx])

                matched_gt = np.zeros(n_gt, dtype=bool)
                matched_pred = np.zeros(n_pred, dtype=bool)

                tp_class_file = 0
                for gt_idx in range(n_gt):
                     best_iou = -1.0
                     best_pred_idx = -1
                     for pred_idx in range(n_pred):
                         if not matched_pred[pred_idx] and iou_matrix[gt_idx, pred_idx] >= iou_thr:
                             if iou_matrix[gt_idx, pred_idx] > best_iou:
                                 best_iou = iou_matrix[gt_idx, pred_idx]
                                 best_pred_idx = pred_idx

                     if best_pred_idx != -1:
                         matched_gt[gt_idx] = True
                         matched_pred[best_pred_idx] = True
                         tp_class_file += 1

                total_tp += tp_class_file

        precision = total_tp / (total_n_pred + EPS)
        recall = total_tp / (total_n_gt + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)

        precision = np.clip(precision, 0.0, 1.0)
        recall = np.clip(recall, 0.0, 1.0)
        f1_score = np.clip(f1_score, 0.0, 1.0)

        results[iou_thr] = (float(precision), float(recall), float(f1_score))

    return results

# -----------------------------------------------------------------------------
# Training and Evaluation Helper Functions
# -----------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch_num: int,
                    num_epochs_total: int,
                    limit: Optional[int] = None) -> float:
    model.train()
    total_loss = 0.0
    batches_processed = 0
    num_batches = len(dataloader)
    batches_to_run = min(limit, num_batches) if limit is not None else num_batches
    if batches_to_run == 0:
        return 0.0

    pbar = tqdm(dataloader, total=batches_to_run, desc=f"Epoch {epoch_num}/{num_epochs_total} [Train]")
    for i, batch_data in enumerate(pbar):
        if limit is not None and i >= limit:
            break

        try:
            features, labels, _ = batch_data
        except ValueError as e:
            logger.error(f"Error unpacking batch {i}: {e}. Skipping batch.")
            continue

        features, labels = features.to(device), labels.to(device)

        try:
            optimizer.zero_grad()
            logits = model(features)

            output_len = logits.size(-1)
            label_len = labels.size(-1)
            min_len = min(output_len, label_len)

            if min_len <= 0:
                logger.warning(f"Skipping batch {i} due to zero/negative minimum length ({min_len}) between output ({output_len}) and labels ({label_len})")
                continue

            loss = criterion(logits[..., :min_len], labels[..., :min_len])

            if torch.isnan(loss):
                 logger.warning(f"NaN loss detected in batch {i}. Skipping backward pass.")
                 optimizer.zero_grad()
                 continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches_processed += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        except Exception as e:
            logger.exception(f"Error during training batch {i}: {e}. Skipping batch.")
            optimizer.zero_grad()

    if batches_processed == 0:
        logger.warning("No batches were processed successfully in train_one_epoch.")
        return 0.0
    return total_loss / batches_processed


def evaluate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   config: Dict[str, Any],
                   compute_event_metrics: bool = False,
                   epoch_num: Optional[int] = None,
                   num_epochs_total: Optional[int] = None,
                   limit: Optional[int] = None,
                   return_detailed: bool = False) -> Union[Tuple[float, Optional[Dict[float, Tuple[float, float, float]]]], 
                                                          Tuple[float, Optional[Dict[float, Tuple[float, float, float]]], List[Dict], List[Dict], List[str]]]:
    model.eval()
    total_loss = 0.0
    batches_processed = 0
    device = config["device"]
    hop_time = config["ssl_hop_length"] / config["sr"]
    threshold = config["threshold"]
    class_names = config["classes"]

    all_ground_truths = []
    all_predictions = []
    all_paths = []

    num_batches = len(dataloader)
    batches_to_run = min(limit, num_batches) if limit is not None else num_batches
    if batches_to_run == 0:
        return 0.0, None

    desc_prefix = f"Epoch {epoch_num}/{num_epochs_total} [Val]" if epoch_num else "[Test Eval]"
    pbar = tqdm(dataloader, total=batches_to_run, desc=desc_prefix)

    with torch.no_grad():
        for i, batch_data in enumerate(pbar):
            if limit is not None and i >= limit:
                break

            try:
                features, labels, paths = batch_data
            except ValueError as e:
                logger.error(f"Error unpacking batch {i} in eval: {e}. Skipping batch.")
                continue

            features, labels = features.to(device), labels.to(device)

            try:
                logits = model(features)
                probabilities = torch.sigmoid(logits)

                output_len = probabilities.size(-1)
                label_len = labels.size(-1)
                min_len = min(output_len, label_len)

                if min_len <= 0:
                    logger.warning(f"Skipping batch {i} in eval due to zero/negative minimum length ({min_len}) between output ({output_len}) and labels ({label_len})")
                    continue

                loss = criterion(logits[..., :min_len], labels[..., :min_len])

                if torch.isnan(loss):
                     logger.warning(f"NaN loss detected during evaluation batch {i}. Skipping loss accumulation for this batch.")
                else:
                     total_loss += loss.item()
                     batches_processed += 1
                     pbar.set_postfix(loss=f"{loss.item():.4f}")

                if compute_event_metrics or return_detailed:
                    probs_matched = probabilities[..., :min_len]
                    labels_matched = labels[..., :min_len]

                    for b_idx in range(probs_matched.size(0)):
                        gt_intervals = {}
                        label_np = labels_matched[b_idx].cpu().numpy()
                        for c_idx, class_name in enumerate(class_names):
                            gt_intervals[class_name] = sequence_to_intervals(label_np[c_idx], hop_time)
                        all_ground_truths.append(gt_intervals)

                        pred_intervals = model_output_to_intervals(
                            probs_matched[b_idx], threshold, hop_time, class_names
                        )
                        all_predictions.append(pred_intervals)
                        
                        if return_detailed:
                            all_paths.append(paths[b_idx])

            except Exception as e:
                 logger.exception(f"Error during evaluation batch {i}: {e}. Skipping batch.")

    if batches_processed == 0:
        logger.warning("No batches were processed successfully in evaluate_epoch.")
        avg_loss = 0.0
    else:
        avg_loss = total_loss / batches_processed

    event_metrics = None
    if compute_event_metrics and len(all_ground_truths) > 0:
        logger.info(f"Calculating event metrics for {len(all_ground_truths)} samples...")
        try:
             event_metrics = calculate_event_metrics(all_ground_truths, all_predictions, class_names)
        except Exception as e:
             logger.error(f"Error calculating event metrics: {e}")
             event_metrics = None

    if return_detailed:
        return avg_loss, event_metrics, all_ground_truths, all_predictions, all_paths

    return avg_loss, event_metrics

# -----------------------------------------------------------------------------
# Test Mode Function
# -----------------------------------------------------------------------------

def test_model(base_config: Dict[str, Any], experiment_config: Dict[str, Any], test_batch_limit: Optional[int] = None):
    # 1. 設定を組み合わせる
    config = {**base_config, **experiment_config}
    exp_name = config["name"]
    save_dir = Path(config["base_save_dir"]) / exp_name
    
    # ログ設定
    log_file = save_dir / f"{exp_name}_test_results.log"
    test_logger = logging.getLogger(f"{exp_name}_test")
    test_logger.propagate = False
    test_logger.setLevel(logging.INFO)
    
    if test_logger.hasHandlers():
        test_logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    test_logger.addHandler(fh)
    
    # コンソール出力も追加
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    test_logger.addHandler(sh)
    
    test_logger.info(f"=== テストモード: {exp_name} ===")
    test_logger.info(f"結果はこのパスに保存されます: {save_dir}")
    
    # 2. モデルの初期化
    try:
        model = EventDetector(
            config=config,
            ssl_model_name=config.get("ssl_model_name"),
            freeze_feature_extractor=config.get("freeze_fe", config["default_freeze_fe"]),
            freeze_transformer_layers=config.get("freeze_transformer", config["default_freeze_transformer"]),
            architecture=config.get("architecture", "gru"),
            feature_type=config.get("feature_type", "raw"),
        ).to(config["device"])
        test_logger.info("モデルが正常に初期化されました。")
        # パラメータ数を計算・ログ出力
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        test_logger.info(f"Total parameters: {num_params:,}")
        test_logger.info(f"Trainable parameters: {num_trainable:,}")
    except Exception as e:
        test_logger.exception(f"モデルの初期化に失敗しました: {e}")
        return
    
    # 3. テストデータのロード
    try:
        test_json_path = Path(config["test_json"])
        with open(test_json_path, 'r') as f:
            test_ann = json.load(f)
        
        dataset_test = SoundEventDataset(
            test_ann, config, config.get("feature_type", "raw"), 1.0
        )
        
        collate_fn = partial(pad_collate, config=config)
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
        
        dataloader_test = DataLoader(
            dataset_test, batch_size=config["batch_size"], shuffle=False, 
            collate_fn=collate_fn, num_workers=num_workers, 
            pin_memory=True, persistent_workers=num_workers > 0
        )
        test_logger.info(f"テストデータセットをロードしました。サンプル数: {len(dataset_test)}")
    except Exception as e:
        test_logger.exception(f"テストデータのロードに失敗しました: {e}")
        return
    
    # 4. 最良のモデルを読み込む
    best_ckpt_path = save_dir / "best.pth"
    if best_ckpt_path.exists():
        try:
            test_logger.info(f"最良モデルの重みを読み込んでいます: {best_ckpt_path}")
            model.load_state_dict(torch.load(best_ckpt_path, map_location=config["device"]))
        except Exception as e:
            test_logger.warning(f"最良モデルを読み込めませんでした: {e}。最後のモデル状態を使用します。")
    else:
        test_logger.warning(f"最良モデルのチェックポイントが見つかりません: {best_ckpt_path}")
        # 最新のモデルを読み込む試み
        latest_ckpt_path = save_dir / "latest.pth"
        if latest_ckpt_path.exists():
            try:
                test_logger.info(f"最新モデルの重みを読み込んでいます: {latest_ckpt_path}")
                checkpoint = torch.load(latest_ckpt_path, map_location=config["device"])
                model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                test_logger.warning(f"最新モデルを読み込めませんでした: {e}。初期状態のモデルを使用します。")
        else:
            test_logger.warning("チェックポイントが見つかりません。初期状態のモデルを使用します。")
    
    # 5. テスト実行
    criterion = nn.BCEWithLogitsLoss()
    test_logger.info("テスト評価を開始します...")
    test_start_time = datetime.datetime.now()
    inference_start = time.time()
    
    try:
        test_loss, standard_metrics, gt_list, pred_list, file_paths = evaluate_epoch(
            model, dataloader_test, criterion, config,
            compute_event_metrics=True,
            limit=test_batch_limit,
            return_detailed=True
        )
        inference_end = time.time()
        total_time = inference_end - inference_start
        num_samples = len(file_paths)
        avg_time = total_time / max(num_samples, 1)
        throughput = num_samples / total_time if total_time > 0 else float('inf')
        test_logger.info(f"Inference total time: {total_time:.3f}s for {num_samples} samples")
        test_logger.info(f"Average time per sample: {avg_time:.3f}s")
        test_logger.info(f"Throughput: {throughput:.2f} samples/sec")
        
        # 基本的なテスト結果
        test_end_time = datetime.datetime.now()
        test_duration = test_end_time - test_start_time
        test_logger.info(f"テスト評価完了（所要時間: {test_duration}）")
        test_logger.info(f"テスト損失: {test_loss:.4f}")
        
        # イベント統計情報を表示
        print_event_statistics(gt_list, pred_list, file_paths, config["classes"])
        
        # クラス別・食材別メトリクスの計算
        detailed_metrics = calculate_event_metrics_by_class_and_food(
            gt_list, pred_list, file_paths, config["classes"]
        )

        raw_iou = collect_raw_iou(gt_list, pred_list, config["classes"])
        # 全体の raw IoU（すべてのクラスの値をまとめて平均）
        all_vals = sum(raw_iou.values(), [])
        overall_raw_iou = float(np.mean(all_vals)) if all_vals else 0.0

        # クラス別平均 raw IoU
        avg_raw_iou_per_class = {
            cn: float(np.mean(vals)) if vals else 0.0
            for cn, vals in raw_iou.items()
        }

        # 食材別×クラス別 raw IoU
        raw_iou_by_food = collect_raw_iou_by_food(gt_list, pred_list, file_paths, config["classes"])
        avg_raw_iou_by_food = {}
        for food, cls_dict in raw_iou_by_food.items():
            avg_raw_iou_by_food[food] = {
                cn: float(np.mean(vals)) if vals else 0.0
                for cn, vals in cls_dict.items()
            }

        # TPマッチペアのIoUを計算
        matched_iou = collect_matched_iou(gt_list, pred_list, config['classes'])
        test_logger.info("=== Matched IoU (TPマッチペア平均) ===")
        for cn, vals in matched_iou.items():
            avg_iou = float(np.mean(vals)) if vals else 0.0
            test_logger.info(f"{cn.ljust(12)}: {avg_iou:.3f}")

        # --- ログ出力 ---
        test_logger.info("=== Raw IoU (生IoU) Summary ===")
        test_logger.info(f"Overall      : {overall_raw_iou:.3f}")
        for cn, avg in avg_raw_iou_per_class.items():
            test_logger.info(f"{cn.ljust(12)}: {avg:.3f}")

        test_logger.info("=== Raw IoU by Food Type ===")
        for food, cls_dict in avg_raw_iou_by_food.items():
            line = f"{food.ljust(10)}"
            for cn in config["classes"]:
                line += f" | {cls_dict.get(cn, 0.0):.3f}"
            test_logger.info(line)

        # --- コンソール出力 ---
        print("\nRaw IoU (生IoU) Summary")
        print(f"Overall      : {overall_raw_iou:.3f}")
        for cn, avg in avg_raw_iou_per_class.items():
            print(f"{cn.ljust(12)}: {avg:.3f}")

        print("\nRaw IoU by Food Type")
        header = "Food".ljust(10) + "".join(f"|{cn.center(12)}" for cn in config["classes"])
        print(header)
        for food, cls_dict in avg_raw_iou_by_food.items():
            row = food.ljust(10) + "".join(f"|{cls_dict.get(cn,0.0):12.3f}" for cn in config["classes"])
            print(row)
        
        # メトリクス結果の出力
        test_logger.info("\n=== クラス別・食材別評価結果 ===")
        
        # 全体結果を最初に表示
        test_logger.info("\n--- 全体結果 ---")
        for class_name, metrics in detailed_metrics["all"].items():
            test_logger.info(f"\n{class_name} クラス:")
            for iou_thr, (p, r, f1) in metrics.items():
                # EPSの表示を調整
                iou_thr_display = f"{iou_thr:.2f}" if iou_thr > EPS else f"~{EPS:.1e}"
                test_logger.info(f"  IoU > {iou_thr_display} | P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")
        
        # 食材別結果
        for food_type, class_metrics in detailed_metrics.items():
            if food_type == "all":
                continue
            
            test_logger.info(f"\n--- {food_type} ---")
            for class_name, class_metrics in class_metrics.items():
                test_logger.info(f"\n{class_name} クラス:")
                for iou_thr, (p, r, f1) in class_metrics.items():
                    # EPSの表示を調整
                    iou_thr_display = f"{iou_thr:.2f}" if iou_thr > EPS else f"~{EPS:.1e}"
                    test_logger.info(f"  IoU > {iou_thr_display} | P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")
        
        # IoU=0.5でのF1スコア比較表
        test_logger.info("\n=== IoU=0.5 での F1スコア比較 ===")
        comparison_table = "食材タイプ |"
        for class_name in config["classes"]:
            comparison_table += f" {class_name} |"
        test_logger.info(comparison_table)
        
        divider = "-" * len(comparison_table)
        test_logger.info(divider)
        
        # 全体行
        all_row = "全体 |"
        for class_name in config["classes"]:
            f1_score = detailed_metrics["all"][class_name][0.5][2]  # F1スコア
            all_row += f" {f1_score:.3f} |"
        test_logger.info(all_row)
        
        # 食材別行
        for food_type in sorted(detailed_metrics.keys()):
            if food_type == "all":
                continue
                
            if food_type not in detailed_metrics:
                continue
                
            food_row = f"{food_type} |"
            for class_name in config["classes"]:
                if class_name in detailed_metrics[food_type]:
                    f1_score = detailed_metrics[food_type][class_name][0.5][2]  # F1スコア
                    food_row += f" {f1_score:.3f} |"
                else:
                    food_row += " - |"
            test_logger.info(food_row)
        
        # 結果をJSONとして保存
        metrics_path = save_dir / "detailed_test_metrics.json"
        try:
            # 値を適切に変換
            serializable_metrics = {}
            for food_type, class_dict in detailed_metrics.items():
                serializable_metrics[food_type] = {}
                for class_name, metrics_dict in class_dict.items():
                    serializable_metrics[food_type][class_name] = {
                        str(k) if isinstance(k, float) and k == EPS else float(k): 
                        tuple(map(float, v)) for k, v in metrics_dict.items()
                    }
            
            with open(metrics_path, 'w') as f:
                json.dump({
                    "test_loss": float(test_loss),
                    "metrics": serializable_metrics,
                    "test_sample_count": len(file_paths)
                }, f, indent=2)
            test_logger.info(f"詳細な評価結果を保存しました: {metrics_path}")
        except Exception as e:
            test_logger.error(f"評価結果の保存に失敗しました: {e}")
        
    except Exception as e:
        test_logger.exception(f"テスト評価中にエラーが発生しました: {e}")
    
    test_logger.info("=== テスト完了 ===")


# -----------------------------------------------------------------------------  
# 3) 推論専用関数を新規追加（test_model の直後あたりに配置すると分かりやすい）  
# -----------------------------------------------------------------------------
def inference_model(base_config: Dict[str, Any],
                    experiment_config: Dict[str, Any],
                    batch_limit: Optional[int] = None) -> None:
    """
    指定モデルをテストセットに対して推論し、
    - test_inference/       : 〈start   end   label〉 の TSV
    - test_frame_probs/     : 〈time_s  cls1  cls2  ...〉 の TSV
    を出力する。
    """
    # --------------- 設定と出力フォルダ ---------------
    cfg       = {**base_config, **experiment_config}
    exp_name  = cfg["name"]
    save_dir  = Path(cfg["base_save_dir"]) / exp_name
    out_dir   = save_dir / "test_inference"
    prob_dir  = save_dir / "test_frame_probs"     # フレーム尤度出力
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    # --------------- モデル読込 ---------------
    model = EventDetector(
        config=cfg,
        ssl_model_name=cfg.get("ssl_model_name"),
        freeze_feature_extractor=cfg.get("freeze_fe", cfg["default_freeze_fe"]),
        freeze_transformer_layers=cfg.get("freeze_transformer", cfg["default_freeze_transformer"]),
        architecture=cfg.get("architecture", "gru"),
        feature_type=cfg.get("feature_type", "raw"),
    ).to(cfg["device"])

    best_ckpt = save_dir / "best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=cfg["device"]))
    else:
        logger.warning(f"[{exp_name}] best.pth が見つかりません（学習前かも）。初期重みで推論します。")

    # --------------- データ読込 ---------------
    with open(cfg["test_json"]) as f:
        test_ann = json.load(f)

    ds = SoundEventDataset(
        test_ann, cfg, cfg.get("feature_type", "raw")
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=partial(pad_collate, config=cfg),
        num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1),
        pin_memory=True,
        persistent_workers=True,
    )

    hop_time  = cfg["ssl_hop_length"] / cfg["sr"]
    threshold = cfg["threshold"]
    classes   = cfg["classes"]

    model.eval()
    with torch.no_grad():
        for feats, _, paths in tqdm(dl, desc=f"[{exp_name}] inference", total=len(dl)):
            feats = feats.to(cfg["device"])
            probs = torch.sigmoid(model(feats))          # (B, C, T)
            probs = probs.cpu()

            for b in range(probs.size(0)):
                wav_path = Path(paths[b])

                # ---------- 区間 (start, end, label) ----------
                intervals = model_output_to_intervals(
                    probs[b], threshold, hop_time, classes
                )

                lines = []
                for cls in classes:
                    short = LABEL_SHORT.get(cls, cls)
                    for st, ed in intervals[cls]:
                        lines.append(f"{st:.6f}\t{ed:.6f}\t{short}")
                lines.sort(key=lambda s: float(s.split()[0]))

                txt_path = out_dir / f"{wav_path.stem}.txt"
                with open(txt_path, "w") as fp:
                    fp.write("\n".join(lines))

                # ---------- フレーム毎の尤度 ----------
                prob_path = prob_dir / f"{wav_path.stem}_probs.tsv"
                header    = "time_s\t" + "\t".join(classes)
                prob_np   = probs[b].T.numpy()                 # (T, C)
                times     = np.arange(prob_np.shape[0]) * hop_time
                out_mat   = np.column_stack((times, prob_np))
                np.savetxt(
                    prob_path,
                    out_mat,
                    fmt="%.6f",
                    delimiter="\t",
                    header=header,
                    comments="",
                )

    logger.info(f"[{exp_name}] inference 完了: 区間→{out_dir.name}, 尤度→{prob_dir.name}")

# -----------------------------------------------------------------------------
# Main Experiment Runner Function
# -----------------------------------------------------------------------------

def expand_all_experiments(experiments_dict: Dict[str, List[Dict]]) -> List[Dict]:
    all_factors = list(experiments_dict.values())
    valid_factors = [f for f in all_factors if isinstance(f, list) and all(isinstance(item, dict) for item in f)]
    if len(valid_factors) != len(all_factors):
         logger.warning("Some items in EXPERIMENTS dict are not lists of dictionaries, skipping them for expansion.")

    if not valid_factors:
        return []

    combinations = list(itertools.product(*valid_factors))

    expanded_experiments = []
    seen_names = set()
    for combo in combinations:
        exp_config = {}
        for factor_dict in combo:
            exp_config.update(factor_dict)

        ssl_part = "no_ssl"
        if exp_config.get("ssl_model_name"):
            ssl_part = exp_config["ssl_model_name"].split('/')[-1]
        elif exp_config.get("feature_type"):
            ssl_part = exp_config["feature_type"]

        name_parts = [
            ssl_part,
            exp_config.get("architecture", "gru"),
            f"frac{exp_config.get('dataset_frac', '1.0')}"
        ]
        exp_name = "_".join(map(str, name_parts)).replace('-','_').replace('.','p')

        original_name = exp_name
        count = 1
        while exp_name in seen_names:
            exp_name = f"{original_name}_{count}"
            count += 1
        seen_names.add(exp_name)

        exp_config["name"] = exp_name
        expanded_experiments.append(exp_config)

    logger.info(f"Generated {len(expanded_experiments)} experiment configurations from 'all'.")
    return expanded_experiments

def train_and_evaluate(base_config: Dict[str, Any],
                       experiment_config: Dict[str, Any],
                       train_batch_limit: Optional[int] = None,
                       eval_batch_limit: Optional[int] = None):
    # 1. 設定を組み合わせる
    config = {**base_config, **experiment_config}
    exp_name = config["name"]
    save_dir = Path(config["base_save_dir"]) / exp_name
    log_path = save_dir / f"{exp_name}.log"

    # ① 既存ログから耐性カウントを復元
    patience_counter = load_patience_from_log(log_path, config["early_stopping_patience"])
    logger.info(f"[RESUME] {exp_name}: patience_counter={patience_counter}/{config['early_stopping_patience']}")

    # ② 既に耐性上限に達していればスキップ
    if patience_counter >= config["early_stopping_patience"]:
        logger.info(f"[SKIP] {exp_name}: already reached patience → test only")
        test_model(base_config, experiment_config)
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. ログ設定
    log_file = save_dir / f"{exp_name}.log"
    exp_logger = logging.getLogger(exp_name)
    exp_logger.propagate = False
    exp_logger.setLevel(logging.INFO)

    if exp_logger.hasHandlers():
         exp_logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    exp_logger.addHandler(fh)

    is_smoke_test = train_batch_limit is not None or eval_batch_limit is not None
    test_type = "Smoke Test" if is_smoke_test else "Experiment"
    exp_logger.info(f"--- Starting {test_type}: {exp_name} ---")
    if is_smoke_test:
        exp_logger.info(f"Smoke test limits: Train batches={train_batch_limit}, Eval batches={eval_batch_limit}")
    try:
        config_str = json.dumps(config, indent=2, default=str)
    except TypeError:
        config_str = str(config)
    exp_logger.info(f"Full Configuration: {config_str}")
    exp_logger.info(f"Results will be saved in: {save_dir}")
    exp_logger.info(f"Using device: {config['device']}")

    # 3. モデル初期化
    try:
        model = EventDetector(
            config=config,
            ssl_model_name=config.get("ssl_model_name"),
            freeze_feature_extractor=config.get("freeze_fe", config["default_freeze_fe"]),
            freeze_transformer_layers=config.get("freeze_transformer", config["default_freeze_transformer"]),
            architecture=config.get("architecture", "gru"),
            feature_type=config.get("feature_type", "raw"),
        ).to(config["device"])
        exp_logger.info("Model initialized successfully.")
        # パラメータ数の出力
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        exp_logger.info(f"Total parameters: {num_params:,}")
        exp_logger.info(f"Trainable parameters: {num_trainable_params:,}")

    except Exception as e:
        exp_logger.exception(f"Failed to initialize model: {e}")
        return

    if config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    elif config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], momentum=config["momentum"])
    else:
        exp_logger.error(f"Unsupported optimizer: {config['optimizer']}")
        return
    criterion = nn.BCEWithLogitsLoss()

    # 4. データ読み込み
    try:
        train_json_path = Path(config["train_json"])
        val_json_path = Path(config["val_json"])
        test_json_path = Path(config["test_json"])

        with open(train_json_path, 'r') as f: train_ann = json.load(f)
        with open(val_json_path, 'r') as f:   val_ann = json.load(f)
        with open(test_json_path, 'r') as f:  test_ann = json.load(f)

        dataset_train = SoundEventDataset(
            train_ann, config, config.get("feature_type", "raw"), config.get("dataset_frac", 1.0)
        )
        dataset_val = SoundEventDataset(val_ann, config, config.get("feature_type", "raw"), 1.0)
        dataset_test = SoundEventDataset(test_ann, config, config.get("feature_type", "raw"), 1.0)

        collate_fn = partial(pad_collate, config=config)

        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
        exp_logger.info(f"Using {num_workers} workers for DataLoaders.")

        dataloader_train = DataLoader(
            dataset_train, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
        dataloader_val = DataLoader(
            dataset_val, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
        dataloader_test = DataLoader(
            dataset_test, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
        exp_logger.info("Datasets and DataLoaders created successfully.")
        exp_logger.info(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}, Test samples: {len(dataset_test)}")
        
        try:
             feat_sample, label_sample, _ = next(iter(dataloader_train))
             exp_logger.info(f"Sample batch feature shape: {feat_sample.shape}")
             exp_logger.info(f"Sample batch label shape: {label_sample.shape}")
        except Exception as e:
             exp_logger.warning(f"Could not get sample batch shape: {e}")

    except FileNotFoundError as e:
        exp_logger.error(f"Data JSON file not found: {e}")
        return
    except Exception as e:
        exp_logger.exception(f"Failed to load data: {e}")
        return

    # 5. チェックポイント読み込み
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_path = save_dir / "latest.pth"
    best_ckpt_path = save_dir / "best.pth"

    if not is_smoke_test and ckpt_path.exists():
        try:
            exp_logger.info(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=config["device"])
            model.load_state_dict(checkpoint["model_state_dict"])
            
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    exp_logger.info("Optimizer state loaded successfully.")
                except ValueError as e:
                    exp_logger.warning(f"Could not load optimizer state dict: {e}. Continuing without loading optimizer state.")
            else:
                exp_logger.warning("Optimizer state not found in checkpoint.")

            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            exp_logger.info(f"Resumed training from epoch {start_epoch + 1}. Best validation loss so far: {best_val_loss:.4f}")
        except Exception as e:
            exp_logger.warning(f"Could not load checkpoint from {ckpt_path}, starting from scratch. Error: {e}")
            start_epoch = 0
            best_val_loss = float("inf")

    # 6. 学習ループ
    patience_counter = 0
    num_epochs = config["num_epochs"]
    early_stopping_patience = config["early_stopping_patience"]

    exp_logger.info(f"Starting training loop from epoch {start_epoch + 1} to {num_epochs}")
    training_start_time = datetime.datetime.now()

    for epoch in range(start_epoch, num_epochs):
        current_epoch_num = epoch + 1
        epoch_start_time = datetime.datetime.now()

        # 学習
        train_loss = train_one_epoch(
            model, dataloader_train, criterion, optimizer, config["device"],
            current_epoch_num, num_epochs,
            limit=train_batch_limit
        )

        # 検証
        val_loss, _ = evaluate_epoch(
            model, dataloader_val, criterion, config, compute_event_metrics=False,
            epoch_num=current_epoch_num, num_epochs_total=num_epochs,
            limit=eval_batch_limit
        )

        epoch_end_time = datetime.datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        exp_logger.info(f"Epoch {current_epoch_num}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Duration: {epoch_duration}")
        print(f"[{exp_name} - E{current_epoch_num}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration}")

        # チェックポイント保存と早期終了
        if not is_smoke_test:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                try:
                    torch.save(model.state_dict(), best_ckpt_path)
                    exp_logger.info(f"New best model saved to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
                except Exception as e:
                    exp_logger.error(f"Failed to save best model checkpoint: {e}")
            else:
                patience_counter += 1
                exp_logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

            # 最新のチェックポイント保存
            try:
                torch.save({
                    'epoch': current_epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config
                }, ckpt_path)
            except Exception as e:
                 exp_logger.error(f"Failed to save latest checkpoint: {e}")

            if patience_counter >= early_stopping_patience:
                exp_logger.info(f"Early stopping triggered at epoch {current_epoch_num}.")
                print(f"[{exp_name}] Early stopping triggered.")
                break
        else:
             exp_logger.info("Smoke test epoch finished.")
             break

    training_end_time = datetime.datetime.now()
    total_training_time = training_end_time - training_start_time
    exp_logger.info(f"Training loop finished. Total training time: {total_training_time}")

    # 7. テストセットでの最終評価
    exp_logger.info("Starting final evaluation on the test set.")
    test_start_time = datetime.datetime.now()
    
    if not is_smoke_test:
        if best_ckpt_path.exists():
            try:
                exp_logger.info(f"Loading best model weights from {best_ckpt_path} for testing.")
                model.load_state_dict(torch.load(best_ckpt_path, map_location=config["device"]))
            except Exception as e:
                exp_logger.warning(f"Could not load best checkpoint from {best_ckpt_path}. Using model from last epoch. Error: {e}")
        else:
             exp_logger.warning(f"Best checkpoint {best_ckpt_path} not found. Using model from last epoch for testing.")

    try:
        test_loss, test_metrics, gt_list, pred_list, file_paths = evaluate_epoch(
            model, dataloader_test, criterion, config,
            compute_event_metrics=True, 
            limit=eval_batch_limit,
            return_detailed=True
        )

        test_end_time = datetime.datetime.now()
        test_duration = test_end_time - test_start_time
        exp_logger.info(f"--- Test Set Results (Duration: {test_duration}) ---")
        exp_logger.info(f"Test Loss: {test_loss:.4f}")
        print(f"\n[{exp_name} - TEST] Loss: {test_loss:.4f} | Eval Time: {test_duration}")

        print_event_statistics(gt_list, pred_list, file_paths, config["classes"])
        
        detailed_metrics = calculate_event_metrics_by_class_and_food(
            gt_list, pred_list, file_paths, config["classes"]
        )
        
        exp_logger.info("\n--- クラス別結果 ---")
        for class_name, metrics in detailed_metrics["all"].items():
            exp_logger.info(f"\n{class_name} クラス:")
            for iou_thr, (p, r, f1) in metrics.items():
                iou_thr_display = f"{iou_thr:.2f}" if iou_thr > EPS else f"~{EPS:.1e}"
                log_msg = f"  IoU > {iou_thr_display} | P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}"
                exp_logger.info(log_msg)
                print(log_msg)
        
        if test_metrics:
            metrics_path = save_dir / "test_metrics.json"
            try:
                with open(metrics_path, 'w') as f:
                    serializable_metrics = {
                        str(k) if k == EPS else float(k): tuple(map(float, v))
                        for k, v in test_metrics.items()
                    }
                    json.dump({
                        "test_loss": float(test_loss), 
                        "event_metrics": serializable_metrics,
                        "metrics_by_class": {
                            class_name: {
                                str(k) if k == EPS else float(k): tuple(map(float, v))
                                for k, v in metrics.items()
                            } for class_name, metrics in detailed_metrics["all"].items()
                        }
                    }, f, indent=2)
                exp_logger.info(f"Test metrics saved to {metrics_path}")
            except Exception as e:
                exp_logger.error(f"Failed to save test metrics to {metrics_path}: {e}")
            
            detailed_metrics_path = save_dir / "detailed_class_food_metrics.json"
            try:
                serializable_detailed = {}
                for food_type, class_dict in detailed_metrics.items():
                    serializable_detailed[food_type] = {}
                    for class_name, metrics_dict in class_dict.items():
                        serializable_detailed[food_type][class_name] = {
                            str(k) if isinstance(k, float) and k == EPS else float(k): 
                            tuple(map(float, v)) for k, v in metrics_dict.items()
                        }
                
                with open(detailed_metrics_path, 'w') as f:
                    json.dump(serializable_detailed, f, indent=2)
                exp_logger.info(f"Detailed class and food metrics saved to {detailed_metrics_path}")
            except Exception as e:
                exp_logger.error(f"Failed to save detailed metrics to {detailed_metrics_path}: {e}")
        else:
            exp_logger.warning("Event metrics calculation did not return results for the test set.")

    except Exception as e:
        exp_logger.exception(f"An error occurred during final test evaluation: {e}")

    exp_logger.info(f"--- {test_type} {exp_name} Finished ---")



# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Sound Event Detection Experiment")
    parser.add_argument("--test", action="store_true", help="Run in test mode only (no training)")
    parser.add_argument("--model", type=str, help="Specific model name to test (default: all)", default=None)
    parser.add_argument("--inference", action="store_true", help="Run inference-only mode: write predicted intervals for each test file")
    args = parser.parse_args()

    script_start_time = datetime.datetime.now()
    logger.info(f"Script started at: {script_start_time}")

    # 実験の選択
    if EXPERIMENT_GROUP == "all":
        logger.info("Running ALL experiment combinations.")
        list_of_experiments = expand_all_experiments(EXPERIMENTS)
    elif EXPERIMENT_GROUP in EXPERIMENTS:
        logger.info(f"Running experiment group: {EXPERIMENT_GROUP}")
        list_of_experiments = EXPERIMENTS[EXPERIMENT_GROUP]
    else:
        logger.error(f"Unknown EXPERIMENT_GROUP: '{EXPERIMENT_GROUP}'. Available groups: {list(EXPERIMENTS.keys())} or 'all'.")
        sys.exit(1)

    # Inference mode
    if args.inference:
        logger.info("=== Inference-only mode ===")
        if args.model:
            list_of_experiments = [e for e in list_of_experiments if e["name"] == args.model]
            if not list_of_experiments:
                logger.error(f"Model '{args.model}' not found.")
                sys.exit(1)

        for i, exp_conf in enumerate(list_of_experiments):
            exp_name = exp_conf["name"]
            print(f"\n=== Inference {i+1}/{len(list_of_experiments)}: {exp_name} ===")
            inference_model(BASE_CONFIG, exp_conf)
        logger.info("All inference runs finished.")
        sys.exit(0)

    # 特定モデルのテスト用フィルタ
    if args.model and args.test:
        list_of_experiments = [e for e in list_of_experiments if e["name"] == args.model]
        if not list_of_experiments:
            logger.error(f"Model '{args.model}' not found in experiment list.")
            sys.exit(1)

    if not list_of_experiments:
        logger.warning(f"No experiments defined or generated for group '{EXPERIMENT_GROUP}'. Exiting.")
        sys.exit(0)

    logger.info(f"Found {len(list_of_experiments)} experiments to run.")

    # テストモード
    if args.test:
        logger.info("=== テストモードで実行します ===")
        for i, exp_conf in enumerate(list_of_experiments):
            exp_name = exp_conf.get("name", f"unnamed_exp_{i+1}")
            print(f"\n=== テスト実行 {i+1}/{len(list_of_experiments)}: {exp_name} ===")
            test_model(BASE_CONFIG, exp_conf)
        logger.info("全てのテスト実行が完了しました。")
        sys.exit(0)

    # スモークテスト
    run_smoke_test = True
    smoke_test_batch_limit = 50
    if run_smoke_test:
        logger.info(f"--- Running Smoke Tests (1 epoch, limited to {smoke_test_batch_limit} batches/step) ---")
        smoke_config = {
            **BASE_CONFIG,
            "num_epochs": 1,
            "base_save_dir": BASE_CONFIG["base_save_dir"] + "_smoke_tests",
        }
        all_smoke_tests_passed = True
        smoke_test_start_time = datetime.datetime.now()

        for i, exp_conf in enumerate(list_of_experiments):
            exp_name = exp_conf.get("name", f"unnamed_exp_{i+1}")
            smoke_save_dir = Path(smoke_config["base_save_dir"]) / exp_name
            smoke_log = smoke_save_dir / f"{exp_name}.log"

            if has_smoke_test_finished(smoke_log):
                print(f"[SKIP] Smoke test already finished for {exp_name}.")
                continue

            print(f"\n[SMOKE {i+1}/{len(list_of_experiments)}] {exp_name}")
            logger.info(f"Starting smoke test for: {exp_name}")
            try:
                train_and_evaluate(
                    smoke_config,
                    exp_conf,
                    train_batch_limit=smoke_test_batch_limit,
                    eval_batch_limit=smoke_test_batch_limit
                )
            except Exception as e:
                smoke_exp_logger = logging.getLogger(exp_name)
                if not smoke_exp_logger.hasHandlers():
                    smoke_exp_logger = logger
                smoke_exp_logger.exception(f"!!! Smoke test FAILED for {exp_name}: {e}")
                print(f"[ERROR] Smoke test failed for {exp_name}. Check logs in {smoke_save_dir}/{exp_name}.log")
                all_smoke_tests_passed = False

        smoke_test_end_time = datetime.datetime.now()
        logger.info(f"--- Smoke Tests Complete (Duration: {smoke_test_end_time - smoke_test_start_time}) ---")
        if not all_smoke_tests_passed:
            print("\n[ERROR] One or more smoke tests failed. Please check the logs.")
            sys.exit(1)
        else:
            print(f"\n[INFO] All smoke tests ({smoke_test_batch_limit} batches/step) passed successfully.")
            print("[INFO] Proceeding to full experiment runs...")

    # 本番実験
    logger.info("--- Starting Full Experiment Runs ---")
    full_runs_start_time = datetime.datetime.now()
    for i, exp_conf in enumerate(list_of_experiments):
        exp_name = exp_conf.get("name", f"unnamed_exp_{i+1}")
        print(f"\n=== Running Experiment {i+1}/{len(list_of_experiments)}: {exp_name} ===")
        logger.info(f"Starting full run for experiment: {exp_name}")
        try:
            train_and_evaluate(BASE_CONFIG, exp_conf)
        except Exception as e:
            exp_logger = logging.getLogger(exp_name)
            if not exp_logger.hasHandlers():
                exp_logger = logger
            exp_logger.exception(f"!!! Experiment '{exp_name}' failed during full run: {e}")
            print(f"[ERROR] Experiment '{exp_name}' failed. Check logs in {BASE_CONFIG['base_save_dir']}/{exp_name}/{exp_name}.log. Continuing...")

    full_runs_end_time = datetime.datetime.now()
    logger.info(f"--- All Full Experiments Finished (Duration: {full_runs_end_time - full_runs_start_time}) ---")
    script_end_time = datetime.datetime.now()
    logger.info(f"Script finished at: {script_end_time}")
    logger.info(f"Total script execution time: {script_end_time - script_start_time}")
    print("\n[INFO] All experiments complete.")
