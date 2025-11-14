import torch
import numpy as np
import pandas as pd
import librosa
from audiomentations import Compose
from typing import List, Dict, Optional


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        filepath_col: str,
        target_col: str,
        class_names: List[str],
        sample_rate: int,
        target_duration: float,
        normalize_audio: bool = True,
        mixup_params: Optional[Dict] = None,
        is_train: bool = True,
        wave_piece: str = "center",
        audio_transforms: Optional[Compose] = None,
    ) -> None:

        self.df = input_df.reset_index(drop=True)

        self.filepath_col = filepath_col
        self.target_col = target_col

        self.sample_rate = sample_rate
        self.target_duration = target_duration

        self.target_sample_count = int(sample_rate * target_duration)
        self.normalize_audio = normalize_audio
        self.is_train = is_train
        self.wave_piece = wave_piece
        assert wave_piece in ("center", "random")

        self.class_names = class_names
        self.n_classes = len(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.audio_transforms = audio_transforms
        self.mixup_audio = mixup_params and is_train
        self.mixup_params = mixup_params or {
            "prob": 0.0,
            "alpha": 0.5,
            "hard_target": False # not multiclass
        }

    def __len__(self):
        return len(self.df)

    def _get_wave(self, idx: int) -> np.ndarray:
        filepath = self.df[self.filepath_col].iloc[idx]
        wave, _ = librosa.load(filepath, sr=self.sample_rate)
        return wave

    def _processes_wave(self, wave: np.ndarray) -> np.ndarray:
        length = len(wave)
        if length < self.target_sample_count:
            wave = np.pad(wave, (0, self.target_sample_count - length), mode="constant")
        else:
            if self.wave_piece == "center":
                start = max(0, (length - self.target_sample_count)//2)
            else:
                start = np.random.randint(0, length - self.target_sample_count + 1)
            wave = wave[start:start+self.target_sample_count]
        return wave

    def _get_mixup_idx(self):
        return np.random.randint(0, len(self.df))

    def _prepare_target(self, idx: int, sec_idx: Optional[int] = None):
        y1 = np.zeros(self.n_classes, dtype=np.float32)
        cls1 = self.df[self.target_col].iloc[idx]
        y1[self.class_to_idx[cls1]] = 1
        if sec_idx is None:
            return y1
        y2 = np.zeros(self.n_classes, dtype=np.float32)
        cls2 = self.df[self.target_col].iloc[sec_idx]
        y2[self.class_to_idx[cls2]] = 1
        alpha = self.mixup_params["alpha"]
        y_mix = alpha * y1 + (1 - alpha) * y2
        return y_mix if not self.mixup_params["hard_target"] else (y_mix > 0).astype(float)

    def __getitem__(self, idx: int):
        wave = self._get_wave(idx)
        wave = self._processes_wave(wave)
        if self.mixup_audio and np.random.rand() < self.mixup_params["prob"]:
            sec_idx = self._get_mixup_idx()
            sec_wave = self._get_wave(sec_idx)
            sec_wave = self._processes_wave(sec_wave)
            alpha = self.mixup_params["alpha"]
            wave = alpha * wave + (1 - alpha) * sec_wave
            target = self._prepare_target(idx, sec_idx)
        else:
            target = self._prepare_target(idx)
        if self.audio_transforms and self.is_train:
            wave = self.audio_transforms(samples=wave, sample_rate=self.sample_rate)
        if self.normalize_audio:
            wave = librosa.util.normalize(wave)
        return torch.from_numpy(wave).float(), torch.from_numpy(target).float()
