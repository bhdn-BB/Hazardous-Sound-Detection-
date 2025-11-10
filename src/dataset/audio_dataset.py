import torch
import numpy as np
import pandas as pd
import librosa
from audiomentations import Compose
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
        use_cache: bool = True,
        wave_piece: str = "center",
        cache_n_samples: int = 0,
        audio_transforms: Optional[Compose] = None,
    ):
        self.df = input_df.reset_index(drop=True)

        self.filepath_col = filepath_col
        self.target_col = target_col
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_sample_count = int(sample_rate * target_duration)
        self.normalize_audio = normalize_audio
        self.is_train = is_train
        self.use_cache = use_cache
        self.wave_piece = wave_piece
        assert wave_piece in ("center", "random")

        self.class_names = class_names
        self.n_classes = len(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.audio_transforms = audio_transforms

        self.mixup_audio = mixup_params is not None and is_train
        self.mixup_params = mixup_params or {
            "prob": 0.0,
            "alpha": 0.5,
            "hard_target": True
        }

        if self.use_cache and cache_n_samples > 0:
            self._cache_samples(top_n=cache_n_samples)

    def __len__(self):
        return len(self.df)

    def _cache_samples(self, top_n: int):
        def load_wave(idx):
            return self._get_wave(idx)

        self.df["wave"] = None
        idx_to_cache = self.df.index[:top_n]
        with ThreadPoolExecutor() as executor:
            waves = list(tqdm(executor.map(load_wave, idx_to_cache),
                              total=len(idx_to_cache), desc="Caching audio"))
        for i, idx in enumerate(idx_to_cache):
            self.df.at[idx, "wave"] = waves[i]

    def _get_wave(self, idx: int) -> np.ndarray:
        filepath = self.df[self.filepath_col].iloc[idx]
        wave, sr = librosa.load(filepath, sr=self.sample_rate)
        assert len(wave.shape) == 1, f"Expected mono audio, got {wave.shape}"
        return wave

    def _prepare_sample_piece(self, wave: np.ndarray) -> np.ndarray:
        length = len(wave)
        if length < self.target_sample_count:
            wave = np.pad(wave, (0, self.target_sample_count - length), mode="constant")
        else:
            if self.wave_piece == "center":
                start = max(0, (length - self.target_sample_count)//2)
            else:  # random
                start = np.random.randint(0, length - self.target_sample_count + 1)
            wave = wave[start:start+self.target_sample_count]
        return wave

    def _get_mixup_idx(self):
        return np.random.randint(0, len(self.df))

    def _prepare_target(self, idx: int, sec_idx: Optional[int] = None):
        y1 = np.zeros(self.n_classes, dtype=np.float32)
        cls1 = self.df[self.target_col].iloc[idx]
        y1[self.class_to_idx[cls1]] = 1.0

        if sec_idx is None:
            return y1

        y2 = np.zeros(self.n_classes, dtype=np.float32)
        cls2 = self.df[self.target_col].iloc[sec_idx]
        y2[self.class_to_idx[cls2]] = 1.0

        alpha = self.mixup_params.get("alpha", 0.5)
        y_mix = alpha * y1 + (1 - alpha) * y2
        return y_mix if not self.mixup_params.get("hard_target", True) else (y_mix > 0).astype(float)

    def _prepare_sample(self, idx: int):
        if self.use_cache:
            wave = self.df["wave"].iloc[idx]
            if wave is None:
                wave = self._get_wave(idx)
        else:
            wave = self._get_wave(idx)

        wave = self._prepare_sample_piece(wave)

        # Mixup
        if self.mixup_audio and np.random.rand() < self.mixup_params.get("prob", 0.0):
            sec_idx = self._get_mixup_idx()
            sec_wave = self.df["wave"].iloc[sec_idx] if self.use_cache else self._get_wave(sec_idx)
            sec_wave = self._prepare_sample_piece(sec_wave)

            alpha = self.mixup_params.get("alpha", 0.5)
            wave = alpha * wave + (1 - alpha) * sec_wave
            target = self._prepare_target(idx, sec_idx)
        else:
            target = self._prepare_target(idx)

        if self.audio_transforms and self.is_train:
            wave = self.audio_transforms(samples=wave, sample_rate=self.sample_rate)

        if self.normalize_audio:
            wave = librosa.util.normalize(wave)

        return torch.from_numpy(wave).float(), torch.from_numpy(target).float()

    def __getitem__(self, idx: int):
        return self._prepare_sample(idx)
