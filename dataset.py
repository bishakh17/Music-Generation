import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class AudioDataset(Dataset):

    def __init__(self, audio_dir, transformation, target_sample_rate, num_samples, device="cpu"):
        self.audio_dir = audio_dir
        # self.device = device
        # self.transformation = transformation.to(self.device)
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        for (root,dirs,files) in os.walk(self.audio_dir):                           
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        file = self.files[index]
        path = os.path.join(self.audio_dir,file)
        return path



if __name__ == "__main__":
    AUDIO_DIR = "/Users/bdboy/Desktop/Projects/Music-Generation/data/drums/test"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 8000

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using device {DEVICE}")

    TRANSFORM = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = AudioDataset(AUDIO_DIR, TRANSFORM, SAMPLE_RATE, NUM_SAMPLES, DEVICE)

    print(f"There are {len(usd)} samples in the dataset.")
    signal = usd[0]