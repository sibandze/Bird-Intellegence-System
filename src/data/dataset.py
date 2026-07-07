import random
import torch
import pandas as pd
import numpy as np
from .process_audio import load_local_spectrogram

class BirdSongDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, segment_size=187, train=True, label_to_idx=None, 
                 freq_mask_param=15, time_mask_param=25):
        self.df = df.reset_index(drop=True)
        self.segment_size = segment_size
        self.train = train
        
        # SpecAugment hyperparameters (maximum widths of the masks)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        species_df = df[['scientific_name_id', 'scientific_name']].drop_duplicates().sort_values('scientific_name_id')

        if label_to_idx is None:
            self.label_to_idx = {row.scientific_name: int(row.scientific_name_id) for _, row in species_df.iterrows()}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = load_local_spectrogram(row['local_spectrogram_path']) # (n_mels, T)

        # Global Min-Max Normalization to bring dB values into safe [0, 1] range
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max > mel_min:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = np.zeros_like(mel)

        T = mel.shape[1]
        if T > self.segment_size:
            start = random.randint(0, T - self.segment_size) if self.train else (T - self.segment_size) // 2
            mel_segment = mel[:, start:start+self.segment_size]
        else:
            pad = self.segment_size - T
            mel_segment = np.pad(mel, ((0,0),(0,pad)), mode='constant')

        # Convert to tensor
        mel_segment = torch.from_numpy(mel_segment).float() # Shape: (n_mels, segment_size)
        
        # --- SPEC_AUGMENT PIPELINE ---
        # Only apply masking during training so evaluation remain pristine and deterministic
        if self.train:
            n_mels, n_frames = mel_segment.shape
            
            # 1. Frequency Masking (horizontal striping)
            # Pick a random mask width up to freq_mask_param, then choose a valid starting coordinate
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, n_mels - f)
            mel_segment[f0:f0+f, :] = 0.0
            
            # 2. Time Masking (vertical striping)
            # Pick a random mask width up to time_mask_param, then choose a valid starting coordinate
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, n_frames - t)
            mel_segment[:, t0:t0+t] = 0.0
        # -----------------------------

        label = torch.tensor(int(row['scientific_name_id'])).long()

        return mel_segment, label
