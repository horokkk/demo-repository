import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def spec_augment_torch(mfcc,
                       time_mask_param=20,
                       freq_mask_param=8,
                       num_time_masks=2,
                       num_freq_masks=2):
    """
    mfcc: torch.Tensor, shape (40, 174)
    """
    v = mfcc.clone()
    n_mels, n_frames = v.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = torch.randint(low=1, high=freq_mask_param + 1, size=(1,)).item()
        if f >= n_mels:
            continue
        f0_max = max(1, n_mels - f)
        f0 = torch.randint(low=0, high=f0_max, size=(1,)).item()
        v[f0:f0+f, :] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t = torch.randint(low=1, high=time_mask_param + 1, size=(1,)).item()
        if t >= n_frames:
            continue
        t0_max = max(1, n_frames - t)
        t0 = torch.randint(low=0, high=t0_max, size=(1,)).item()
        v[:, t0:t0+t] = 0.0

    return v


class MFCCDatasetAug(Dataset):
    def __init__(self, X, y, train=True):
        """
        X: numpy array
           - (N, 40, 174) 또는 (N, 40, 174, 1)
        """
        X = np.asarray(X)

        # (N, 40, 174, 1)인 경우 채널 제거
        if X.ndim == 4:
            X = X[..., 0]

        if X.ndim != 3:
            raise ValueError(f"X shape should be (N,40,174) or (N,40,174,1), got {X.shape}")

        self.X = X.astype(np.float32)    # (N, 40, 174)
        self.y = y.astype(np.int64)
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mfcc = self.X[idx]           # (40, 174)
        label = self.y[idx]

        mfcc_t = torch.from_numpy(mfcc)  # (40, 174)

        # train일 때만 SpecAugment 적용
        if self.train:
            mfcc_t = spec_augment_torch(mfcc_t)

        # CNN 입력 shape: (C, H, W) = (1, 40, 174)
        mfcc_t = mfcc_t.unsqueeze(0)  # (1, 40, 174)

        return mfcc_t, label
