import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader


def split_and_normalize(SAVE_PATH: str, METADATA_PATH: str):
    # Load the preprocessed data
    x = np.load(os.path.join(SAVE_PATH, "x.npy"))
    y = np.load(os.path.join(SAVE_PATH, "y.npy"))

    metadata = pd.read_csv(METADATA_PATH)  # UrbanSound8K.csv
    folds = metadata["fold"].values        # fold info (1~10)

    # Classify by index
    train_idx = np.where(np.isin(folds, [1,2,3,4,5,6,7,8]))[0]
    val_idx   = np.where(folds == 9)[0]
    test_idx  = np.where(folds == 10)[0]

    # Dataset Split
    x_train = x[train_idx]
    y_train = y[train_idx]

    x_val = x[val_idx]
    y_val = y[val_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    print("Train:", x_train.shape, y_train.shape) # ~80%
    print("Val:  ", x_val.shape, y_val.shape) # ~9%
    print("Test: ", x_test.shape, y_test.shape) # ~10%
    
    # normalize
    
    mean=x_train.mean()
    std=x_train.std()
    
    x_train_norm = (x_train - mean) / std
    x_val_norm   = (x_val   - mean )/ std
    x_test_norm  = (x_test -  mean )/ std
    
    print("normed shapes:", x_train_norm.shape, x_val_norm.shape, x_test_norm.shape)

    return x_train_norm, x_val_norm, x_test_norm, y_train, y_val, y_test