from src.data.utils.download import download_dataset
from src.data.utils.preprocess import extract_features
from src.data.utils.preprocess import extract_features
from src.data.utils.split_normalize import split_and_normalize
from src.data.utils.augment import MFCCDatasetAug

import kagglehub
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def run_preprocessing(
    SAVE_PATH: str = "./input_data",
    n_mfcc: int = 40,
    max_len: int = 174
):
    
    path = download_dataset(SAVE_PATH)
    # 2. Preprocess
    # Define variables to extract features
    AUDIO_PATH = path
    METADATA_PATH = os.path.join(path, "UrbanSound8K.csv")
    
    # Define variables for preprocessing 
    metadata = pd.read_csv(METADATA_PATH)
    features = []
    labels = []

    print("\nStart proprocessing...")

    # Full data preprocessing
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):

        # file directory
        file_name = os.path.join(AUDIO_PATH, 'fold' + str(row["fold"]), str(row["slice_file_name"]))
        class_label = row["classID"]

        data = extract_features(file_name, n_mfcc=n_mfcc, max_len=max_len)

        if data is not None:
            features.append(data)
            labels.append(class_label)

    # Save the preprocessed result
    x = np.array(features)
    y = np.array(labels)

    print("\nSaving the preprocessed data...")
    np.save(os.path.join(SAVE_PATH, "x.npy"), x)
    np.save(os.path.join(SAVE_PATH, "y.npy"), y)
    print("Save complete!")

    print(f"X shape: {x.shape}")  # (num_audio_files, num_mfcc_coeffs, num_time_frames) -> audio represented as a matrix
    print(f"Y shape: {y.shape}")  # each audio file is represented by a class label (integer) -> 1D


    # 3. Split and normalize
    x_train_norm, x_val_norm, x_test_norm, y_train, y_val, y_test = split_and_normalize(SAVE_PATH, METADATA_PATH)

    np.save(os.path.join(SAVE_PATH, "x_train_norm.npy"), x_train_norm)
    np.save(os.path.join(SAVE_PATH, "x_val_norm.npy"),   x_val_norm)
    np.save(os.path.join(SAVE_PATH, "x_test_norm.npy"),  x_test_norm)
    np.save(os.path.join(SAVE_PATH, "y_train.npy"),      y_train)
    np.save(os.path.join(SAVE_PATH, "y_val.npy"),        y_val)
    np.save(os.path.join(SAVE_PATH, "y_test.npy"),       y_test)
    
    print("\nNormalized splits saved.")
    print("Train:", x_train_norm.shape, y_train.shape)
    print("Val:  ", x_val_norm.shape,   y_val.shape)
    print("Test: ", x_test_norm.shape,  y_test.shape)
    
    
def prepare_data(SAVE_PATH: str = "./input_data"):
    
    needed_files = [
        "x_train_norm.npy", "x_val_norm.npy", "x_test_norm.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
    ]
    if not all(os.path.exists(os.path.join(SAVE_PATH, f)) for f in needed_files):
        print("Preprocessed files not found. Running full preprocessing pipeline...")
        run_preprocessing(SAVE_PATH=SAVE_PATH)
    else:
        print("Found preprocessed files. Loading from disk...")

    x_train_norm = np.load(os.path.join(SAVE_PATH, "x_train_norm.npy"))
    x_val_norm   = np.load(os.path.join(SAVE_PATH, "x_val_norm.npy"))
    x_test_norm  = np.load(os.path.join(SAVE_PATH, "x_test_norm.npy"))
    y_train      = np.load(os.path.join(SAVE_PATH, "y_train.npy"))
    y_val        = np.load(os.path.join(SAVE_PATH, "y_val.npy"))
    y_test       = np.load(os.path.join(SAVE_PATH, "y_test.npy"))

    return x_train_norm, x_val_norm, x_test_norm, y_train, y_val, y_test

def get_data_loader(SAVE_PATH: str = "./input_data", batch_size: int=64):
    
    x_train_norm, x_val_norm, x_test_norm, y_train, y_val, y_test = prepare_data(
        SAVE_PATH=SAVE_PATH
    )
    # 4. Augment    
    train_ds = MFCCDatasetAug(x_train_norm, y_train, train=True)
    val_ds   = MFCCDatasetAug(x_val_norm,   y_val,   train=False)
    test_ds  = MFCCDatasetAug(x_test_norm,  y_test,  train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Sanity check
    xb, yb = next(iter(train_loader))
    print("batch X:", xb.shape)   # Expected shape: (64, 1, 40, 174)
    print("batch y:", yb.shape)   # Expected shape: (64,)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_data_loader()
