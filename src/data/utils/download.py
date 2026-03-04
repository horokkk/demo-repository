import kagglehub
import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

def download_dataset(SAVE_PATH: str):

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Folder created at: {SAVE_PATH}")
    else:
        print(f"Save path: {SAVE_PATH}")
        
            # 1. Download dataset (from Kaggle server to local server)
    print("\nDownloading dataset...")
    path = kagglehub.dataset_download("chrisfilo/urbansound8k")
    print(f"Download complete! Path: {path}")
    print("Contents in folder:", os.listdir(path))   # for checking
    
    return path

