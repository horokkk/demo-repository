import librosa
import numpy as np



def extract_features(file_name: str, n_mfcc: int, max_len: int):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=4)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]

        return mfccs

    except Exception as e:
        print("Error occured in file: ", file_name)
        print("Error message:", e)
        return None

