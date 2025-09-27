import os
import librosa
import numpy as np
from pathlib import Path
from constants import DATA_DIR, FEATURES_DIR  # import constants

class FSDD:
    """
    Free Spoken Digit Dataset (FSDD) loader + MFCC extractor.
    """

    def __init__(self, 
                 root_dir: Path = DATA_DIR, 
                 feature_dir: Path = FEATURES_DIR,
                 n_mfcc: int = 13, 
                 sr: int = 8000):

        self.root_dir = Path(root_dir)
        self.feature_dir = Path(feature_dir)
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.files = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self._load_files()
        os.makedirs(self.feature_dir, exist_ok=True)

    def _load_files(self):
        for fname in os.listdir(self.root_dir):
            if not fname.endswith(".wav"):
                continue
            # filename format: {digit}_{speaker}_{index}.wav
            label = fname.split("_")[0]
            path = self.root_dir / fname
            self.files.append(path)
            self.labels.append(label)

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

    def extract_features(self, file_path):
        """
        Load audio, compute MFCC, return features (frames x n_mfcc).
        Saves features as .npy for later reuse.
        """
        fname = Path(file_path).with_suffix(".npy").name
        fpath = self.feature_dir / fname

        if fpath.exists():  # load cached features
            return np.load(fpath)

        y, sr = librosa.load(file_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=256, hop_length=80
        )
        mfcc = mfcc.T  # shape: (frames, n_mfcc)

        np.save(fpath, mfcc)  # save for next time
        return mfcc

    def prepare_data(self):
        X = [self.extract_features(f) for f in self.files]
        y = [self.label_to_idx[l] for l in self.labels]
        return X, y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.extract_features(self.files[idx]), self.label_to_idx[self.labels[idx]]