import os

import torch
from torch.utils.data import TensorDataset

import numpy as np
import matplotlib.pyplot as plt

# Randomly shuffle dataset
def RandomPermute(X, y):
    assert X.shape[0] == y.shape[0]
    permute = np.random.permutation(X.shape[0])
    return X[permute], y[permute]

"""
Clips the values by frequency, really useful for balancing datasets.
Essentially creates a random distribution through the set, meaning removed
values don't accumulate in chunks.
"""
def ClipByFrequency(X, y, bins=301, max_per_bin=1800):
    print(f"Using {bins} bins with a max of {max_per_bin}")
    bin_edges = np.histogram_bin_edges(y, bins=bins)
    bin_ids = np.digitize(y, bin_edges) - 1
    keep_indices = []

    for b in range(bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) > max_per_bin:
            idx = np.random.choice(idx, max_per_bin, replace=False)
        keep_indices.append(idx)

    keep_indices = np.concatenate(keep_indices)
    return X[keep_indices], y[keep_indices]

def ReturnDataset(PATH, split, shuffle=False, balance=(51, 1800), figure=None):
    # Force the use of pickle on every load
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    BINS = balance[0]
    MAX_PER_BIN = balance[1]

    X = None
    y = None

    # Load all values from the NPY files
    X_center = np.load(os.path.abspath(f"{PATH}\\center.npy"))[::3]
    X_left   = np.load(os.path.abspath(f"{PATH}\\left.npy"))[::3]
    X_right  = np.load(os.path.abspath(f"{PATH}\\right.npy"))[::3]
    y        = np.load(os.path.abspath(f"{PATH}\\steer.npy"))[::6]

    #Plot figure
    if figure is not None:
        plt.subplot(1, 2, 1)
        plt.hist(y, bins=BINS, density=True)
        plt.title("Unbalanced Dataset")
        plt.ylabel("Normalised Occurrences")
        plt.xlabel("Steering Range Value")

    # Join all loaded frames together. Adding constants to steering values to accomodate camera view
    X = np.concatenate([X_center, X_left, X_right], axis=0, dtype=np.float32)
    y = np.clip(np.concatenate([y, y+0.35, y-0.35], axis=0, dtype=np.float32), a_max=1, a_min=-1.0)
    print(f"Dataset Size: {X.shape[0]}")
    if balance is not None:
        X, y = ClipByFrequency(X, y, bins=BINS, max_per_bin=MAX_PER_BIN)
    del X_center, X_left, X_right

    # Plot after clipping
    if figure is not None:
        plt.subplot(1, 2, 2)
        plt.hist(y, bins=BINS, density=True)
        plt.title("Balanced Dataset")
        plt.ylabel("Normalised Occurrences")
        plt.xlabel("Steering Range Value")
        #plt.show()
        plt.savefig(os.path.abspath(f"media/{figure}.png"))

    # Shuffle creates random mix for train/test selection
    if shuffle:
        X, y = RandomPermute(X, y)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Returns as TensorDataset
    ds_len = X.shape[0]
    return (
        TensorDataset(X[:int(ds_len * split)], y[:int(ds_len * split)]), # Training set
        TensorDataset(X[int(ds_len * split):], y[int(ds_len * split):])  # Validation/Test set
    )

if __name__ == "__main__":
    print("Running")
    _, _ = ReturnDataset("npysets", 0.95, figure="Plots")

