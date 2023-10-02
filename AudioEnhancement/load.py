import os
import numpy as np


def load_data_set(spectrograms_path):
    x_train = []
    sorted_file_names = []

    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            if file_name == '.DS_Store' or file_name.startswith("._") or not file_name.endswith('.npy'):
                continue
            sorted_file_names.append(file_name)

    sorted_file_names = sorted(sorted_file_names)

    for root, _, _ in os.walk(spectrograms_path):
        for file_name in sorted_file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path, allow_pickle=True)  # (n_bins, n_frames, 1)
            x_train.append(spectrogram)

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train