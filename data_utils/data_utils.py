import random, os
import numpy as np
import torch
from torch.utils.data import TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_for_data_loader(df, genre_cols):
    # Features and labels
    user_indices = df['user_idx'].values
    movie_indices = df['movie_idx'].values
    occ_indices = df['occupation'].values
    age_indices = df['age_idx'].values
    gender_indices = df['gender_idx'].values
    genre_matrix = df[genre_cols].values.astype(np.float32)
    labels = df['rating'].values.astype(np.float32)

    # Train-test split
    SEED = 42
    set_seed(SEED)

    num_samples = len(df)
    perm = np.random.permutation(num_samples)
    train_size = int(0.8 * num_samples)
    train_idx, test_idx = perm[:train_size], perm[train_size:]

    def create_tensors(idx):
        return (
            torch.tensor(user_indices[idx], dtype=torch.long),
            torch.tensor(movie_indices[idx], dtype=torch.long),
            torch.tensor(occ_indices[idx], dtype=torch.long),
            torch.tensor(age_indices[idx], dtype=torch.long),
            torch.tensor(gender_indices[idx], dtype=torch.long),
            torch.tensor(genre_matrix[idx], dtype=torch.float32),
            torch.tensor(labels[idx], dtype=torch.float32),
        )

    train_tensors = create_tensors(train_idx)
    test_tensors = create_tensors(test_idx)

    return TensorDataset(*train_tensors), TensorDataset(*test_tensors)
