import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from data_utils.data_utils import prepare_for_data_loader
from tqdm import tqdm

from models.matrix_factorization import BiasedMF


def additional_preprocess():
    # Load the dataset
    df = pd.read_csv("dataset.csv")
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # One-hot encoded to index
    df['age_idx'] = df[['age_group_child', 'age_group_teen', 'age_group_adult', 'age_group_senior']].dot(np.array([0, 1, 2, 3]))
    df['gender_idx'] = df[['gender_F', 'gender_M']].dot(np.array([0, 1]))
    df['user_idx'] = df['uid'].astype("category").cat.codes
    df['movie_idx'] = df['movie_id'].astype("category").cat.codes

    # Genre columns
    genre_cols = [col for col in df.columns if col not in ['uid', 'movie_id', 'rating', 'occupation', 'age_idx', 'gender_idx', 'user_idx', 'movie_idx']
                and not col.startswith('age_group') and not col.startswith('gender_')]
    
    return df, genre_cols

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(train_loader):
        pred = model(*batch[:-1])
        loss = criterion(pred, batch[-1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        class_pred = (torch.sigmoid(pred) >= 0.5).float()
        correct += (class_pred == batch[-1]).sum().item()
        total += len(batch[-1])
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            pred = model(*batch[:-1])
            loss = criterion(pred, batch[-1])
            total_loss += loss.item()
            
            probs = torch.sigmoid(pred)
            class_pred = (probs >= 0.5).float()
            correct += (class_pred == batch[-1]).sum().item()
            total += len(batch[-1])
            all_logits.append(probs.numpy())
            all_labels.append(batch[-1].numpy())
    
    val_loss = total_loss / len(val_loader)
    val_accuracy = correct / total
    val_auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_logits))

    return val_loss, val_accuracy, val_auc


def main():
    # Model parameters
    BATCH_SIZE = 512
    LR = 0.01
    LATENT_DIM = 20
    EPOCHS = 5
    
    # Preprocess the data
    df, genre_cols = additional_preprocess()

    # Prepare data loaders
    train_dataset, val_dataset = prepare_for_data_loader(df, genre_cols)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    num_users = df['user_idx'].nunique()
    num_movies = df['movie_idx'].nunique()
    num_age = 4
    num_gender = 2
    num_occ = df['occupation'].nunique()
    num_genres = len(genre_cols)

    model = BiasedMF(num_users, num_movies, num_age, num_gender, num_occ, num_genres, LATENT_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    valid_auc_history = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc, valid_auc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}")
        
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)
        valid_auc_history.append(valid_auc)
        
if __name__ == "__main__":
    main()
   








