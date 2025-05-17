# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import argparse
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(3407)

# 4PL IRT Model with Embedding Output


class IRT(nn.Module):
    def __init__(self, num_students, num_items, hidden_dim=64, embedding_dim=128):
        super().__init__()
        self.num_students = num_students
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Student ability network with embedding output
        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.student_ability_out = nn.Linear(embedding_dim, 1)

        # Item parameter network with embedding output
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_param_out = nn.Linear(embedding_dim, 4)

    def forward(self, student, item):
        student_embedding = self.student_net(student)
        item_embedding = self.item_net(item)

        theta = self.student_ability_out(student_embedding)
        params = self.item_param_out(item_embedding)

        # Parameter parsing
        beta = params[:, 0].unsqueeze(1)
        a = params[:, 1].unsqueeze(1)
        # Guessing parameter [0, 1]
        c = torch.sigmoid(params[:, 2].unsqueeze(1))
        d = torch.sigmoid(params[:, 3].unsqueeze(1))

        logits = a * (theta - beta)
        prob = c + (d - c) * torch.sigmoid(logits)

        return prob, theta, params, student_embedding, item_embedding

# Dataset class


class IRTDataset(Dataset):
    def __init__(self, df):
        self.num_students = len(df)
        self.num_items = len(df.columns)
        self.data = []
        for sid in range(self.num_students):
            for qid in range(self.num_items):
                label = df.iloc[sid, qid]
                if not pd.isnull(label):
                    self.data.append((sid, qid, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid, qid, label = self.data[idx]
        student_vec = torch.zeros(self.num_students).float()
        student_vec[sid] = 1.0
        item_vec = torch.zeros(self.num_items).float()
        item_vec[qid] = 1.0
        return {
            "student": student_vec,
            "item": item_vec,
            "label": torch.tensor(label, dtype=torch.float32),
        }

# Main program


def main(learning_rate=0.003, weight_decay=0.0001, batch_size=512, max_epochs=30, hidden_dim=64, embedding_dim=128):
    print(f"Using 4PL model with embeddings on device: {device}")

    # Load data
    train_df = pd.read_csv("./combine.csv", header=None)

    num_students = len(train_df)
    num_items = len(train_df.columns)

    # Create dataset and data loader
    train_dataset = IRTDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    model = IRT(
        num_students=num_students,
        num_items=num_items,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim
    ).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    print("Start training...")
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", unit="batch"):
            optimizer.zero_grad()
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            labels = batch["label"].to(device)
            prob, _, _, _, _ = model(student, item)
            loss = criterion(prob.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.4f}")

    # Save parameters and embeddings
    save_params(model, train_df, embedding_dim)

# Parameter saving function


def save_params(model, train_df, embedding_dim):
    num_students = len(train_df)
    num_items = len(train_df.columns)

    # Get student abilities and embeddings
    student_abilities = torch.zeros(num_students, 1)
    student_embeddings = torch.zeros(num_students, embedding_dim)
    with torch.no_grad():
        for sid in range(num_students):
            student_vec = torch.zeros(
                num_students).float().unsqueeze(0).to(device)
            student_vec[0, sid] = 1.0
            embedding = model.student_net(student_vec)
            ability = model.student_ability_out(embedding)
            student_abilities[sid] = ability.cpu()
            student_embeddings[sid] = embedding.squeeze().cpu()
    pd.DataFrame({
        "student_id": range(num_students),
        "ability": student_abilities.squeeze().numpy()
    }).to_csv("student_abilities.csv", index=False)
    pd.DataFrame(student_embeddings.numpy()).to_csv(
        "student_embeddings.csv", index=False)
    print("Student parameters and embeddings saved.")

    # Get item parameters and embeddings
    item_params = torch.zeros(num_items, 4)
    item_embeddings = torch.zeros(num_items, embedding_dim)
    with torch.no_grad():
        for qid in range(num_items):
            item_vec = torch.zeros(num_items).float().unsqueeze(0).to(device)
            item_vec[0, qid] = 1.0
            embedding = model.item_net(item_vec)
            params = model.item_param_out(embedding)
            item_params[qid] = params.cpu()
            item_embeddings[qid] = embedding.squeeze().cpu()
    item_param_df = pd.DataFrame(item_params.numpy(), columns=[
        "difficulty", "discrimination", "guessing", "feasibility"])
    item_param_df["guessing"] = 1 / \
        (1 + np.exp(-item_param_df["guessing"]))
    item_param_df["feasibility"] = 1 / \
        (1 + np.exp(-item_param_df["feasibility"]))
    item_param_df.to_csv("item_parameters.csv", index=False)
    pd.DataFrame(item_embeddings.numpy()).to_csv(
        "item_embeddings.csv", index=False)
    print("Item parameters and embeddings saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float,
                        default=0.003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help='Weight decay')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum number of epochs')
    parser.add_argument('--hidden_dim', type=int,
                        default=64, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension for students and items')
    args = parser.parse_args()
    main(learning_rate=args.learning_rate, weight_decay=args.weight_decay,
         batch_size=args.batch_size, max_epochs=args.max_epochs,
         hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim)
