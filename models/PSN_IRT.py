# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

# -------------------- Model Definition --------------------


class PSN_IRT(nn.Module):
    def __init__(self, num_students, num_items, hidden_dim=64, model_type='4PL'):
        super().__init__()
        self.model_type = model_type

        if model_type == '1PL':
            item_output_dim = 1
        elif model_type == '2PL':
            item_output_dim = 2
        elif model_type == '3PL':
            item_output_dim = 3
        elif model_type == '4PL':
            item_output_dim = 4
        else:
            raise ValueError(
                "model_type must be one of '1PL', '2PL', '3PL', or '4PL'")

        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_output_dim)
        )

    def forward(self, student, item):
        theta = self.student_net(student)
        params = self.item_net(item)

        b = torch.zeros_like(theta)
        a = torch.ones_like(theta)
        c = torch.zeros_like(theta)
        d = torch.ones_like(theta)

        if self.model_type == '1PL':
            b = params[:, 0].unsqueeze(1)
        elif self.model_type == '2PL':
            b = params[:, 0].unsqueeze(1)
            a = params[:, 1].unsqueeze(1)
        elif self.model_type == '3PL':
            b = params[:, 0].unsqueeze(1)
            a = params[:, 1].unsqueeze(1)
            c = torch.sigmoid(params[:, 2].unsqueeze(1))
        elif self.model_type == '4PL':
            b = params[:, 0].unsqueeze(1)
            a = params[:, 1].unsqueeze(1)
            c = torch.sigmoid(params[:, 2].unsqueeze(1))
            d = torch.sigmoid(params[:, 3].unsqueeze(1))

        exponent = a * (theta - b)
        prob = c + (d - c) * torch.sigmoid(exponent)
        return prob

# -------------------- Dataset Class --------------------


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
        student_vec = torch.zeros(self.num_students)
        student_vec[sid] = 1.0
        item_vec = torch.zeros(self.num_items)
        item_vec[qid] = 1.0
        return {
            "student": student_vec.float(),
            "item": item_vec.float(),
            "label": torch.tensor(label, dtype=torch.float),
        }

# -------------------- Evaluation Function --------------------


def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            y_true = batch["label"]  # Keep labels on CPU for easy appending

            y_pred_probs = model(student, item)

            # Handle different output shapes
            if y_pred_probs.dim() > 1:
                y_pred_probs = y_pred_probs.squeeze()
            if y_pred_probs.dim() == 0:
                y_pred_probs = y_pred_probs.unsqueeze(0)

            y_pred_binary = (y_pred_probs > 0.5).float()

            all_probs.extend(y_pred_probs.cpu().numpy())
            all_preds.extend(y_pred_binary.cpu().numpy())
            all_labels.extend(y_true.numpy())

    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.5

    metrics = {
        'acc': accuracy_score(all_labels, all_preds),
        'auc': auc_score,
        'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0)
    }

    # Return raw results for statistical testing
    return metrics, all_labels, all_preds, all_probs


# -------------------- Main Program --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    prediction_file_path = "results/test_predictions_psn_irt.csv"
    if os.path.exists(prediction_file_path):
        print(
            f"\nFound existing prediction file at '{prediction_file_path}'. Skipping training.")
        print("Calculating metrics from existing file...")

        # Load the predictions and calculate metrics
        predictions_df = pd.read_csv(prediction_file_path)
        test_labels = predictions_df['ground_truth']
        test_preds = predictions_df['prediction']
        test_probs = predictions_df['probability']

        try:
            auc = roc_auc_score(test_labels, test_probs)
        except ValueError:
            auc = 0.5

        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds,
                      average='binary', zero_division=0)

        print("\nFinal Test Performance (from pre-existing file):")
        print(f"ACC: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Exit the function early
        return
    try:
        train_df = pd.read_csv("./data/train.csv", header=None)
        test_df = pd.read_csv("./data/test.csv", header=None)
    except FileNotFoundError as e:
        print(
            f"Error: Data file not found {e.filename}. Make sure train.csv and test.csv are in the data/ directory.")
        return

    full_train_dataset = IRTDataset(train_df)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size])
    test_dataset = IRTDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    MODEL_TYPE_TO_TRAIN = '4PL'
    print(f"Training a {MODEL_TYPE_TO_TRAIN} model...")

    model = PSN_IRT(
        num_students=len(train_df),
        num_items=len(train_df.columns),
        hidden_dim=64,
        model_type=MODEL_TYPE_TO_TRAIN
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    criterion = nn.BCELoss()

    best_f1 = 0.0
    patience = 4
    epochs_no_improve = 0

    # Simplified training loop for clarity
    for epoch in range(50):  # Assuming a max of 50 epochs
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False):
            optimizer.zero_grad()
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            labels = batch["label"].to(device)
            probs = model(student, item)
            loss = criterion(probs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        print(
            f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val ACC: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_psn_irt.pth")
            print(f"New best model saved with Val F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    print("\nLoaded best model for final evaluation.")

    # --- Save predictions for statistical test ---
    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, device)

    print("\nFinal Test Performance:")
    print(
        f"ACC: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")

    # Save results to a file
    predictions_df = pd.DataFrame({
        'ground_truth': test_labels,
        'prediction': test_preds,
        'probability': test_probs
    })
    predictions_df.to_csv("results/test_predictions_psn_irt.csv", index=False)
    print("Test predictions saved to test_predictions_psn_irt.csv")
    # --- Parameter saving ---
    print("\nSaving estimated parameters...")
    model.eval()  # Ensure model is in evaluation mode
    student_ability_list = []
    with torch.no_grad():
        for sid in range(len(train_df)):
            student_vec = torch.zeros(len(train_df)).float()
            student_vec[sid] = 1.0
            # Ensure the input tensor is on the correct device
            ability = model.student_net(student_vec.to(device)).cpu().item()
            student_ability_list.append(ability)
    pd.DataFrame(
        {"student_id": range(len(train_df)), "ability": student_ability_list}
    ).to_csv("student_abilities.csv", index=False)

    item_params_list = []
    with torch.no_grad():
        for qid in range(len(train_df.columns)):
            item_vec = torch.zeros(len(train_df.columns)).float()
            item_vec[qid] = 1.0

            params_raw = model.item_net(item_vec.unsqueeze(0).to(device))

            item_data = {"item_id": qid}
            model_type = model.model_type

            if model_type == '1PL':
                item_data['difficulty'] = params_raw[0, 0].item()
            elif model_type == '2PL':
                item_data['difficulty'] = params_raw[0, 0].item()
                item_data['discriminability'] = params_raw[0, 1].item()
            elif model_type == '3PL':
                item_data['difficulty'] = params_raw[0, 0].item()
                item_data['discriminability'] = params_raw[0, 1].item()
                item_data['guessing'] = torch.sigmoid(params_raw[0, 2]).item()
            elif model_type == '4PL':
                item_data['difficulty'] = params_raw[0, 0].item()
                item_data['discriminability'] = params_raw[0, 1].item()
                c = torch.sigmoid(params_raw[0, 2]).item()
                d_raw = torch.sigmoid(params_raw[0, 3]).item()
                d = c + (1 - c) * d_raw
                item_data['guessing'] = c
                item_data['feasibility'] = d

            item_params_list.append(item_data)

    item_params_df = pd.DataFrame(item_params_list)

    all_param_cols = ['item_id', 'difficulty',
                      'discriminability', 'guessing', 'feasibility']
    for col in all_param_cols:
        if col not in item_params_df.columns:
            item_params_df[col] = np.nan
    item_params_df = item_params_df[all_param_cols]
    item_params_df.to_csv("item_parameters.csv", index=False)

    print("Results saved to student_abilities.csv and item_parameters.csv.")


if __name__ == "__main__":
    main()
