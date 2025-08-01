# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import random
import argparse
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Configuration --------------------


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(114514)

# -------------------- Graph Data Builder --------------------


class CognitiveGraph:
    def __init__(self, dataset):
        self.num_students = dataset.num_students
        self.num_items = dataset.num_items
        self.edge_index, self.responses = self._build_edges(dataset)

    def _build_edges(self, dataset):
        s_indices, e_indices, responses = [], [], []
        for s, e, r in dataset.interactions:
            s_indices.append(s)
            e_indices.append(e)
            responses.append(r)

        edge_index = torch.tensor([s_indices, e_indices], dtype=torch.long)
        return edge_index.to(device), torch.tensor(responses, dtype=torch.long).to(device)

# -------------------- GNN Encoder --------------------


class GNNEncoder(nn.Module):
    def __init__(self, num_students, num_items, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim

        self.student_emb = nn.Embedding(num_students, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # Cognitive Degree Networks from formula ls,q = Wl,rel * [eq, Wrel * rel] + bl,rel
        self.W_rel = nn.Linear(1, emb_dim, bias=False)
        self.learning_gate = nn.Linear(emb_dim * 2, emb_dim)

        # Attention Networks from formula as,qi = Wa * [Wn * es, Wn * cs,qi]
        self.Wn_s = nn.Linear(emb_dim, emb_dim)  # Wn for student embedding
        # Wn for cognitive state (student update)
        self.Wn_c_s = nn.Linear(emb_dim, emb_dim)
        self.Wa_s = nn.Linear(emb_dim * 2, 1)  # Wa for student update

        self.Wn_i = nn.Linear(emb_dim, emb_dim)  # Wn for item embedding
        # Wn for cognitive state (item update)
        self.Wn_c_i = nn.Linear(emb_dim, emb_dim)
        self.Wa_i = nn.Linear(emb_dim * 2, 1)  # Wa for item update

    def forward(self, edge_index, responses):
        es_initial = self.student_emb.weight
        eq_initial = self.item_emb.weight

        s_idx, e_idx = edge_index

        # --- Student Embedding Update ---
        rel_scalar = responses.float().unsqueeze(-1)
        W_rel_times_rel = self.W_rel(rel_scalar)
        eq_on_edges = eq_initial[e_idx]

        ls_q_input = torch.cat([eq_on_edges, W_rel_times_rel], dim=1)
        ls_q = self.learning_gate(ls_q_input)
        cs_q = eq_on_edges * ls_q

        es_on_edges = es_initial[s_idx]

        # Attention calculation according to as,qi = Wa * [Wn * es, Wn * cs,qi]
        Wn_es = self.Wn_s(es_on_edges)
        Wn_csq = self.Wn_c_s(cs_q)
        attn_s_input = torch.cat([Wn_es, Wn_csq], dim=1)
        attn_s_scores = F.leaky_relu(self.Wa_s(attn_s_input))

        # Manual softmax over neighbors using index_add_
        attn_s_scores_exp = torch.exp(attn_s_scores)
        s_neighbor_exp_sum = torch.zeros(es_initial.shape[0], 1, device=device)
        s_neighbor_exp_sum.index_add_(0, s_idx, attn_s_scores_exp)
        # Avoid division by zero for isolated nodes
        s_neighbor_exp_sum[s_neighbor_exp_sum == 0] = 1
        alpha_s = attn_s_scores_exp / s_neighbor_exp_sum[s_idx]

        agg_items = torch.zeros_like(es_initial)
        agg_items.index_add_(0, s_idx, alpha_s * cs_q)
        es_updated = es_initial + agg_items

        # --- Item Embedding Update (Symmetric) ---
        ls_s_input = torch.cat([es_on_edges, W_rel_times_rel], dim=1)
        ls_s = self.learning_gate(ls_s_input)
        cs_s = es_on_edges * ls_s

        Wn_eq = self.Wn_i(eq_on_edges)
        Wn_css = self.Wn_c_i(cs_s)
        attn_i_input = torch.cat([Wn_eq, Wn_css], dim=1)
        attn_i_scores = F.leaky_relu(self.Wa_i(attn_i_input))

        attn_i_scores_exp = torch.exp(attn_i_scores)
        i_neighbor_exp_sum = torch.zeros(eq_initial.shape[0], 1, device=device)
        i_neighbor_exp_sum.index_add_(0, e_idx, attn_i_scores_exp)
        i_neighbor_exp_sum[i_neighbor_exp_sum == 0] = 1
        alpha_i = attn_i_scores_exp / i_neighbor_exp_sum[e_idx]

        agg_students = torch.zeros_like(eq_initial)
        agg_students.index_add_(0, e_idx, alpha_i * cs_s)
        eq_updated = eq_initial + agg_students

        return es_updated, eq_updated

# -------------------- IRT Model --------------------


class IRTModel(nn.Module):
    def __init__(self, encoder, model_type='3pl'):
        super().__init__()
        self.encoder = encoder
        self.model_type = model_type.lower()
        self.theta_layer = nn.Linear(encoder.emb_dim, 1)

        param_dims = {'1pl': 1, '2pl': 2, '3pl': 3, '4pl': 4}
        self.item_param_head = nn.Linear(
            encoder.emb_dim, param_dims[self.model_type])

    def forward(self, s_idx, e_idx, graph_edges, graph_responses):
        es_updated, eq_updated = self.encoder(graph_edges, graph_responses)
        es_batch = es_updated[s_idx]
        eq_batch = eq_updated[e_idx]
        theta = self.theta_layer(es_batch)
        params = self.item_param_head(eq_batch)
        b = params[:, 0].unsqueeze(
            1) if params.shape[1] >= 1 else torch.zeros_like(theta)
        a = F.softplus(params[:, 1].unsqueeze(1)) if self.model_type in [
            '2pl', '3pl', '4pl'] else torch.ones_like(theta)
        c = torch.sigmoid(params[:, 2].unsqueeze(1)) if self.model_type in [
            '3pl', '4pl'] else torch.zeros_like(theta)
        d = c + (1 - c) * torch.sigmoid(params[:, 3].unsqueeze(
            1)) if self.model_type == '4pl' else torch.ones_like(theta)
        logits = a * (theta - b)
        prob = c + (d - c) * torch.sigmoid(logits)
        return prob.clamp(1e-6, 1-1e-6).squeeze(-1)

    def get_all_params(self, graph_edges, graph_responses):
        with torch.no_grad():
            es_updated, eq_updated = self.encoder(graph_edges, graph_responses)
            theta = self.theta_layer(es_updated).cpu().numpy().flatten()
            params = self.item_param_head(eq_updated).cpu()
            item_params = {'difficulty': params[:, 0].numpy(
            ) if params.shape[1] >= 1 else np.nan}
            if self.model_type in ['2pl', '3pl', '4pl']:
                item_params['discrimination'] = F.softplus(
                    params[:, 1]).numpy()
            if self.model_type in ['3pl', '4pl']:
                item_params['guessing-rate'] = torch.sigmoid(
                    params[:, 2]).numpy()
            if self.model_type == '4pl':
                c = torch.sigmoid(params[:, 2])
                d_raw = torch.sigmoid(params[:, 3])
                item_params['feasibility'] = (c + (1-c)*d_raw).numpy()
        return theta, item_params

# -------------------- Dataset --------------------


class CognitiveDataset(Dataset):
    def __init__(self, df):
        self.num_students = df.shape[0]
        self.num_items = df.shape[1]
        self.interactions = []
        df.columns = range(self.num_items)
        for s in range(self.num_students):
            for e in range(self.num_items):
                score = df.iloc[s, e]
                if pd.notnull(score):
                    self.interactions.append((s, e, int(score)))

    def __len__(self): return len(self.interactions)

    def __getitem__(self, idx):
        s, e, score = self.interactions[idx]
        return {'student_idx': s, 'item_idx': e, 'label': torch.tensor(score, dtype=torch.float32)}

# -------------------- Trainer --------------------


class Trainer:
    def __init__(self, model, graph, train_loader, val_loader):
        self.model = model.to(device)
        self.graph = graph
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', patience=5, factor=0.1, verbose=True)
        self.criterion = nn.BCELoss()
        self.best_f1 = 0.0
        self.patience = 10
        self.epochs_no_improve = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            s = batch['student_idx'].to(device)
            e = batch['item_idx'].to(device)
            y = batch['label'].to(device)
            pred = self.model(s, e, self.graph.edge_index,
                              self.graph.responses)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                s = batch['student_idx'].to(device)
                e = batch['item_idx'].to(device)
                y = batch['label'].cpu().numpy()
                p = self.model(s, e, self.graph.edge_index,
                               self.graph.responses).cpu().numpy()
                if hasattr(p, '__iter__'):
                    preds.extend(p)
                else:
                    preds.append(p)
                labels.extend(y)
        preds_bin = [1 if x > 0.5 else 0 for x in preds]
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5
        return {'f1': f1_score(labels, preds_bin), 'acc': accuracy_score(labels, preds_bin), 'auc': auc}

    def run(self, max_epochs=50):
        for epoch in range(max_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            print(
                f"\nEpoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['acc']:.4f} | AUC: {val_metrics['auc']:.4f}")
            self.scheduler.step(val_metrics['f1'])
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(
                    f"Best model saved with validation F1: {self.best_f1:.4f}")
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

# -------------------- Main Execution --------------------


def main(model_type='3pl'):
    try:
        train_df = pd.read_csv("./data/train.csv", header=None)
        test_df = pd.read_csv("./data/test.csv", header=None)
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found!")
        return

    full_train_data = CognitiveDataset(train_df)
    test_data = CognitiveDataset(test_df)

    train_size = int(0.8 * len(full_train_data))
    train_set, val_set = random_split(
        full_train_data, [train_size, len(full_train_data)-train_size])

    graph = CognitiveGraph(full_train_data)

    encoder = GNNEncoder(
        num_students=full_train_data.num_students,
        num_items=full_train_data.num_items
    )
    model = IRTModel(encoder, model_type=model_type)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)
    test_loader = DataLoader(test_data, batch_size=512)

    trainer = Trainer(model, graph, train_loader, val_loader)
    trainer.run(max_epochs=50)

    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = trainer.evaluate(test_loader)
    print("\nFinal Test Results:")
    print(
        f"F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['acc']:.4f} | AUC: {test_metrics['auc']:.4f}")

    try:
        theta, item_params = model.get_all_params(
            graph.edge_index, graph.responses)
        pd.DataFrame({'student_id': range(len(theta)), 'ability': theta}).to_csv(
            f'results/student_abilities_gnn_{model_type}.csv', index=False)
        item_df = pd.DataFrame(item_params)
        item_df['item_id'] = item_df.index
        cols = ['item_id'] + \
            [col for col in item_df.columns if col != 'item_id']
        item_df = item_df[cols]
        item_df.to_csv(
            f'results/item_parameters_gnn_{model_type}.csv', index=False)
        print(
            f"\nParameters saved to 'results/student_abilities_gnn_{model_type}.csv' and 'results/item_parameters_gnn_{model_type}.csv'")
    except Exception as e:
        print(f"Error saving parameters: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='4pl',
                        choices=['1pl', '2pl', '3pl', '4pl'])
    args = parser.parse_args()
    main(args.model_type)
