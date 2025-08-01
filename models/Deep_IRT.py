# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import os
from tqdm import tqdm

# -------------------- 模型定义 --------------------


class Deep_IRT(nn.Module):
    def __init__(self, num_students, num_items, hidden_dim=64):
        super().__init__()
        # 学生能力网络
        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim), nn.ReLU(
            ), nn.Linear(hidden_dim, 1)
        )
        # 题目参数网络
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim), nn.ReLU(
            ), nn.Linear(hidden_dim, 1)
        )
        # 全连接层，将能力值与难度值的差输入，输出两个 logit（做对/做不对）
        self.fc = nn.Linear(1, 2)  # 输入是 theta - beta 的差，输出 2 个类别

    def forward(self, student, item):
        theta = self.student_net(student)  # 学生能力值
        beta = self.item_net(item)      # 题目难度值

        # 计算能力值与难度值的差
        diff = theta - beta  # shape: [batch_size, 1]

        # 通过全连接层输出两个 logit
        logits = self.fc(diff)  # shape: [batch_size, 2]

        # 通过 softmax 转化为概率
        probs = torch.softmax(logits, dim=1)  # shape: [batch_size, 2]

        return probs, logits  # 返回概率和 logit（训练用 logit，评估用 probs）


# -------------------- 数据集类 --------------------


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
            "label": torch.tensor(label, dtype=torch.long),  # 用于交叉熵
        }

# -------------------- 评估函数 (已修改) --------------------


def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            y_true = batch["label"]  # Keep labels on CPU

            # We only need probabilities for evaluation
            y_pred_probs, _ = model(student, item)

            # Get the probability of the positive class (class 1)
            positive_probs = y_pred_probs[:, 1].cpu().numpy()
            y_pred_binary = (positive_probs > 0.5).astype(int)

            all_probs.extend(positive_probs)
            all_preds.extend(y_pred_binary)
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

# -------------------- 主程序 --------------------


def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    try:
        train_df = pd.read_csv("./data/train.csv", header=None)
        test_df = pd.read_csv("./data/test.csv", header=None)
    except FileNotFoundError as e:
        print(
            f"错误: 找不到数据文件 {e.filename}。请确保 train.csv 和 test.csv 在 data/ 目录中。")
        return

    # 创建数据集和验证集
    full_train_dataset = IRTDataset(train_df)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size])
    test_dataset = IRTDataset(test_df)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 模型初始化
    model = Deep_IRT(
        num_students=len(train_df), num_items=len(train_df.columns), hidden_dim=64
    ).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    SAVE_DIR = "models"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_model_deep_irt.pth")
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(
            f"No pre-trained model found at '{MODEL_SAVE_PATH}'. Starting training...")

        best_f1 = 0.0
        patience = 4
        epochs_no_improve = 0

        for epoch in range(50):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False):
                optimizer.zero_grad()
                student = batch["student"].to(device)
                item = batch["item"].to(device)
                labels = batch["label"].to(device)

                _, logits = model(student, item)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_metrics, _, _, _ = evaluate(model, val_loader, device)
            print(
                f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved with Val F1: {best_f1:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break
    else:
        print(
            f"Found existing model at '{MODEL_SAVE_PATH}'. Skipping training.")

    print(f"\nLoading best model from '{MODEL_SAVE_PATH}' for final evaluation.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("\nLoaded best model for final evaluation.")

    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, device)

    print("\nFinal Test Performance for Deep-IRT:")
    print(
        f"ACC: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")

    # --- 保存测试集预测结果 ---
    predictions_df = pd.DataFrame({
        'ground_truth': test_labels,
        'prediction': test_preds,
        'probability': test_probs
    })
    predictions_df.to_csv("results/test_predictions_deep_irt.csv", index=False)
    print("Test predictions for Deep-IRT saved to test_predictions_deep_irt.csv")
    # -------------------- 参数保存 --------------------
    student_ability = []
    with torch.no_grad():
        for sid in range(len(train_df)):
            student = torch.zeros(len(train_df)).float()
            student[sid] = 1.0
            ability = model.student_net(student.to(device)).cpu().item()
            student_ability.append(ability)
    pd.DataFrame(
        {"student_id": range(len(train_df)), "ability": student_ability}
    ).to_csv("student_abilities_deep_irt.csv", index=False)

    item_params = []
    with torch.no_grad():
        for qid in range(len(train_df.columns)):
            item = torch.zeros(len(train_df.columns)).float()
            item[qid] = 1.0
            params = model.item_net(item.to(device)).cpu().numpy()
            item_params.append({"item_id": qid, "difficulty": params[0]})
    pd.DataFrame(item_params).to_csv("item_parameters.csv", index=False)

    print("Results saved to student_abilities_deep_irt.csv and item_parameters.csv.")


if __name__ == "__main__":
    main()
