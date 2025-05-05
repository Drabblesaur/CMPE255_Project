# use training.csv and test.csv

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import wandb
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label encoding
LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, attention_mask, label
        return input_ids, attention_mask


# Load data
df = pd.read_csv("Training.csv")
X = df["text"].astype(str).tolist()
y = df["bias_rating"].map(LABEL2ID).tolist()

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Load test data
df_test = pd.read_csv("Test.csv")
X_test = df_test["text"].astype(str).tolist()

# Model and tokenizer
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="microsoft/deberta-v3-large",
    help="Huggingface model name",
)
args = parser.parse_args()
MODEL_NAME = args.model_name

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_length = min(tokenizer.model_max_length, 1024)
print(f"Using max_length={max_length} for tokenization.")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(
    device
)

# Datasets and loaders
train_ds = NewsDataset(X_train, y_train, tokenizer, max_length=max_length)
val_ds = NewsDataset(X_val, y_val, tokenizer, max_length=max_length)
test_ds = NewsDataset(X_test, tokenizer=tokenizer, max_length=max_length)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)
test_loader = DataLoader(test_ds, batch_size=8)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Set device
wandb.init(
    project="news-bias-classification",
    config={
        "model": MODEL_NAME,
        "epochs": 5,
        "batch_size": 16,
        "max_length": max_length,
        "optimizer": "AdamW",
        "lr": 2e-5,
    },
    save_code=True,
)


# Training loop
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(model, loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            labels_all.extend(labels.cpu().numpy())
    return np.array(preds), np.array(labels_all)


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return np.array(preds)


# Main training
best_f1 = 0
patience = 3
wait = 0
for epoch in range(5):
    print(f"Epoch {epoch + 1}")
    train_loss = train_epoch(model, train_loader, optimizer)
    val_preds, val_labels = eval_model(model, val_loader)
    f1 = f1_score(val_labels, val_preds, average="macro")
    acc = accuracy_score(val_labels, val_preds)
    class_report = classification_report(
        val_labels, val_preds, target_names=LABEL2ID.keys(), output_dict=True
    )
    print(f"Train loss: {train_loss:.4f} | Val F1 (macro): {f1:.4f}")
    # Log metrics to wandb
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_f1_macro": f1,
            "val_accuracy": acc,
            **{
                f"val_f1_{k}": v["f1-score"]
                for k, v in class_report.items()
                if k in LABEL2ID.keys()
            },
            **{
                f"val_precision_{k}": v["precision"]
                for k, v in class_report.items()
                if k in LABEL2ID.keys()
            },
            **{
                f"val_recall_{k}": v["recall"]
                for k, v in class_report.items()
                if k in LABEL2ID.keys()
            },
        }
    )
    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_report = class_report
        wait = 0
        torch.save(model.state_dict(), "best_model.pt")
        # Log best model weights as artifact
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file("best_model.pt")
        wandb.log_artifact(artifact)
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))

# Final validation report
val_preds, val_labels = eval_model(model, val_loader)
final_acc = accuracy_score(val_labels, val_preds)
final_f1 = f1_score(val_labels, val_preds, average="macro")
final_report = classification_report(
    val_labels, val_preds, target_names=LABEL2ID.keys(), output_dict=True
)
print("Validation Accuracy:", final_acc)
print("Validation F1 (macro):", final_f1)
print(classification_report(val_labels, val_preds, target_names=LABEL2ID.keys()))

# Log best metrics to wandb
wandb.summary["best_val_f1_macro"] = best_f1
wandb.summary["best_val_accuracy"] = best_acc
for k in LABEL2ID.keys():
    wandb.summary[f"best_val_f1_{k}"] = best_report[k]["f1-score"]
    wandb.summary[f"best_val_precision_{k}"] = best_report[k]["precision"]
    wandb.summary[f"best_val_recall_{k}"] = best_report[k]["recall"]

# Predict on test set
test_preds = predict(model, test_loader)
test_labels = [ID2LABEL[i] for i in test_preds]
pd.Series(test_labels).to_csv("predictions.csv", index=False, header=False)
print("Test predictions saved to predictions.csv")

# If test set has labels, you can evaluate. If not, skip this block.
if "bias_rating" in df_test.columns:
    y_test = df_test["bias_rating"].map(LABEL2ID).tolist()
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    test_report = classification_report(
        y_test, test_preds, target_names=LABEL2ID.keys(), output_dict=True
    )
    print("Test Accuracy:", test_acc)
    print("Test F1 (macro):", test_f1)
    print(classification_report(y_test, test_preds, target_names=LABEL2ID.keys()))
    # Log test metrics to wandb
    wandb.summary["test_accuracy"] = test_acc
    wandb.summary["test_f1_macro"] = test_f1
    for k in LABEL2ID.keys():
        wandb.summary[f"test_f1_{k}"] = test_report[k]["f1-score"]
        wandb.summary[f"test_precision_{k}"] = test_report[k]["precision"]
        wandb.summary[f"test_recall_{k}"] = test_report[k]["recall"]

# Log predictions and code as artifacts
pred_artifact = wandb.Artifact("test_predictions", type="predictions")
pred_artifact.add_file("predictions.csv")
wandb.log_artifact(pred_artifact)

code_artifact = wandb.Artifact("source_code", type="code")
code_artifact.add_file("train.py")
wandb.log_artifact(code_artifact)

wandb.finish()
