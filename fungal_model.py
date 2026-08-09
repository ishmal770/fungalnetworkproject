import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

CLASSES = ["none", "vegetative_stress", "fungal_fruiting_body"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ============================================================
# MODEL
# ============================================================
class FungalNet(nn.Module):
    def __init__(self, num_classes=len(CLASSES), dropout_p=0.3):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")

        # freeze everything except the last conv block — fast to train, still adapts
        # to your specific images instead of relying purely on generic ImageNet features
        for name, param in backbone.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop original fc layer
        feat_dim = backbone.fc.in_features  # 512 for resnet18

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # kept active at inference for MC Dropout uncertainty
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feats = self.features(x).flatten(1)
        class_logits = self.classifier_head(feats)
        network_score = self.regression_head(feats).squeeze(-1)
        return class_logits, network_score


# ============================================================
# DATASET
# ============================================================
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # standard ImageNet stats
])

class FungalDataset(Dataset):
    def __init__(self, csv_path, transform=IMG_TRANSFORM):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        class_idx = CLASS_TO_IDX[row["class"]]
        score = float(row["network_score"])
        return img, class_idx, score


# ============================================================
# TRAINING
# ============================================================
def train_model(train_csv="datasets/train.csv", epochs=8, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = FungalDataset(train_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FungalNet().to(device)
    class_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, class_idx, scores in loader:
            imgs, class_idx, scores = imgs.to(device), class_idx.to(device), scores.to(device).float()

            optimizer.zero_grad()
            class_logits, pred_scores = model(imgs)

            loss = class_loss_fn(class_logits, class_idx) + reg_loss_fn(pred_scores, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "fungal_model.pth")
    print("Saved fungal_model.pth")
    return model


def evaluate_model(model, test_csv="datasets/test.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FungalDataset(test_csv)
    loader = DataLoader(dataset, batch_size=16)

    model.eval()
    correct, total, mae_sum = 0, 0, 0
    with torch.no_grad():
        for imgs, class_idx, scores in loader:
            imgs, class_idx, scores = imgs.to(device), class_idx.to(device), scores.to(device).float()
            class_logits, pred_scores = model(imgs)
            preds = class_logits.argmax(dim=1)
            correct += (preds == class_idx).sum().item()
            total += len(class_idx)
            mae_sum += (pred_scores - scores).abs().sum().item()

    print(f"Test classification accuracy: {correct/total:.2%}")
    print(f"Test regression MAE: {mae_sum/total:.4f}")


# ============================================================
# INFERENCE — plugs directly into app.py's mc_dropout_predict
# ============================================================
def load_model(checkpoint_path="fungal_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FungalNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return IMG_TRANSFORM(img).unsqueeze(0)  # add batch dimension


def mc_dropout_predict(model, image_tensor, n_passes=25):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    model.train()  # keeps dropout layers active during inference — this IS the technique
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            _, score = model(image_tensor)
            preds.append(score.item())

    return {"mean": float(np.mean(preds)), "std": float(np.std(preds))}


if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)