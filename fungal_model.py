import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

# "background" = ground, sky, gravel, turf, rock — a frame with no organism of
# interest in it. It is most of what the plot camera actually photographs, and
# without it softmax has to force those frames into an organism class.
CLASSES = ["background", "none", "vegetative_stress", "fungal_fruiting_body"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FUNGAL_IDX = CLASS_TO_IDX["fungal_fruiting_body"]


class FungalNet(nn.Module):
    def __init__(self, num_classes=len(CLASSES), dropout_p=0.5):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")

        
        for name, param in backbone.named_parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop original fc layer
        feat_dim = backbone.fc.in_features  # 512 for resnet18

        
        self._frozen_bn = [m for name, m in self.features.named_modules()
                           if isinstance(m, nn.BatchNorm2d)]
        for m in self._frozen_bn:
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

        
        self.trunk = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
        )
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_p),  # kept active at inference for MC Dropout uncertainty
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def train(self, mode=True):
        super().train(mode)
        # keep the frozen backbone's BatchNorm in inference mode no matter what
        for m in getattr(self, "_frozen_bn", []):
            m.eval()
        return self

    def forward(self, x):
        feats = self.features(x).flatten(1)
        shared = self.trunk(feats)
        class_logits = self.classifier_head(shared)
        network_score = self.regression_head(shared).squeeze(-1)
        return class_logits, network_score



NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard ImageNet stats

# eval/inference: deterministic, no augmentation
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),      # top-down soil shots have no canonical up
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
    transforms.ToTensor(),
    NORMALIZE,
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

class FungalDataset(Dataset):
    def __init__(self, csv_path, transform=IMG_TRANSFORM):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def labels(self):
        return self.df["class"].map(CLASS_TO_IDX).to_numpy()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        class_idx = CLASS_TO_IDX[row["class"]]
        score = float(row["network_score"])
        return img, class_idx, score



def train_model(train_csv="datasets/train.csv", epochs=16, batch_size=32, lr=3e-4,
                weight_decay=5e-3, val_split=0.2, patience=6, seed=42):
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    
    train_view = FungalDataset(train_csv, transform=TRAIN_TRANSFORM)
    val_view = FungalDataset(train_csv, transform=IMG_TRANSFORM)

    
    from sklearn.model_selection import train_test_split
    labels = train_view.labels()
    train_idx, val_idx = train_test_split(
        np.arange(len(train_view)), test_size=val_split, stratify=labels, random_state=seed)

    train_subset = Subset(train_view, train_idx)
    val_subset = Subset(val_view, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2)

    present = np.unique(labels)
    if len(present) < len(CLASSES):
        missing = [CLASSES[i] for i in range(len(CLASSES)) if i not in present]
        print(f"warn: no training examples for {missing} — those outputs are untrained")

    model = FungalNet().to(device)

    
    counts = np.bincount(labels[train_idx], minlength=len(CLASSES)).astype(float)
    weights = np.where(counts > 0, len(train_idx) / (len(present) * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    class_loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1)
    reg_loss_fn = nn.MSELoss()
    
    reg_weight = 0.3

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        total_train_loss, correct, total = 0, 0, 0
        for imgs, class_idx, scores in train_loader:
            imgs, class_idx, scores = imgs.to(device), class_idx.to(device), scores.to(device).float()
            optimizer.zero_grad()
            class_logits, pred_scores = model(imgs)
            loss = class_loss_fn(class_logits, class_idx) + reg_weight * reg_loss_fn(pred_scores, scores)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            correct += (class_logits.argmax(dim=1) == class_idx).sum().item()
            total += len(class_idx)
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(correct / total)

        
        model.eval()
        total_val_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, class_idx, scores in val_loader:
                imgs, class_idx, scores = imgs.to(device), class_idx.to(device), scores.to(device).float()
                class_logits, pred_scores = model(imgs)
                loss = class_loss_fn(class_logits, class_idx) + reg_weight * reg_loss_fn(pred_scores, scores)
                total_val_loss += loss.item()
                v_correct += (class_logits.argmax(dim=1) == class_idx).sum().item()
                v_total += len(class_idx)
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float("nan")
        val_losses.append(avg_val_loss)
        val_accs.append(v_correct / v_total if v_total > 0 else float("nan"))
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} — train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}, "
              f"train acc: {train_accs[-1]:.2%}, val acc: {val_accs[-1]:.2%}")


       
        if avg_val_loss < best_val - 1e-4:
            best_val, best_epoch, stale = avg_val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {epoch+1} — no val improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights from epoch {best_epoch+1} (val loss {best_val:.4f})")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(), "fungal_model.pth")
    print("Saved fungal_model.pth")
    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_losses, label="Train loss", marker="o")
    ax1.plot(epochs, val_losses, label="Validation loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (classification + regression combined)")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train accuracy", marker="o")
    ax2.plot(epochs, val_accs, label="Validation accuracy", marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Classification accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle("Training Progress — watch for val loss rising while train loss keeps dropping (overfitting)")
    fig.tight_layout()
    fig.savefig("training_curves.png")
    print("Saved training_curves.png")
    plt.show()


def evaluate_model(model, test_csv="datasets/test.csv"):
    device = next(model.parameters()).device
    dataset = FungalDataset(test_csv)
    loader = DataLoader(dataset, batch_size=32)  # shuffle=False, so row order is preserved

    model.eval()
    correct, total, mae_sum = 0, 0, 0
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    all_preds = []
    with torch.no_grad():
        for imgs, class_idx, scores in loader:
            imgs, class_idx, scores = imgs.to(device), class_idx.to(device), scores.to(device).float()
            class_logits, pred_scores = model(imgs)
            preds = class_logits.argmax(dim=1)
            correct += (preds == class_idx).sum().item()
            total += len(class_idx)
            mae_sum += (pred_scores - scores).abs().sum().item()
            all_preds.extend(preds.cpu().numpy().tolist())
            for t, p in zip(class_idx.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1

    print(f"Test classification accuracy: {correct/total:.2%}")
    print(f"Test regression MAE: {mae_sum/total:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_string())
    # per-class recall matters more than overall accuracy here — a model that
    # never predicts fungal_fruiting_body can still score well on accuracy alone
    for i, c in enumerate(CLASSES):
        if confusion[i].sum():
            print(f"  recall[{c}]: {confusion[i, i] / confusion[i].sum():.2%}")

    report_source_confound(dataset.df, all_preds)


def report_source_confound(df, preds):
    """The failure that produced sky-is-a-fungus was invisible in accuracy, because
    every class came from exactly one dataset — so 'recognise the dataset' and
    'recognise the subject' score identically on a held-out split drawn from the
    same datasets. This prints the class x source table so that degeneracy is
    visible, and reports accuracy per source: if one source is near-perfect while
    another collapses, the model is keying on capture style, not on content.
    """
    if "source" not in df.columns:
        return
    df = df.copy()
    df["pred"] = [CLASSES[p] for p in preds]
    df["correct"] = df["pred"] == df["class"]

    print("\nclass x source in the test set:")
    print(pd.crosstab(df["class"], df["source"]).to_string())

    single = [c for c, g in df.groupby("class") if g["source"].nunique() == 1]
    if single:
        print(f"warn: {single} each come from a single source — accuracy on these "
              f"cannot distinguish subject recognition from dataset recognition")

    print("\naccuracy by source:")
    for src, g in df.groupby("source"):
        print(f"  {src:14s} {g['correct'].mean():.2%}  (n={len(g)})")



def load_model(checkpoint_path="fungal_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FungalNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return IMG_TRANSFORM(img).unsqueeze(0)  # add batch dimension


def preprocess_tiles(image_path, grid=3, overlap=0.5):
    """Every training photo is a macro shot with the organism filling the frame.
    A plot photo covers a 5-ft square, where a fruiting body might be 2% of the
    pixels — squashing that whole frame to 224x224 shrinks the one thing we care
    about to a smudge. Scoring overlapping crops instead presents each region at
    roughly the scale the model was trained at. The full frame is included too,
    so large/diffuse growth is not missed by the tiling.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    step_x, step_y = w / grid, h / grid
    win_x, win_y = step_x * (1 + overlap), step_y * (1 + overlap)

    crops = [img]
    for r in range(grid):
        for c in range(grid):
            left = min(max(0, c * step_x - (win_x - step_x) / 2), max(0, w - win_x))
            top = min(max(0, r * step_y - (win_y - step_y) / 2), max(0, h - win_y))
            crops.append(img.crop((int(left), int(top),
                                   int(min(left + win_x, w)), int(min(top + win_y, h)))))
    return torch.stack([IMG_TRANSFORM(c) for c in crops])


def predict_image(model, image_path, grid=3, n_passes=25, min_confidence=0.65):
    """Tile-aware prediction. Risk for the tile is the *strongest* fungal evidence
    found anywhere in it — averaging would let 8 patches of bare dirt bury the one
    patch that actually has a mushroom in it.

    min_confidence raised 0.5 -> 0.65: this is the main lever against
    ambiguous frames (like sky) getting committed to a class instead of
    abstaining. Combined with app.py's rule-based looks_like_sky() pre-filter,
    this should catch the sky/fungi confusion from both the ML and non-ML side.
    """
    device = next(model.parameters()).device
    tiles = preprocess_tiles(image_path, grid=grid).to(device)

    
    probs, scores = _mc_forward(model, tiles, n_passes)
    per_tile = [_summarise(probs[:, i].mean(axis=0), probs[:, i, FUNGAL_IDX],
                           scores[:, i].mean(), min_confidence)
                for i in range(tiles.shape[0])]

    scored = [p for p in per_tile if not p["abstain"]]
    if not scored:
        # nothing in the frame the model is willing to call — sky, bare ground, blur
        return {**per_tile[0], "mean": 0.0, "abstain": True, "n_tiles_scored": 0}

    best = max(scored, key=lambda p: p["mean"])
    return {**best, "abstain": False, "n_tiles_scored": len(scored)}


def _mc_forward(model, batch, n_passes):
    """n_passes stochastic forward passes over a whole batch at once.

    Enables ONLY the dropout layers. model.train() would also flip BatchNorm into
    batch-statistics mode, which normalises each batch against itself — predictions
    come out wrong and the "uncertainty" ends up measuring BatchNorm noise rather
    than dropout. Returns (probs, scores) shaped (n_passes, B, C) and (n_passes, B).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    probs, scores = [], []
    with torch.no_grad():
        for _ in range(n_passes):
            logits, score = model(batch)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            scores.append(score.cpu().numpy().reshape(-1))

    model.eval()
    return np.stack(probs), np.stack(scores)


def _summarise(mean_probs, fungal_samples, score_mean, min_confidence):
    top_idx = int(mean_probs.argmax())
    
    abstain = CLASSES[top_idx] == "background" or float(mean_probs[top_idx]) < min_confidence
    return {
        "mean": float(fungal_samples.mean()),   # P(fungal) — this is the risk score
        "std": float(fungal_samples.std()),     # MC-dropout spread on that probability
        "class": CLASSES[top_idx],
        "class_probs": {c: float(p) for c, p in zip(CLASSES, mean_probs)},
        "abstain": bool(abstain),
        "score_head": float(score_mean),        # legacy regression output, for reference
    }


def mc_dropout_predict(model, image_tensor, n_passes=25, min_confidence=0.65):
    """Returns 'mean' = P(fungal_fruiting_body), NOT the regression head.

    The regression head is trained on network_score, which compute_network_score
    derives deterministically from `class` — so it is a lossy re-encoding of the
    classifier and nothing more. Reading risk off it also silently floors every
    fungal prediction at 0.7, which is above app.py's 0.6 alert threshold, so any
    argmax==fungal tile became a sampling site regardless of confidence. The class
    probability is the calibratable quantity, so risk comes from that.
    """
    device = next(model.parameters()).device
    probs, scores = _mc_forward(model, image_tensor.to(device), n_passes)
    return _summarise(probs[:, 0].mean(axis=0), probs[:, 0, FUNGAL_IDX],
                      scores[:, 0].mean(), min_confidence)


if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)