# ================================
# 🔐 0. AUTH SETUP (SECRETS)
# ================================
import os
import sys
from pathlib import Path


_RESULTS_STREAM = None


def redirect_process_output(filename: str) -> None:
    global _RESULTS_STREAM
    try:
        results_path = Path(__file__).resolve().with_name(filename)
    except NameError:
        results_path = Path.cwd() / filename

    results_path.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_STREAM = open(results_path, "w", encoding="utf-8", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(_RESULTS_STREAM.fileno(), 1)
    os.dup2(_RESULTS_STREAM.fileno(), 2)


redirect_process_output("q2_results.log")

def load_local_env():
    env_candidates = [
        os.path.join(os.getcwd(), ".env"),
    ]
    try:
        env_candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    except NameError:
        pass
    seen = set()
    for env_path in env_candidates:
        if env_path in seen or not os.path.exists(env_path):
            continue
        seen.add(env_path)
        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return
        except OSError:
            pass

load_local_env()

def get_secret(key):
 
    val = os.environ.get(key)
    if val:
        return val

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        val = user_secrets.get_secret(key)
        if val:
            return val
    except Exception:
        pass
        
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val: 
            return val
    except Exception:
        pass

    return None

def setup_auth():
    hf_token = get_secret("HF_TOKEN")
    wandb_key = get_secret("wandb_api_key") or get_secret("WANDB_API_KEY") or get_secret("WANDB_KEY")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("✅ HF token loaded")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("✅ WandB key loaded")
        
    if not hf_token and not wandb_key:
        print("⚠️ Secrets not set")

setup_auth()


import argparse
import copy
import json
import random
import subprocess
from typing import Dict, List, Tuple


def maybe_install(requirements: List[str]) -> None:
    import site
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
        
    import_name_overrides = {
        "adversarial-robustness-toolbox": "art",
    }
    for req in requirements:
        pkg_name = req.split("==")[0].split(">=")[0]
        import_name = import_name_overrides.get(pkg_name, pkg_name.replace("-", "_"))
        try:
            __import__(import_name)
        except Exception:
            last_error = None
            install_commands = [
                [sys.executable, "-m", "pip", "install", "-q", req],
                [sys.executable, "-m", "pip", "install", "-q", "--user", req],
                [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", req],
            ]
            for cmd in install_commands:
                try:
                    subprocess.check_call(cmd)
                    last_error = None
                    break
                except subprocess.CalledProcessError as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            import importlib
            importlib.invalidate_caches()


maybe_install(["adversarial-robustness-toolbox>=1.18.0", "wandb>=0.17.0"])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
from art.attacks.evasion import (
    BasicIterativeMethod,
    FastGradientMethod,
    ProjectedGradientDescentPyTorch,
)
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_wandb: bool = False,
) -> Dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    history = []
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss += loss.item() * bs
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += bs

        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                test_loss += loss.item() * bs
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += bs

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_acc": train_correct / train_total,
            "test_loss": test_loss / test_total,
            "test_acc": test_correct / test_total,
        }
        history.append(row)
        best_test_acc = max(best_test_acc, row["test_acc"])

        if use_wandb:
            wandb.log(
                {
                    "classifier/epoch": epoch,
                    "classifier/train_loss": row["train_loss"],
                    "classifier/train_acc": row["train_acc"],
                    "classifier/test_loss": row["test_loss"],
                    "classifier/test_acc": row["test_acc"],
                    "classifier/lr": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_acc={row['train_acc']:.4f}, test_acc={row['test_acc']:.4f}"
        )

    return {"history": history, "best_test_acc": best_test_acc}


def fgsm_from_scratch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    model.eval()
    x = images.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x.grad.detach().sign()
    x_adv = x + epsilon * grad_sign
    return torch.clamp(x_adv, min=-3.0, max=3.0).detach()


@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def eval_accuracy_on_tensors(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 256
) -> float:
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def collect_n_samples(loader: DataLoader, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for images, labels in loader:
        xs.append(images)
        ys.append(labels)
        if sum(v.size(0) for v in xs) >= n:
            break
    x = torch.cat(xs, dim=0)[:n]
    y = torch.cat(ys, dim=0)[:n]
    return x, y


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616), device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def save_comparison_grid(
    clean: torch.Tensor,
    adv_scratch: torch.Tensor,
    adv_art: torch.Tensor,
    out_file: Path,
    max_show: int = 10,
) -> None:
    n = min(max_show, clean.size(0))
    clean = denormalize(clean[:n]).cpu().clamp(0, 1)
    adv_scratch = denormalize(adv_scratch[:n]).cpu().clamp(0, 1)
    adv_art = denormalize(adv_art[:n]).cpu().clamp(0, 1)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    titles = ["Clean", "FGSM Scratch", "FGSM ART"]
    for r, imgs in enumerate([clean, adv_scratch, adv_art]):
        for c in range(n):
            axes[r, c].imshow(np.transpose(imgs[c].numpy(), (1, 2, 0)))
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(titles[r])
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def save_attack_gallery(
    clean: torch.Tensor,
    adv: torch.Tensor,
    attack_name: str,
    out_file: Path,
    max_show: int = 10,
) -> None:
    n = min(max_show, clean.size(0))
    clean = denormalize(clean[:n]).cpu().clamp(0, 1)
    adv = denormalize(adv[:n]).cpu().clamp(0, 1)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
    for c in range(n):
        axes[0, c].imshow(np.transpose(clean[c].numpy(), (1, 2, 0)))
        axes[0, c].axis("off")
        axes[1, c].imshow(np.transpose(adv[c].numpy(), (1, 2, 0)))
        axes[1, c].axis("off")
    axes[0, 0].set_ylabel("Clean")
    axes[1, 0].set_ylabel(attack_name)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def make_wandb_image_table(
    clean: torch.Tensor,
    adv: torch.Tensor,
    attack_name: str,
    max_show: int = 10,
):
    n = min(max_show, clean.size(0))
    clean = denormalize(clean[:n]).cpu().clamp(0, 1)
    adv = denormalize(adv[:n]).cpu().clamp(0, 1)

    table = wandb.Table(columns=["index", "clean", "adversarial"])
    for idx in range(n):
        clean_img = np.transpose(clean[idx].numpy(), (1, 2, 0))
        adv_img = np.transpose(adv[idx].numpy(), (1, 2, 0))
        table.add_data(
            idx,
            wandb.Image(clean_img, caption=f"clean_{idx}"),
            wandb.Image(adv_img, caption=f"{attack_name.lower()}_{idx}"),
        )
    return table


def plot_eps_vs_drop(results_df: pd.DataFrame, out_file: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(results_df["epsilon"], results_df["fgsm_scratch_acc"], marker="o", label="FGSM scratch")
    plt.plot(results_df["epsilon"], results_df["fgsm_art_acc"], marker="o", label="FGSM ART")
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial accuracy")
    plt.title("Perturbation strength vs performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def make_art_classifier(
    model: nn.Module,
    device: torch.device,
    lr: float,
) -> PyTorchClassifier:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(-3.0, 3.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


class DetectorFeatureDataset(Dataset):
    def __init__(self, clean_features: np.ndarray, adv_features: np.ndarray):
        self.x = np.concatenate([clean_features, adv_features], axis=0).astype(np.float32)
        self.y = np.concatenate(
            [np.zeros(len(clean_features), dtype=np.int64), np.ones(len(adv_features), dtype=np.int64)],
            axis=0,
        )
        idx = np.arange(len(self.x))
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class MLPDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def forward_with_features(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Use the trained classifier's penultimate representation as a richer signal than raw pixels.
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    features = torch.flatten(x, 1)
    logits = model.fc(features)
    return logits, features


@torch.no_grad()
def build_detector_features(
    classifier: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    classifier.eval()
    features = []

    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        xb_smooth = F.avg_pool2d(xb, kernel_size=3, stride=1, padding=1)

        logits, penultimate = forward_with_features(classifier, xb)
        logits_smooth, penultimate_smooth = forward_with_features(classifier, xb_smooth)

        probs = logits.softmax(dim=1)
        probs_smooth = logits_smooth.softmax(dim=1)
        top2 = probs.topk(k=2, dim=1).values

        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        margin = (top2[:, :1] - top2[:, 1:2]).contiguous()
        prob_shift = (probs - probs_smooth).abs().mean(dim=1, keepdim=True)
        logit_shift = torch.norm(logits - logits_smooth, p=2, dim=1, keepdim=True)
        feature_shift = torch.norm(penultimate - penultimate_smooth, p=2, dim=1, keepdim=True)
        pred_change = (probs.argmax(dim=1) != probs_smooth.argmax(dim=1)).float().unsqueeze(1)

        residual = xb - xb_smooth
        residual_abs_mean = residual.abs().mean(dim=(2, 3))
        residual_std = residual.flatten(2).std(dim=2)
        residual_energy = residual.pow(2).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        batch_features = torch.cat(
            [
                penultimate,
                probs,
                entropy,
                margin,
                prob_shift,
                logit_shift,
                feature_shift,
                pred_change,
                residual_abs_mean,
                residual_std,
                residual_energy,
            ],
            dim=1,
        )
        features.append(batch_features.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float32)


def train_detector(
    attack_name: str,
    classifier: nn.Module,
    clean_x_train: np.ndarray,
    adv_x_train: np.ndarray,
    clean_x_test: np.ndarray,
    adv_x_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    use_wandb: bool = False,
) -> Dict:
    clean_train_features = build_detector_features(classifier, clean_x_train, device, batch_size)
    adv_train_features = build_detector_features(classifier, adv_x_train, device, batch_size)
    clean_test_features = build_detector_features(classifier, clean_x_test, device, batch_size)
    adv_test_features = build_detector_features(classifier, adv_x_test, device, batch_size)

    feature_mean = np.mean(
        np.concatenate([clean_train_features, adv_train_features], axis=0), axis=0, keepdims=True
    )
    feature_std = np.std(
        np.concatenate([clean_train_features, adv_train_features], axis=0), axis=0, keepdims=True
    )
    feature_std = np.clip(feature_std, 1e-6, None)

    clean_train_features = (clean_train_features - feature_mean) / feature_std
    adv_train_features = (adv_train_features - feature_mean) / feature_std
    clean_test_features = (clean_test_features - feature_mean) / feature_std
    adv_test_features = (adv_test_features - feature_mean) / feature_std

    train_ds = DetectorFeatureDataset(clean_train_features, adv_train_features)
    test_ds = DetectorFeatureDataset(clean_test_features, adv_test_features)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MLPDetector(input_dim=train_ds.x.shape[1])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_total += y.size(0)

        model.eval()
        test_loss, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item() * y.size(0)
                te_correct += (logits.argmax(dim=1) == y).sum().item()
                te_total += y.size(0)

        te_acc = te_correct / max(te_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)
        scheduler.step(te_acc)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())
        hist.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(tr_total, 1),
                "train_acc": tr_acc,
                "test_loss": test_loss / max(te_total, 1),
                "test_acc": te_acc,
            }
        )
        if use_wandb:
            wandb.log(
                {
                    f"detector/{attack_name.lower()}/epoch": epoch,
                    f"detector/{attack_name.lower()}/train_loss": hist[-1]["train_loss"],
                    f"detector/{attack_name.lower()}/train_acc": tr_acc,
                    f"detector/{attack_name.lower()}/test_loss": hist[-1]["test_loss"],
                    f"detector/{attack_name.lower()}/test_acc": te_acc,
                    f"detector/{attack_name.lower()}/lr": optimizer.param_groups[0]["lr"],
                }
            )
        print(
            f"[{attack_name}] Epoch {epoch}/{epochs}, "
            f"train_acc={tr_acc:.4f}, detector test acc={te_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return {"attack": attack_name, "best_test_acc": best_acc, "history": hist}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--clf_epochs", type=int, default=20)
    parser.add_argument("--clf_lr", type=float, default=2e-3)
    parser.add_argument("--clf_wd", type=float, default=5e-4)
    parser.add_argument("--detector_epochs", type=int, default=20)
    parser.add_argument("--detector_lr", type=float, default=1e-3)
    parser.add_argument("--detector_train_samples", type=int, default=10000)
    parser.add_argument("--detector_test_samples", type=int, default=2000)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dlops-assignment5-q2-art")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and (not args.no_cuda) else "cpu")
    print(f"Using device: {device}")

    use_wandb = args.use_wandb or bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        if os.getenv("WANDB_API_KEY"):
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            job_type="assignment-q2",
        )
        print(f"WandB URL: {wandb.run.url}")
    else:
        print("⚠️ WandB disabled")

    train_loader, test_loader = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)

    # Part (i): Train clean ResNet18 from scratch
    clf = torchvision.models.resnet18(weights=None)
    clf.fc = nn.Linear(clf.fc.in_features, 10)
    train_info = train_classifier(
        clf,
        train_loader,
        test_loader,
        device=device,
        epochs=args.clf_epochs,
        lr=args.clf_lr,
        weight_decay=args.clf_wd,
        use_wandb=use_wandb,
    )
    clean_acc = eval_accuracy(clf.to(device), test_loader, device)
    print(f"Clean test accuracy: {clean_acc:.4f}")

    # Prepare arrays for ART and attack generation.
    x_test_t, y_test_t = collect_n_samples(test_loader, n=10000)
    x_test_np = x_test_t.numpy().astype(np.float32)
    y_test_np = y_test_t.numpy().astype(np.int64)

    art_classifier = make_art_classifier(clf, device=device, lr=args.clf_lr)

    rows = []
    for eps in args.epsilons:
        # FGSM from scratch
        x_adv_scratch_batches = []
        for i in range(0, len(x_test_t), args.batch_size):
            xb = x_test_t[i : i + args.batch_size].to(device)
            yb = y_test_t[i : i + args.batch_size].to(device)
            x_adv = fgsm_from_scratch(clf, xb, yb, epsilon=eps)
            x_adv_scratch_batches.append(x_adv.detach().cpu())
        x_adv_scratch = torch.cat(x_adv_scratch_batches, dim=0).numpy().astype(np.float32)
        acc_scratch = eval_accuracy_on_tensors(clf, x_adv_scratch, y_test_np, device)

        # FGSM with ART
        fgsm_art = FastGradientMethod(estimator=art_classifier, eps=eps)
        x_adv_art = fgsm_art.generate(x=x_test_np)
        acc_art = eval_accuracy_on_tensors(clf, x_adv_art, y_test_np, device)

        rows.append(
            {
                "epsilon": eps,
                "clean_acc": clean_acc,
                "fgsm_scratch_acc": acc_scratch,
                "fgsm_art_acc": acc_art,
                "drop_scratch": clean_acc - acc_scratch,
                "drop_art": clean_acc - acc_art,
            }
        )
        print(
            f"eps={eps:.4f} | clean={clean_acc:.4f}, scratch={acc_scratch:.4f}, art={acc_art:.4f}"
        )

    attack_df = pd.DataFrame(rows)
    attack_df.to_csv(out_dir / "fgsm_comparison_table.csv", index=False)
    plot_eps_vs_drop(attack_df, out_dir / "epsilon_vs_accuracy.png")
    if use_wandb:
        wandb.log(
            {
                "fgsm/comparison_table": wandb.Table(dataframe=attack_df),
                "fgsm/clean_test_accuracy": clean_acc,
            }
        )

    # Visual comparison using a representative epsilon.
    eps_vis = args.epsilons[min(2, len(args.epsilons) - 1)]
    x_vis, y_vis = collect_n_samples(test_loader, n=10)
    x_adv_vis_scratch = fgsm_from_scratch(clf, x_vis.to(device), y_vis.to(device), epsilon=eps_vis).cpu()
    fgsm_art_vis = FastGradientMethod(estimator=art_classifier, eps=eps_vis)
    x_adv_vis_art = torch.tensor(
        fgsm_art_vis.generate(x=x_vis.numpy().astype(np.float32)), dtype=torch.float32
    )
    save_comparison_grid(
        x_vis, x_adv_vis_scratch, x_adv_vis_art, out_file=out_dir / "fgsm_visual_comparison.png", max_show=10
    )
    save_attack_gallery(x_vis, x_adv_vis_scratch, "FGSM Scratch", out_dir / "fgsm_scratch_samples.png", max_show=10)
    save_attack_gallery(x_vis, x_adv_vis_art, "FGSM ART", out_dir / "fgsm_art_samples.png", max_show=10)

    # Also save 10 samples each for FGSM-scratch, FGSM-ART, PGD, BIM (for WandB/report).
    pgd = ProjectedGradientDescentPyTorch(estimator=art_classifier, eps=0.05, eps_step=0.01, max_iter=10)
    bim = BasicIterativeMethod(estimator=art_classifier, eps=0.05, eps_step=0.01, max_iter=10)
    x_adv_pgd_vis = torch.tensor(pgd.generate(x=x_vis.numpy().astype(np.float32)), dtype=torch.float32)
    x_adv_bim_vis = torch.tensor(bim.generate(x=x_vis.numpy().astype(np.float32)), dtype=torch.float32)
    save_attack_gallery(x_vis, x_adv_pgd_vis, "PGD", out_dir / "pgd_samples.png", max_show=10)
    save_attack_gallery(x_vis, x_adv_bim_vis, "BIM", out_dir / "bim_samples.png", max_show=10)
    save_comparison_grid(x_vis, x_adv_pgd_vis, x_adv_bim_vis, out_file=out_dir / "pgd_bim_visuals.png", max_show=10)

    if use_wandb:
        wandb.log(
            {
                "samples/fgsm_visual_comparison": wandb.Image(str(out_dir / "fgsm_visual_comparison.png")),
                "samples/pgd_bim_visuals": wandb.Image(str(out_dir / "pgd_bim_visuals.png")),
                "samples/fgsm_scratch_gallery": wandb.Image(str(out_dir / "fgsm_scratch_samples.png")),
                "samples/fgsm_art_gallery": wandb.Image(str(out_dir / "fgsm_art_samples.png")),
                "samples/pgd_gallery": wandb.Image(str(out_dir / "pgd_samples.png")),
                "samples/bim_gallery": wandb.Image(str(out_dir / "bim_samples.png")),
                "samples/fgsm_scratch_table": make_wandb_image_table(x_vis, x_adv_vis_scratch, "fgsm_scratch"),
                "samples/fgsm_art_table": make_wandb_image_table(x_vis, x_adv_vis_art, "fgsm_art"),
                "samples/pgd_table": make_wandb_image_table(x_vis, x_adv_pgd_vis, "pgd"),
                "samples/bim_table": make_wandb_image_table(x_vis, x_adv_bim_vis, "bim"),
            }
        )

    # Part (ii): Detection model using PGD and BIM-generated adversarial data.
    x_train_t, y_train_t = collect_n_samples(train_loader, n=args.detector_train_samples)
    x_test_det_t, y_test_det_t = collect_n_samples(test_loader, n=args.detector_test_samples)
    x_train_np = x_train_t.numpy().astype(np.float32)
    x_test_det_np = x_test_det_t.numpy().astype(np.float32)

    pgd_train = ProjectedGradientDescentPyTorch(
        estimator=art_classifier, eps=0.05, eps_step=0.01, max_iter=10
    )
    pgd_test = ProjectedGradientDescentPyTorch(
        estimator=art_classifier, eps=0.05, eps_step=0.01, max_iter=10
    )
    bim_train = BasicIterativeMethod(estimator=art_classifier, eps=0.05, eps_step=0.01, max_iter=10)
    bim_test = BasicIterativeMethod(estimator=art_classifier, eps=0.05, eps_step=0.01, max_iter=10)

    x_adv_pgd_train = pgd_train.generate(x=x_train_np)
    x_adv_pgd_test = pgd_test.generate(x=x_test_det_np)
    x_adv_bim_train = bim_train.generate(x=x_train_np)
    x_adv_bim_test = bim_test.generate(x=x_test_det_np)

    det_pgd = train_detector(
        attack_name="PGD",
        classifier=clf,
        clean_x_train=x_train_np,
        adv_x_train=x_adv_pgd_train,
        clean_x_test=x_test_det_np,
        adv_x_test=x_adv_pgd_test,
        device=device,
        epochs=args.detector_epochs,
        batch_size=args.batch_size,
        lr=args.detector_lr,
        use_wandb=use_wandb,
    )
    det_bim = train_detector(
        attack_name="BIM",
        classifier=clf,
        clean_x_train=x_train_np,
        adv_x_train=x_adv_bim_train,
        clean_x_test=x_test_det_np,
        adv_x_test=x_adv_bim_test,
        device=device,
        epochs=args.detector_epochs,
        batch_size=args.batch_size,
        lr=args.detector_lr,
        use_wandb=use_wandb,
    )

    detector_df = pd.DataFrame(
        [
            {"attack": "PGD", "best_detection_acc": det_pgd["best_test_acc"]},
            {"attack": "BIM", "best_detection_acc": det_bim["best_test_acc"]},
        ]
    )
    detector_df.to_csv(out_dir / "detector_results.csv", index=False)

    # Save summary JSON for report-ready values.
    summary = {
        "clean_test_accuracy": clean_acc,
        "classifier_best_test_accuracy": train_info["best_test_acc"],
        "fgsm_table_csv": str(out_dir / "fgsm_comparison_table.csv"),
        "fgsm_visuals": str(out_dir / "fgsm_visual_comparison.png"),
        "fgsm_scratch_samples": str(out_dir / "fgsm_scratch_samples.png"),
        "fgsm_art_samples": str(out_dir / "fgsm_art_samples.png"),
        "pgd_bim_visuals": str(out_dir / "pgd_bim_visuals.png"),
        "pgd_samples": str(out_dir / "pgd_samples.png"),
        "bim_samples": str(out_dir / "bim_samples.png"),
        "detector_results_csv": str(out_dir / "detector_results.csv"),
        "detector_pgd_best_acc": det_pgd["best_test_acc"],
        "detector_bim_best_acc": det_bim["best_test_acc"],
    }
    with open(out_dir / "q2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.log(
            {
                "clean_test_accuracy": clean_acc,
                "fgsm/table": wandb.Table(dataframe=attack_df),
                "detector/table": wandb.Table(dataframe=detector_df),
                "detector/pgd_best_test_acc": det_pgd["best_test_acc"],
                "detector/bim_best_test_acc": det_bim["best_test_acc"],
                "artifacts/q2_summary": summary,
            }
        )
        wandb.summary["clean_test_accuracy"] = clean_acc
        wandb.summary["classifier_best_test_accuracy"] = train_info["best_test_acc"]
        wandb.summary["detector_pgd_best_acc"] = det_pgd["best_test_acc"]
        wandb.summary["detector_bim_best_acc"] = det_bim["best_test_acc"]
        wandb.finish()

    torch.save(clf.state_dict(), out_dir / "resnet18_clean_classifier.pt")
    print("Q2 completed.")


if __name__ == "__main__":
    main()
