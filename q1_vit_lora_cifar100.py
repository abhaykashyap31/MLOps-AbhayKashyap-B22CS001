# ================================
# 🔐 0. AUTH SETUP (SECRETS)
# ================================
import os
import json
import shutil
import sys
from pathlib import Path


_RESULTS_STREAM = None


def redirect_process_output(filename):
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


redirect_process_output("q1_results.log")

HF_TOKEN_GLOBAL = None

def load_local_env():
    env_candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
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
    # 1. Try local .env / environment variables
    val = os.environ.get(key)
    if val:
        return val

    # 2. Try Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        val = user_secrets.get_secret(key)
        if val:
            return val
    except Exception:
        pass
        
    # 3. Try Colab Secrets
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val: 
            return val
    except Exception:
        pass

    return None

def setup_auth():
    global HF_TOKEN_GLOBAL
    hf_token = get_secret("HF_TOKEN")
    wandb_key = get_secret("wandb_api_key") or get_secret("WANDB_API_KEY") or get_secret("WANDB_KEY")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        HF_TOKEN_GLOBAL = hf_token
        print("✅ HF token loaded")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("✅ WandB key loaded")
        
    if not hf_token and not wandb_key:
        print("⚠️ Secrets not set")

setup_auth()


# ================================
# 📦 1. Install dependencies
# ================================
import subprocess

def maybe_install(requirements):
    for req in requirements:
        pkg = req.split("==")[0].split(">=")[0]
        try:
            __import__(pkg.replace("-", "_"))
        except:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", req])

maybe_install([
    "timm>=1.0.0",
    "peft>=0.13.0",
    "optuna>=3.6.0",
    "wandb>=0.17.0",
    "huggingface_hub>=0.24.0",
    "tabulate"  # For README markdown tables
])


# ================================
# ⚙️ 2. Config
# ================================
class Config:
    data_dir = "./data"
    output_dir = "./q1_outputs"
    epochs = 3
    batch_size = 16
    lr = 2e-4
    seed = 42

    use_wandb = True
    wandb_project = "vit-lora-cifar100"

    hf_repo_id = "subilal/lora-vit-cifar100"  

cfg = Config()


# ================================
# 📚 3. Imports
# ================================
import random, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import timm
import wandb
import optuna

from peft import LoraConfig, get_peft_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


# ================================
# 🔐 WandB Setup
# ================================
def init_wandb():
    if os.getenv("WANDB_API_KEY") and cfg.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        return True
    print("⚠️ WandB disabled")
    return False

cfg.use_wandb = init_wandb()


# ================================
# 🌱 Utils
# ================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_run_info(run_info):
    return {
        "name": str(run_info["name"]),
        "lora": str(run_info["lora"]),
        "rank": int(run_info["rank"]),
        "alpha": int(run_info["alpha"]),
        "dropout": float(run_info["dropout"]),
        "test_acc": float(run_info["test_acc"]),
        "params": int(run_info["params"]),
        "hub_pushable": bool(run_info["hub_pushable"])
    }


def prepare_hub_export(run_info):
    run_info = normalize_run_info(run_info)
    run_name = run_info["name"]
    export_dir = Path(cfg.output_dir) / "hf_exports" / run_name
    model_dir = Path(cfg.output_dir) / f"{run_name}_model"
    model_file = Path(cfg.output_dir) / f"{run_name}_model.pt"

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if model_dir.exists():
        shutil.copytree(model_dir, export_dir / "model")
    elif model_file.exists():
        shutil.copy2(model_file, export_dir / model_file.name)
    else:
        print(f"⚠️ No saved checkpoint found for {run_name}")
        return None

    for suffix in ["history.csv", "curves.png", "gradients.png", "hist.png"]:
        source = Path(cfg.output_dir) / f"{run_name}_{suffix}"
        if source.exists():
            shutil.copy2(source, export_dir / source.name)

    with open(export_dir / "metadata.json", "w") as f:
        json.dump({
            **run_info,
            "base_model": "vit_small_patch16_224",
            "dataset": "CIFAR-100",
            "attention_lora_target": "fused_qkv",
            "attention_lora_note": "timm ViT exposes a fused qkv projection, so LoRA is applied to the combined QKV attention projection instead of separate q_proj, k_proj, and v_proj layers.",
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.lr
        }, f, indent=2)

    with open(export_dir / "README.md", "w") as f:
        f.write(
            f"# {run_name}\n\n"
            f"- Base model: `vit_small_patch16_224`\n"
            f"- Dataset: `CIFAR-100`\n"
            f"- Test accuracy: `{run_info['test_acc']:.4f}`\n"
            f"- LoRA enabled: `{run_info['lora']}`\n"
            f"- LoRA rank: `{run_info['rank']}`\n"
            f"- LoRA alpha: `{run_info['alpha']}`\n"
            f"- Trainable params: `{run_info['params']}`\n\n"
            "Attention note: timm ViT exposes a fused `qkv` projection, so LoRA is attached to the combined QKV layer rather than separate `q_proj`, `k_proj`, and `v_proj` modules.\n\n"
            "This folder was exported automatically from the assignment training script.\n"
        )

    return export_dir


def upload_run_to_hub(run_info, repo_id, token):
    run_info = normalize_run_info(run_info)
    run_name = run_info["name"]
    export_dir = prepare_hub_export(run_info)
    if export_dir is None:
        return False

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(export_dir),
            path_in_repo=f"runs/{run_name}",
            token=token,
            commit_message=f"Upload {run_name} artifacts (acc={run_info['test_acc']:.4f})"
        )
        print(f"✅ Uploaded {run_name} to Hugging Face Hub under runs/{run_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {run_name} to Hugging Face Hub: {e}")
        return False


# ================================
# 📦 Data
# ================================
def build_loaders():
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR100(root=cfg.data_dir, train=True, transform=train_transform, download=True)
    full_eval = datasets.CIFAR100(root=cfg.data_dir, train=True, transform=eval_transform, download=True)
    test = datasets.CIFAR100(root=cfg.data_dir, train=False, transform=eval_transform, download=True)

    idx = np.arange(len(full_train))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=full_train.targets)

    return (
        DataLoader(Subset(full_train, train_idx), batch_size=cfg.batch_size, shuffle=True),
        DataLoader(Subset(full_eval, val_idx), batch_size=cfg.batch_size),
        DataLoader(test, batch_size=cfg.batch_size)
    )


# ================================
# 🧠 Model
# ================================
def build_model():
    return timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=100)


def baseline_model(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def lora_model(model, r, alpha, partial_freeze=False):
    for p in model.parameters():
        p.requires_grad = False
        
    if partial_freeze:
        # Unfreeze the last block's layer norm and self attention for partial freeze experiment
        for p in model.blocks[-1].parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True

    cfg_lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        # timm ViT uses a fused qkv projection, so this is the correct LoRA injection point.
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["head"]
    )
    return get_peft_model(model, cfg_lora)


# ================================
# 🔁 Training + Grad Norm
# ================================
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)

    total_loss, total_acc = 0, 0
    grad_norm = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        if train:
            optimizer.zero_grad()
            loss.backward()

            # 👉 LoRA grad norm tracking
            for n, p in model.named_parameters():
                if "lora" in n and p.grad is not None:
                    grad_norm += p.grad.norm().item()

            optimizer.step()

        total_loss += loss.item()
        total_acc += (out.argmax(1) == y).float().mean().item()

    return total_loss/len(loader), total_acc/len(loader), grad_norm


# ================================
# 🧪 Test + Classwise Accuracy
# ================================
def evaluate(model, loader):
    model.eval()
    correct = np.zeros(100)
    total = np.zeros(100)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)

            for i in range(len(y)):
                total[y[i].item()] += 1
                if pred[i] == y[i]:
                    correct[y[i].item()] += 1

    class_acc = correct / (total + 1e-8)
    return class_acc.mean(), class_acc


# ================================
# 🚀 Experiment
# ================================
results = []
Path(cfg.output_dir).mkdir(exist_ok=True, parents=True)

def run_experiment(name, r=None, alpha=None, partial_freeze=False):
    model = build_model()

    if r is None:
        model = baseline_model(model)
    else:
        model = lora_model(model, r, alpha, partial_freeze=partial_freeze)

    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    history = []

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=name)

    for epoch in range(cfg.epochs):
        tr_l, tr_a, grad = run_epoch(model, train_loader, criterion, optimizer)
        va_l, va_a, _ = run_epoch(model, val_loader, criterion)

        history.append([epoch, tr_l, va_l, tr_a, va_a, grad])

        if cfg.use_wandb:
            wandb.log({
                "train_loss": tr_l,
                "val_loss": va_l,
                "train_acc": tr_a,
                "val_acc": va_a,
                "grad_norm": grad,
                "lr": optimizer.param_groups[0]["lr"]
            })

        scheduler.step()

    test_acc, class_acc = evaluate(model, test_loader)

    # 📊 Save History CSV
    history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "grad_norm"])
    history_df.to_csv(f"{cfg.output_dir}/{name}_history.csv", index=False)

    # 📉 Save Loss and Accuracy Graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker='o')
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker='o')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} - Loss Curve")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", marker='o')
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", marker='o')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} - Accuracy Curve")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/{name}_curves.png")
    plt.close()

    # 📉 Save LoRA gradient norm graph
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["grad_norm"], label="LoRA Grad Norm", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title(f"{name} - LoRA Gradient Norm")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/{name}_gradients.png")
    plt.close()

    # 📊 Save histogram
    plt.figure(figsize=(12, 5))
    plt.bar(range(100), class_acc)
    plt.title(f"{name} - Class-wise Accuracy")
    plt.xlabel("Class Index")
    plt.ylabel("Accuracy")
    plt.savefig(f"{cfg.output_dir}/{name}_hist.png")
    plt.close()

    params = count_trainable_params(model)

    results.append({
        "name": name,
        "lora": "Yes" if r is not None else "No",
        "rank": r if r is not None else 0,
        "alpha": alpha if alpha is not None else 0,
        "dropout": 0.1 if r is not None else 0.0,
        "test_acc": test_acc,
        "params": params,
        "hub_pushable": bool(r is not None and not partial_freeze)
    })

    # Save model weights to disk to allow pushing the best later
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(f"{cfg.output_dir}/{name}_model")
    else:
        torch.save(model.state_dict(), f"{cfg.output_dir}/{name}_model.pt")

    if cfg.use_wandb:
        wandb.finish()

    return model, test_acc


# ================================
# 🏁 MAIN
# ================================
if __name__ == "__main__":
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders()

    print("🚀 Running Baseline")
    run_experiment("baseline")

    print("\n🚀 Running LoRA Grid Search")
    for r in [2, 4, 8]:
        for a in [2, 4, 8]:
            print(f"Running r={r}, alpha={a}")
            run_experiment(f"lora_r{r}_a{a}", r, a)

    print("\n🚀 Running Optional Partial Freeze + LoRA")
    run_experiment("lora_partial_freeze", r=8, alpha=8, partial_freeze=True)


    # ================================
    # 🔎 Optuna Hyperparameter Search
    # ================================
    print("\n🚀 Running Optuna Hyperparameter Search")
    def objective(trial):
        # Search space
        r = trial.suggest_categorical("r", [2, 4, 8, 16])
        alpha = trial.suggest_categorical("alpha", [2, 4, 8, 16])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        model = build_model()
        model = lora_model(model, r, alpha)
        model.to(device)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        criterion = nn.CrossEntropyLoss()

        # Fast 1-epoch evaluation for Optuna tuning
        run_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        _, va_a, _ = run_epoch(model, val_loader, criterion)
        
        return va_a

    # Create and run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)  # Keeping trials low for assignment time limits
    print(f"✅ Best Optuna Parameters: {study.best_params}")

    print("\n🚀 Training Optuna Best LoRA Configuration")
    best_optuna_params = study.best_params
    best_optuna_name = f"optuna_best_r{best_optuna_params['r']}_a{best_optuna_params['alpha']}"
    original_lr = cfg.lr
    cfg.lr = best_optuna_params["lr"]
    run_experiment(best_optuna_name, r=best_optuna_params["r"], alpha=best_optuna_params["alpha"])
    cfg.lr = original_lr


    # ================================
    # 📊 Save Results & Select Best
    # ================================
    df = pd.DataFrame(results)
    df.to_csv(f"{cfg.output_dir}/final_results.csv", index=False)

    # Export README Table (Markdown format)
    with open(f"{cfg.output_dir}/README_results.md", "w") as f:
        f.write("# LoRA ViT CIFAR-100 Experiment Results\n\n")
        f.write(df.to_markdown(index=False))

    print("\n✅ FINAL RESULTS SAVED TO CSV and MD")
    print(df.sort_values("test_acc", ascending=False).to_string(index=False))

    print("\n🚀 Selecting Best Model...")
    best_run = df.sort_values("test_acc", ascending=False).iloc[0]
    best_name = best_run["name"]
    print(f"🏆 Best run: {best_name} with Accuracy: {best_run['test_acc']:.4f}")

    # ================================
    # ☁️ Push Best Model to Hub
    # ================================
    hf_token = HF_TOKEN_GLOBAL
    print(f"HF_TOKEN exists: {hf_token is not None}")
    print(f"Repo: {cfg.hf_repo_id}")
    if hf_token and cfg.hf_repo_id:
        push_targets = [best_run]
        push_candidates = df[df["hub_pushable"]].sort_values("test_acc", ascending=False)
        if not push_candidates.empty:
            best_lora_run = push_candidates.iloc[0]
            if best_lora_run["name"] != best_run["name"]:
                push_targets.append(best_lora_run)

        uploaded_count = 0
        for run_info in push_targets:
            print(f"\n☁️ Uploading {run_info['name']} to Hugging Face Hub ({cfg.hf_repo_id})...")
            if upload_run_to_hub(run_info, cfg.hf_repo_id, hf_token):
                uploaded_count += 1

        if uploaded_count == 0:
            print("⚠️ All Hugging Face upload attempts failed.")
        else:
            print(f"✅ Uploaded {uploaded_count} model artifact set(s) to Hugging Face Hub.")
    else:
        print("\n❌ HF upload skipped — token missing")