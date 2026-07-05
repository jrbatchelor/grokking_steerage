#!/usr/bin/env python3
"""
Grokking + Benford Experiment (Local Version)
Includes steerage mechanisms + returns structured metrics.
"""

import argparse
import json
import math
import os
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Optional AMP imports (graceful fallback if not available)
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    autocast = None
    GradScaler = None


# ----------------------------- Utilities -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_benford_stats(values: np.ndarray):
    values = np.asarray(values).flatten()
    values = np.abs(values)
    values = values[values > 1e-30]
    if len(values) < 10:
        return None
    logs = np.log10(values)
    mantissas = values / (10 ** np.floor(logs))
    digits = np.floor(mantissas).astype(int)
    digits = np.clip(digits, 1, 9)
    counts = np.bincount(digits, minlength=10)[1:10]
    total = counts.sum()
    if total == 0:
        return None
    freq = counts / total
    benford = np.array([math.log10(1 + 1.0 / d) for d in range(1, 10)])
    chi2 = np.sum((freq - benford)**2 / (benford + 1e-12))
    p1 = freq[0]
    return {'chi2': float(chi2), 'p1': float(p1)}


# ----------------------------- Models -----------------------------
class Stabilizer(nn.Module):
    def __init__(self, hidden_dim: int, stabilizer_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, stabilizer_dim),
            nn.GELU(),
            nn.Linear(stabilizer_dim, hidden_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.net(x)


class GrokkingMLP(nn.Module):
    def __init__(self, p: int, embed_dim: int = 32, hidden_dim: int = 512,
                 num_hidden_layers: int = 3, use_stabilizer: bool = False):
        super().__init__()
        self.p = p
        self.use_stabilizer = use_stabilizer
        self.embed = nn.Embedding(p, embed_dim)
        layers = []
        in_dim = 2 * embed_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        if use_stabilizer:
            self.stabilizer = Stabilizer(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, p)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        emb = self.embed(src)
        flat = emb.view(emb.size(0), -1)
        hidden = self.net(flat)
        if self.use_stabilizer:
            hidden = self.stabilizer(hidden)
        return self.fc_out(hidden)


# ----------------------------- Data -----------------------------
def generate_modular_data(p: int, train_frac: float = 0.45, seed: int = 42):
    set_seed(seed)
    inputs, labels = [], []
    for a in range(p):
        for b in range(p):
            inputs.append([a, b])
            labels.append((a + b) % p)
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    n_total = len(inputs)
    n_train = int(n_total * train_frac)
    perm = torch.randperm(n_total)
    return (inputs[perm[:n_train]], labels[perm[:n_train]]), \
           (inputs[perm[n_train:]], labels[perm[n_train:]])


# ----------------------------- Training -----------------------------
def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    (train_in, train_lab), (test_in, test_lab) = generate_modular_data(
        args.p, args.train_frac, args.seed
    )

    # Optimized DataLoader settings for RTX 3060
    batch_size = getattr(args, 'batch_size', 512)  # Increased default for better GPU utilization
    num_workers = getattr(args, 'num_workers', 8)
    persistent_workers = getattr(args, 'persistent_workers', True) and num_workers > 0

    train_ds = TensorDataset(train_in, train_lab)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None
    )
    train_iter = cycle(train_loader)

    model = GrokkingMLP(
        p=args.p,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        use_stabilizer=args.use_stabilizer
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # === Automatic Mixed Precision (AMP) Setup ===
    use_amp = getattr(args, 'use_amp', True) and device.type == 'cuda' and AMP_AVAILABLE
    scaler = GradScaler() if use_amp else None

    if use_amp:
        print(f"[OPTIMIZATION] Using Automatic Mixed Precision (AMP) for faster training")
    if getattr(args, 'compile_model', False):
        try:
            model = torch.compile(model)
            print(f"[OPTIMIZATION] Model compiled with torch.compile()")
        except Exception as e:
            print(f"[WARNING] torch.compile() failed: {e}")

    # History tracking
    history = {
        "steps": [],
        "train_losses": [],
        "test_losses": [],
        "test_accs": [],
        "weight_norms": [],
        "benford_chi2": [],
        "benford_p1": [],
    }

    step = 0
    while step < args.max_steps:
        batch_in, batch_lab = next(train_iter)
        batch_in, batch_lab = batch_in.to(device), batch_lab.to(device)

        optimizer.zero_grad()

        # === Forward pass with Automatic Mixed Precision ===
        if use_amp:
            with autocast():
                logits = model(batch_in)
                loss = criterion(logits, batch_lab)

            # === Steerage Mechanisms (must be inside autocast for AMP) ===
            # (All steerage code that was previously after backward is now here)

        else:
            logits = model(batch_in)
            loss = criterion(logits, batch_lab)

        # === Steerage Mechanisms (outside autocast block for clarity) ===
        # Mirror-Closure
        if getattr(args, 'use_mirror_closure', False):
            sym_errors = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.ndim == 2 and 'weight' in name:
                    if param.shape[0] == param.shape[1]:
                        err = torch.norm(param - param.t()) / (param.numel() ** 0.5 + 1e-8)
                        sym_errors.append(err)
            if sym_errors:
                avg_sym = torch.stack(sym_errors).mean()
                loss = loss + getattr(args, 'mirror_lambda', 0.01) * avg_sym

        # Polarity Gradient Steering
        if getattr(args, 'use_polarity_steering', False):
            with torch.no_grad():
                emb = model.embed(batch_in)
                flat = emb.view(emb.size(0), -1)
                hidden = model.net(flat).detach()
            if not hasattr(args, '_polarity_anomaly_ema'):
                args._polarity_anomaly_ema = hidden.mean(dim=0)
                args._polarity_resolution_ema = hidden.mean(dim=0)
            if step > 2000:
                beta = 0.99
                args._polarity_anomaly_ema   = beta * args._polarity_anomaly_ema   + (1 - beta) * hidden.mean(dim=0)
                args._polarity_resolution_ema = beta * args._polarity_resolution_ema + (1 - beta) * hidden.mean(dim=0)
                direction = args._polarity_resolution_ema - args._polarity_anomaly_ema
                direction = direction / (direction.norm() + 1e-8)
                proj = (hidden @ direction).mean()
                loss = loss + getattr(args, 'polarity_lambda', 0.05) * (-proj)

        # Internal Multi-Agent Mirror Closure
        if getattr(args, 'use_internal_mirror_closure', False):
            with torch.no_grad():
                emb = model.embed(batch_in)
                flat = emb.view(emb.size(0), -1)
                hidden = model.net(flat).detach()
            views = []
            num_agents = getattr(args, 'num_internal_agents', 4)
            for _ in range(num_agents):
                noise = torch.randn_like(hidden) * 0.01
                view = hidden + noise
                views.append(view)
            views = torch.stack(views)
            mean_view = views.mean(dim=0)
            consistency_loss = sum(F.mse_loss(views[i], mean_view) for i in range(num_agents))
            mirror_closure_loss = consistency_loss / num_agents
            loss = loss + getattr(args, 'internal_mirror_lambda', 0.05) * mirror_closure_loss

        # Epistemic Self-Improvement Loss
        if getattr(args, 'use_epistemic_self_improvement', False):
            if step >= getattr(args, 'epistemic_start_step', 2000):
                with torch.no_grad():
                    emb = model.embed(batch_in)
                    flat = emb.view(emb.size(0), -1)
                    hidden = model.net(flat).detach()
                if (not hasattr(args, '_epistemic_target_ema') or
                        args._epistemic_target_ema.shape != hidden.shape):
                    args._epistemic_target_ema = hidden.clone()
                beta = getattr(args, 'epistemic_ema_beta', 0.995)
                args._epistemic_target_ema = beta * args._epistemic_target_ema + (1 - beta) * hidden
                if args._epistemic_target_ema.shape == hidden.shape:
                    loss = loss + getattr(args, 'epistemic_lambda', 0.03) * F.mse_loss(hidden, args._epistemic_target_ema)

        # Polarity Navigation Regularization
        if getattr(args, 'use_polarity_navigation', False):
            if step >= 3000:
                with torch.no_grad():
                    emb = model.embed(batch_in)
                    flat = emb.view(emb.size(0), -1)
                    hidden = model.net(flat).detach()
                    noise_strong = torch.randn_like(hidden) * getattr(args, 'polarity_noise_strong', 0.15)
                    noise_weak   = torch.randn_like(hidden) * getattr(args, 'polarity_noise_weak', 0.02)
                    view_emp = hidden + noise_weak
                    view_ded = hidden + noise_strong
                    W = model.fc_out.weight.detach()
                    b = model.fc_out.bias.detach() if model.fc_out.bias is not None else None
                    loss_emp_val = float(criterion(F.linear(view_emp, W, b), batch_lab))
                    loss_ded_val = float(criterion(F.linear(view_ded, W, b), batch_lab))
                    polarity_loss_val = abs(loss_emp_val - loss_ded_val)
                loss = loss + getattr(args, 'polarity_navigation_lambda', 0.02) * polarity_loss_val

        # Holonomy Regularization (occasional)
        if args.use_holonomy_reg and (step + 1) % getattr(args, 'holonomy_check_interval', 100) == 0:
            with torch.no_grad():
                emb = model.embed(batch_in)
                flat = emb.view(emb.size(0), -1)
                hidden = model.net(flat)
                noise = 0.01 * torch.randn_like(hidden)
                perturbed = hidden + noise
            expansion = (perturbed - hidden).norm(dim=1).mean() / (noise.norm(dim=1).mean() + 1e-8)
            if expansion > getattr(args, 'holonomy_target', 0.95):
                loss = loss + args.holonomy_lambda * (expansion - getattr(args, 'holonomy_target', 0.95))**2

        # === Backward pass (only once, after all terms are added) ===
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        step += 1

        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                t_logits = model(train_in.to(device))
                t_loss = criterion(t_logits, train_lab.to(device)).item()
                t_acc = (t_logits.argmax(1) == train_lab.to(device)).float().mean().item()

                v_logits = model(test_in.to(device))
                v_loss = criterion(v_logits, test_lab.to(device)).item()
                v_acc = (v_logits.argmax(1) == test_lab.to(device)).float().mean().item()

                w_norm = sum(p.norm().item()**2 for p in model.parameters() if p.requires_grad) ** 0.5

                w_vals = []
                for name, param in model.named_parameters():
                    if param.requires_grad and 'weight' in name:
                        w_vals.append(param.detach().cpu().numpy().flatten())
                benford_stats = compute_benford_stats(np.concatenate(w_vals)) if w_vals else None

            history["steps"].append(step)
            history["train_losses"].append(t_loss)
            history["test_losses"].append(v_loss)
            history["test_accs"].append(v_acc)
            history["weight_norms"].append(w_norm)

            if benford_stats:
                history["benford_chi2"].append(benford_stats['chi2'])
                history["benford_p1"].append(benford_stats['p1'])
            else:
                history["benford_chi2"].append(np.nan)
                history["benford_p1"].append(np.nan)

            print(f"Step {step:5d} | Train loss {t_loss:.4f} acc {t_acc:.3f} | "
                  f"Test loss {v_loss:.4f} acc {v_acc:.3f}")

            model.train()

    # Compute final metrics
    final_metrics = {
        "final_test_acc": history["test_accs"][-1] if history["test_accs"] else 0.0,
        "steps_to_0.5": None,
        "steps_to_0.9": None,
        "weight_norm_reduction": None,
    }

    # Calculate steps to thresholds
    test_accs_arr = np.array(history["test_accs"])
    steps_arr = np.array(history["steps"])

    above_50 = np.where(test_accs_arr >= 0.5)[0]
    above_90 = np.where(test_accs_arr >= 0.9)[0]

    if len(above_50) > 0:
        final_metrics["steps_to_0.5"] = int(steps_arr[above_50[0]])
    if len(above_90) > 0:
        final_metrics["steps_to_0.9"] = int(steps_arr[above_90[0]])

    if history["weight_norms"]:
        final_metrics["weight_norm_reduction"] = float(
            (history["weight_norms"][0] - history["weight_norms"][-1]) / history["weight_norms"][0]
        )

    # Merge and return
    result = {**history, **final_metrics}

    if output_dir:
        np.savez(output_dir / "grokking_benford_metrics.npz", **history)
        with open(output_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

    return result


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--wd", type=float, default=0.40)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--train_frac", type=float, default=0.45)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_polarity_steering", action="store_true")
    parser.add_argument("--polarity_lambda", type=float, default=0.05)
    parser.add_argument("--use_holonomy_reg", action="store_true")
    parser.add_argument("--holonomy_lambda", type=float, default=0.03)
    parser.add_argument("--use_stabilizer", action="store_true")
    parser.add_argument("--use_mirror_closure", action="store_true")
    parser.add_argument("--mirror_lambda", type=float, default=0.01)

    # Internal Multi-Agent Mirror Closure
    parser.add_argument("--use_internal_mirror_closure", action="store_true")
    parser.add_argument("--num_internal_agents", type=int, default=4)
    parser.add_argument("--internal_mirror_lambda", type=float, default=0.05)

    # Epistemic Self-Improvement Loss
    parser.add_argument("--use_epistemic_self_improvement", action="store_true")
    parser.add_argument("--epistemic_lambda", type=float, default=0.03)
    parser.add_argument("--epistemic_ema_beta", type=float, default=0.995)
    parser.add_argument("--epistemic_start_step", type=int, default=2000)

    # Polarity Navigation Regularization
    parser.add_argument("--use_polarity_navigation", action="store_true")
    parser.add_argument("--polarity_navigation_lambda", type=float, default=0.02)
    parser.add_argument("--polarity_noise_strong", type=float, default=0.15)
    parser.add_argument("--polarity_noise_weak", type=float, default=0.02)

    # === Performance Optimization Arguments ===
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size (default: 512 for better GPU utilization)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader num_workers (default: 8)")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Enable Automatic Mixed Precision (default: True on CUDA)")
    parser.add_argument("--compile_model", action="store_true",
                        help="Compile model with torch.compile() for extra speed (PyTorch 2.0+)")
    parser.add_argument("--persistent_workers", action="store_true", default=True,
                        help="Use persistent workers in DataLoader (default: True)")

    args = parser.parse_args()
    result = run_experiment(args)
    print(f"\nFinal Test Accuracy: {result.get('final_test_acc', 'N/A'):.4f}")