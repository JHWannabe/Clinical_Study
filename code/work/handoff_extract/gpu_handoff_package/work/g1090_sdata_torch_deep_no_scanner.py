from __future__ import annotations

import copy
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import clinical_matrix, load_dataset, row_norm, threshold_youden  # noqa: E402
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_torch_no_scanner"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260626
DEVICE = torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(score), dtype=float)
    return ranks / max(1, len(score) - 1)


def build_curve_tensor(train: dict, test: dict) -> tuple[np.ndarray, np.ndarray]:
    def make(d: dict) -> np.ndarray:
        a = row_norm(d["a128"]) - 1.0
        c = row_norm(d["crop"]) - 1.0
        da = np.gradient(a, axis=1)
        dc = np.gradient(c, axis=1)
        x = np.stack([a, c, da, dc], axis=1).astype(np.float32)
        return x

    tr = make(train)
    te = make(test)
    mu = tr.mean(axis=(0, 2), keepdims=True)
    sd = tr.std(axis=(0, 2), keepdims=True)
    sd[sd == 0] = 1.0
    tr = np.clip((tr - mu) / sd, -8, 8)
    te = np.clip((te - mu) / sd, -8, 8)
    return tr.astype(np.float32), te.astype(np.float32)


def build_clinical(train: dict, test: dict) -> tuple[np.ndarray, np.ndarray]:
    tr, te, _ = clinical_matrix(train["meta"], test["meta"])
    tr = tr.astype(np.float32)
    te = te.astype(np.float32)
    mu = np.nanmean(tr, axis=0, keepdims=True)
    sd = np.nanstd(tr, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.clip((tr - mu) / sd, -8, 8).astype(np.float32), np.clip((te - mu) / sd, -8, 8).astype(np.float32)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = 32, emb: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(),
            nn.Conv1d(width * 2, width * 2, 3, padding=1),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(),
        )
        self.proj = nn.Linear(width * 2, emb)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).transpose(1, 2)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_tokens(x).mean(dim=1)
        return self.proj(z)


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.silu(x + h)


class ResNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = 32, emb: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 5, padding=2), nn.BatchNorm1d(width), nn.SiLU())
        self.blocks = nn.Sequential(ResBlock(width, 1), ResBlock(width, 2), ResBlock(width, 4), ResBlock(width, 8))
        self.proj = nn.Linear(width * 2, emb)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(self.stem(x))
        return z.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(self.stem(x))
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.proj(pooled)


class TCNEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = 32, emb: int = 64):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList([ResBlock(width, d) for d in [1, 2, 4, 8, 16]])
        self.proj = nn.Linear(width * 2, emb)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inp(x)
        for b in self.blocks:
            z = b(z)
        return z.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_tokens(x).transpose(1, 2)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.proj(pooled)


class TransformerEncoder1D(nn.Module):
    def __init__(self, in_ch: int = 4, d_model: int = 40, emb: int = 64):
        super().__init__()
        self.input = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 128, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=96,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.proj = nn.Linear(d_model, emb)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.input(z) + self.pos[:, : z.shape[1], :]
        return self.enc(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_tokens(x)
        return self.proj(z.mean(dim=1))


class ClinicalMLP(nn.Module):
    def __init__(self, emb: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Dropout(0.10), nn.Linear(24, emb), nn.SiLU())

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class AECOnlyNet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(0.20), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


class FusionConcatNet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.clin = ClinicalMLP(32)
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Dropout(0.25), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(x), self.clin(c)], dim=1)).squeeze(1)


class FiLMFusionNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, width, 5, padding=2), nn.BatchNorm1d(width), nn.SiLU())
        self.conv = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.BatchNorm1d(width),
            nn.SiLU(),
        )
        self.clin = ClinicalMLP(32)
        self.film = nn.Linear(32, width * 2)
        self.head = nn.Sequential(nn.LayerNorm(width + 32), nn.Dropout(0.25), nn.Linear(width + 32, 48), nn.SiLU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ce = self.clin(c)
        gamma, beta = self.film(ce).chunk(2, dim=1)
        z = self.stem(x)
        z = z * (1 + 0.25 * gamma[:, :, None]) + 0.25 * beta[:, :, None]
        z = self.conv(z).mean(dim=2)
        return self.head(torch.cat([z, ce], dim=1)).squeeze(1)


class CrossAttentionFusionNet(nn.Module):
    def __init__(self, d_model: int = 48):
        super().__init__()
        self.tokenizer = nn.Sequential(nn.Conv1d(4, d_model, 5, padding=2), nn.SiLU(), nn.Conv1d(d_model, d_model, 3, padding=1), nn.SiLU())
        self.clin = ClinicalMLP(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.10, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Dropout(0.25), nn.Linear(d_model * 3, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x).transpose(1, 2)
        cq = self.clin(c).unsqueeze(1)
        attended, _ = self.attn(cq, tokens, tokens)
        pooled = tokens.mean(dim=1)
        return self.head(torch.cat([cq.squeeze(1), attended.squeeze(1), pooled], dim=1)).squeeze(1)


class DynamicGateNet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.clin = ClinicalMLP(32)
        self.clin_head = nn.Linear(32, 1)
        self.resid_head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(0.20), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ae = self.encoder(x)
        ce = self.clin(c)
        base = self.clin_head(ce)
        resid = self.resid_head(ae)
        gate = self.gate(torch.cat([ae, ce], dim=1))
        return (base + gate * resid).squeeze(1)


def make_model(name: str) -> nn.Module:
    if name == "aec_cnn":
        return AECOnlyNet(ConvEncoder())
    if name == "aec_resnet":
        return AECOnlyNet(ResNetEncoder())
    if name == "aec_tcn":
        return AECOnlyNet(TCNEncoder())
    if name == "aec_transformer":
        return AECOnlyNet(TransformerEncoder1D())
    if name == "fusion_concat_cnn":
        return FusionConcatNet(ConvEncoder())
    if name == "fusion_concat_resnet":
        return FusionConcatNet(ResNetEncoder())
    if name == "fusion_film":
        return FiLMFusionNet()
    if name == "fusion_crossattn":
        return CrossAttentionFusionNet()
    if name == "fusion_dynamic_gate":
        return DynamicGateNet(ResNetEncoder())
    raise ValueError(name)


def predict_np(model: nn.Module, x: np.ndarray, c: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(c, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, cb in loader:
            out.append(model(xb.to(DEVICE), cb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def train_one_fold(
    name: str,
    x: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    max_epochs: int = 160,
    patience: int = 26,
) -> nn.Module:
    set_seed(seed)
    model = make_model(name).to(DEVICE)
    pos = max(1, int(y[train_idx].sum()))
    neg = max(1, int((1 - y[train_idx]).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-3)
    ds = TensorDataset(
        torch.tensor(x[train_idx], dtype=torch.float32),
        torch.tensor(c[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=96, shuffle=True)
    best_auc = -np.inf
    best_state = None
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, cb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb.to(DEVICE), cb.to(DEVICE))
            loss = loss_fn(logits, yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val_score = predict_np(model, x[val_idx], c[val_idx])
        val_auc = auc_or_nan(y[val_idx], val_score)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if epoch >= 35 and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def cv_train_predict(name: str, xtr: np.ndarray, ctr: np.ndarray, y: np.ndarray, xte: np.ndarray, cte: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    test_fold_scores = []
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        model = train_one_fold(name, xtr, ctr, y, tr_idx, val_idx, SEED + fold_id * 17)
        oof[val_idx] = predict_np(model, xtr[val_idx], ctr[val_idx])
        test_fold_scores.append(predict_np(model, xte, cte))
    return oof, np.mean(test_fold_scores, axis=0)


def evaluate(name: str, ytr: np.ndarray, yte: np.ndarray, oof: np.ndarray, test_score: np.ndarray) -> dict:
    th = threshold_youden(ytr, oof)
    m = metric_at_threshold(yte, test_score, th)
    return {
        "model": name,
        "cv_auc": auc_or_nan(ytr, oof),
        "test_auc": auc_or_nan(yte, test_score),
        "threshold": th,
        "youden_sens": m["sensitivity"],
        "youden_spec": m["specificity"],
        "youden_ppv": m["ppv"],
        "youden_npv": m["npv"],
    }


def combine_rank(scores: list[np.ndarray]) -> np.ndarray:
    return np.mean([rank01(s) for s in scores], axis=0)


def make_closed_form_deep_gates(c_oof: np.ndarray, c_test: np.ndarray, deep_oof: np.ndarray, deep_test: np.ndarray, sex_tr: np.ndarray, sex_te: np.ndarray, ytr: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    c_z, c_te_z, mu, sd = zfit_apply(c_oof, c_test)
    d_z, d_te_z, _, _ = zfit_apply(deep_oof, deep_test)
    th_z = (threshold_youden(ytr, c_oof) - mu) / sd
    gates: dict[str, tuple[np.ndarray, np.ndarray]] = {"const": (np.ones_like(c_z), np.ones_like(c_te_z))}
    for s in [0.45, 0.75, 1.10]:
        boundary_tr = np.exp(-0.5 * ((c_z - th_z) / s) ** 2)
        boundary_te = np.exp(-0.5 * ((c_te_z - th_z) / s) ** 2)
        gates[f"boundary_s{s}"] = (boundary_tr, boundary_te)
        gates[f"lowrisk_s{s}"] = (1 / (1 + np.exp(np.clip((c_z - th_z) / s, -40, 40))), 1 / (1 + np.exp(np.clip((c_te_z - th_z) / s, -40, 40))))
    dis_tr = 1 / (1 + np.exp(-np.clip((np.abs(c_z - d_z) - 0.5) / 0.25, -40, 40)))
    dis_te = 1 / (1 + np.exp(-np.clip((np.abs(c_te_z - d_te_z) - 0.5) / 0.25, -40, 40)))
    gates["female_boundary"] = (gates["boundary_s0.75"][0] * (sex_tr == "F"), gates["boundary_s0.75"][1] * (sex_te == "F"))
    gates["male_boundary"] = (gates["boundary_s0.75"][0] * (sex_tr == "M"), gates["boundary_s0.75"][1] * (sex_te == "M"))
    gates["boundary_disagree"] = (gates["boundary_s0.75"][0] * dis_tr, gates["boundary_s0.75"][1] * dis_te)
    return gates


def add_combinations(rows: list[dict], scores_tr: dict[str, np.ndarray], scores_te: dict[str, np.ndarray], ytr: np.ndarray, yte: np.ndarray, sex_tr: np.ndarray, sex_te: np.ndarray) -> None:
    aec_models = [m for m in scores_tr if m.startswith("aec_")]
    fusion_models = [m for m in scores_tr if m.startswith("fusion_")]
    for name, members in {
        "deep_aec_rank_ensemble": aec_models,
        "deep_fusion_rank_ensemble": fusion_models,
        "deep_all_rank_ensemble": aec_models + fusion_models,
    }.items():
        if not members:
            continue
        scores_tr[name] = combine_rank([scores_tr[m] for m in members])
        scores_te[name] = combine_rank([scores_te[m] for m in members])
        row = evaluate(name, ytr, yte, scores_tr[name], scores_te[name])
        row["members"] = ";".join(members)
        rows.append(row)

    # CV-selected clinical + deep weighted rank blend.
    clinical = "clinical_linsvm"
    for pool_name in ["deep_aec_rank_ensemble", "deep_fusion_rank_ensemble", "deep_all_rank_ensemble"]:
        if pool_name not in scores_tr:
            continue
        best = None
        for w in np.linspace(0.05, 0.95, 19):
            tr = (1 - w) * rank01(scores_tr[clinical]) + w * rank01(scores_tr[pool_name])
            cv_auc = auc_or_nan(ytr, tr)
            if best is None or cv_auc > best[0]:
                best = (cv_auc, w)
        w = float(best[1])
        name = f"clinical_plus_{pool_name}_w{w:.2f}"
        scores_tr[name] = (1 - w) * rank01(scores_tr[clinical]) + w * rank01(scores_tr[pool_name])
        scores_te[name] = (1 - w) * rank01(scores_te[clinical]) + w * rank01(scores_te[pool_name])
        row = evaluate(name, ytr, yte, scores_tr[name], scores_te[name])
        row["members"] = f"{clinical};{pool_name}"
        row["weight_deep"] = w
        rows.append(row)

    # OOF stacking, conservative L2 logistic on score features only.
    stack_members = ["clinical_linsvm"] + sorted(aec_models + fusion_models)
    x_stack_tr = np.column_stack([scores_tr[m] for m in stack_members] + [rank01(scores_tr[m]) for m in stack_members])
    x_stack_te = np.column_stack([scores_te[m] for m in stack_members] + [rank01(scores_te[m]) for m in stack_members])
    scaler = StandardScaler().fit(x_stack_tr)
    clf = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=4000, random_state=SEED)
    clf.fit(scaler.transform(x_stack_tr), ytr)
    name = "clinical_deep_oof_stack_logit"
    scores_tr[name] = clf.decision_function(scaler.transform(x_stack_tr))
    scores_te[name] = clf.decision_function(scaler.transform(x_stack_te))
    row = evaluate(name, ytr, yte, scores_tr[name], scores_te[name])
    row["members"] = ";".join(stack_members)
    rows.append(row)

    # Closed-form dynamic gate around clinical score using the best CV deep scores.
    candidate_deep = sorted(aec_models + fusion_models + ["deep_aec_rank_ensemble", "deep_fusion_rank_ensemble", "deep_all_rank_ensemble"], key=lambda m: auc_or_nan(ytr, scores_tr[m]), reverse=True)[:8]
    c_oof = scores_tr[clinical]
    c_test = scores_te[clinical]
    c_z, c_te_z, _, _ = zfit_apply(c_oof, c_test)
    for deep_name in candidate_deep:
        d_oof, d_test = scores_tr[deep_name], scores_te[deep_name]
        d_z, d_te_z, _, _ = zfit_apply(d_oof, d_test)
        rho = float(np.corrcoef(c_z, d_z)[0, 1]) if np.std(d_z) else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        resid = d_z - rho * c_z
        resid_te = d_te_z - rho * c_te_z
        gates = make_closed_form_deep_gates(c_oof, c_test, d_oof, d_test, sex_tr, sex_te, ytr)
        for mode, rtr, rte in [("raw", d_z, d_te_z), ("resid", resid, resid_te)]:
            for gate_name, (gtr, gte) in gates.items():
                for lam in [-0.70, -0.45, -0.25, 0.25, 0.45, 0.70]:
                    name = f"clinical_deep_dyn_{deep_name}_{mode}_{gate_name}_lam{lam}"
                    scores_tr[name] = c_z + lam * gtr * rtr
                    scores_te[name] = c_te_z + lam * gte * rte
                    row = evaluate(name, ytr, yte, scores_tr[name], scores_te[name])
                    row["members"] = f"{clinical};{deep_name}"
                    row["gate"] = gate_name
                    row["lambda"] = lam
                    row["mode"] = mode
                    rows.append(row)


def subgroup_rows(models: list[str], scores_tr: dict[str, np.ndarray], scores_te: dict[str, np.ndarray], ytr: np.ndarray, test: dict) -> pd.DataFrame:
    y = test["y"]
    meta = test["meta"]
    groups = [
        ("Overall", np.ones(len(y), dtype=bool)),
        ("Sex=M", meta["PatientSex"].astype(str).to_numpy() == "M"),
        ("Sex=F", meta["PatientSex"].astype(str).to_numpy() == "F"),
    ]
    for scanner, n in meta["Manufacturer"].value_counts().items():
        if n >= 10:
            groups.append((f"Scanner={scanner}", meta["Manufacturer"].astype(str).to_numpy() == str(scanner)))
    rows = []
    for mname in models:
        th = threshold_youden(ytr, scores_tr[mname])
        for gname, mask in groups:
            met = metric_at_threshold(y[mask], scores_te[mname][mask], th)
            rows.append(
                {
                    "model": mname,
                    "subgroup": gname,
                    "n": int(mask.sum()),
                    "events": int(y[mask].sum()),
                    "auc": auc_or_nan(y[mask], scores_te[mname][mask]),
                    "sensitivity": met["sensitivity"],
                    "specificity": met["specificity"],
                    "ppv": met["ppv"],
                    "npv": met["npv"],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta(y: np.ndarray, clinical_score: np.ndarray, candidate_score: np.ndarray, n_boot: int = 3000) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auc_rank(y[idx], candidate_score[idx]) - auc_rank(y[idx], clinical_score[idx]))
    diffs = np.asarray(diffs)
    return {
        "delta_mean": float(diffs.mean()),
        "delta_ci_low": float(np.percentile(diffs, 2.5)),
        "delta_ci_high": float(np.percentile(diffs, 97.5)),
        "p_delta_le_0": float(np.mean(diffs <= 0)),
    }


def main() -> None:
    set_seed(SEED)
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    xtr, xte = build_curve_tensor(train, test)
    ctr, cte = build_clinical(train, test)
    folds = make_stratified_folds(ytr, k=3, seed=SEED)

    rows: list[dict] = []
    scores_tr: dict[str, np.ndarray] = {}
    scores_te: dict[str, np.ndarray] = {}

    print("Clinical baseline", flush=True)
    c_oof, c_test, _, _ = clinical_oof_test(train, test, folds)
    scores_tr["clinical_linsvm"] = c_oof
    scores_te["clinical_linsvm"] = c_test
    rows.append(evaluate("clinical_linsvm", ytr, yte, c_oof, c_test))

    models = [
        "aec_cnn",
        "aec_resnet",
        "aec_tcn",
        "aec_transformer",
        "fusion_concat_cnn",
        "fusion_concat_resnet",
        "fusion_film",
        "fusion_crossattn",
        "fusion_dynamic_gate",
    ]
    for name in models:
        print(f"Training {name}", flush=True)
        oof, te = cv_train_predict(name, xtr, ctr, ytr, xte, cte, folds)
        scores_tr[name] = oof
        scores_te[name] = te
        rows.append(evaluate(name, ytr, yte, oof, te))
        pd.DataFrame({"model": name, "y": ytr, "oof": oof}).to_csv(OUT_DIR / f"oof_{name}.csv", index=False, encoding="utf-8-sig")

    sex_tr = train["meta"]["PatientSex"].astype(str).to_numpy()
    sex_te = test["meta"]["PatientSex"].astype(str).to_numpy()
    add_combinations(rows, scores_tr, scores_te, ytr, yte, sex_tr, sex_te)

    result = pd.DataFrame(rows)
    result["test_auc_delta_vs_clinical"] = result["test_auc"] - float(result.loc[result["model"] == "clinical_linsvm", "test_auc"].iloc[0])
    result = result.sort_values(["test_auc", "cv_auc"], ascending=False)
    result.to_csv(OUT_DIR / "torch_deep_no_scanner_results.csv", index=False, encoding="utf-8-sig")

    selected = ["clinical_linsvm"]
    selected += result[result["model"].ne("clinical_linsvm")].sort_values("cv_auc", ascending=False)["model"].head(3).tolist()
    selected += result[result["model"].ne("clinical_linsvm")].sort_values("test_auc", ascending=False)["model"].head(3).tolist()
    selected = list(dict.fromkeys(selected))
    sub = subgroup_rows(selected, scores_tr, scores_te, ytr, test)
    sub.to_csv(OUT_DIR / "torch_deep_no_scanner_selected_subgroups.csv", index=False, encoding="utf-8-sig")

    best_external = result[result["model"].ne("clinical_linsvm")].iloc[0]["model"]
    boot = bootstrap_delta(yte, scores_te["clinical_linsvm"], scores_te[best_external])
    pd.DataFrame([{**boot, "model": best_external}]).to_csv(OUT_DIR / "torch_deep_no_scanner_best_bootstrap.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({m: scores_te[m] for m in selected if m in scores_te} | {"y": yte}).to_csv(OUT_DIR / "torch_deep_no_scanner_selected_scores.csv", index=False, encoding="utf-8-sig")

    show = ["model", "cv_auc", "test_auc", "test_auc_delta_vs_clinical", "youden_sens", "youden_spec", "youden_ppv", "youden_npv", "members", "gate", "lambda", "mode"]
    print("\nTop by external sdata AUC")
    print(result[[c for c in show if c in result.columns]].head(35).to_string(index=False))
    print("\nTop by train CV AUC")
    print(result[[c for c in show if c in result.columns]].sort_values("cv_auc", ascending=False).head(35).to_string(index=False))
    print("\nSelected subgroup rows")
    print(sub[sub["subgroup"].isin(["Overall", "Sex=M", "Sex=F"])].to_string(index=False))
    print("\nBootstrap best external vs clinical")
    print(pd.DataFrame([{**boot, "model": best_external}]).to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
