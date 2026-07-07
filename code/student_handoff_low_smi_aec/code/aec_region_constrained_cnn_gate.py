from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    SIGMA,
    auc_with_p,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
)


OUT_DIR = Path(r"C:\Users\user\OneDrive\Dokumen\radiation\outputs\aec_region_constrained_cnn_gate")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [20260701, 20260711]

# Broad windows around the hand-crafted late-dynamics features.
# 1-indexed inclusive AEC positions.
REGIONS = {
    "R1_slope_around_082_085": (76, 92),
    "R2_abs_slope_around_094_099": (88, 106),
    "R3_curv_around_103_110": (96, 118),
    "R4_curv_around_097_100": (90, 110),
}


@dataclass(frozen=True)
class TrainConfig:
    name: str
    dropout: float
    weight_decay: float
    lr: float
    low_weight: float
    aux_weight: float
    max_epochs: int = 120
    patience: int = 16
    batch_size: int = 96


CONFIGS = [
    TrainConfig("balanced_aux", dropout=0.25, weight_decay=1.0e-3, lr=8.0e-4, low_weight=5.0, aux_weight=0.25),
    TrainConfig("low_smi_guard", dropout=0.35, weight_decay=2.0e-3, lr=6.0e-4, low_weight=8.0, aux_weight=0.20),
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def row_z(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (x - mu) / sd


def d1(x: np.ndarray) -> np.ndarray:
    v = np.diff(x, axis=1)
    return np.column_stack([v[:, :1], v])


def d2(x: np.ndarray) -> np.ndarray:
    v = np.diff(d1(x), axis=1)
    return np.column_stack([v[:, :1], v])


def make_channels(norm: np.ndarray) -> np.ndarray:
    # Three channels: relative curve shape, slope, and curvature.
    # Each channel is patient-wise standardized to force morphology, not raw level.
    return np.stack([row_z(norm), row_z(d1(norm)), row_z(d2(norm))], axis=1).astype(np.float32)


def standardize_channels_train_apply(xg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = xg.mean(axis=(0, 2), keepdims=True)
    sd = xg.std(axis=(0, 2), keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return ((xg - mu) / sd).astype(np.float32), ((xs - mu) / sd).astype(np.float32)


def soft_2of4_prob(branch_logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(branch_logits)
    q = 1.0 - p
    p0 = q.prod(dim=1)
    p1 = torch.zeros_like(p0)
    for j in range(p.shape[1]):
        p1 = p1 + p[:, j] * torch.cat([q[:, :j], q[:, j + 1 :]], dim=1).prod(dim=1)
    out = 1.0 - p0 - p1
    return torch.clamp(out, 1e-5, 1.0 - 1e-5)


class RegionBranch(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 8, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.head(self.drop(pooled)).squeeze(1)


class RegionConstrainedCnn(nn.Module):
    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.regions = list(REGIONS.items())
        self.branches = nn.ModuleList([RegionBranch(dropout=dropout) for _ in self.regions])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = []
        for branch, (_, (start, end)) in zip(self.branches, self.regions):
            logits.append(branch(x[:, :, start - 1 : end]))
        branch_logits = torch.stack(logits, dim=1)
        gate_prob = soft_2of4_prob(branch_logits)
        return branch_logits, gate_prob


def train_loss(
    branch_logits: torch.Tensor,
    gate_prob: torch.Tensor,
    target_nonlow: torch.Tensor,
    sample_weight: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    low_weight = torch.where(target_nonlow < 0.5, torch.full_like(target_nonlow, cfg.low_weight), torch.ones_like(target_nonlow))
    weight = sample_weight * low_weight
    gate_loss = F.binary_cross_entropy(gate_prob, target_nonlow, weight=weight)
    branch_prob = torch.sigmoid(branch_logits)
    aux_target = target_nonlow[:, None].expand_as(branch_prob)
    aux_weight = weight[:, None].expand_as(branch_prob)
    aux_loss = F.binary_cross_entropy(branch_prob, aux_target, weight=aux_weight)
    return gate_loss + cfg.aux_weight * aux_loss


def clinical_positive_weights(c_g: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    mat = np.column_stack([(c_g >= thresholds[op]).astype(float) for op, _ in OPS])
    return (0.35 + mat.mean(axis=1)).astype(np.float32)


def stratified_folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


def predict_model(model: RegionConstrainedCnn, x: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    gates = []
    branches = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            bl, gp = model(xb)
            branches.append(torch.sigmoid(bl).cpu().numpy())
            gates.append(gp.cpu().numpy())
    return np.concatenate(gates), np.vstack(branches)


def train_one_fold(
    cfg: TrainConfig,
    x: np.ndarray,
    y_low: np.ndarray,
    sample_weight: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    model = RegionConstrainedCnn(dropout=cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    target = (1 - y_low).astype(np.float32)
    xt = torch.tensor(x[train_idx], dtype=torch.float32)
    yt = torch.tensor(target[train_idx], dtype=torch.float32)
    wt = torch.tensor(sample_weight[train_idx], dtype=torch.float32)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(target[val_idx], dtype=torch.float32, device=DEVICE)
    wv = torch.tensor(sample_weight[val_idx], dtype=torch.float32, device=DEVICE)

    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = xt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)
            wb = wt[idx].to(DEVICE)
            xb = xb + 0.02 * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            bl, gp = model(xb)
            loss = train_loss(bl, gp, yb, wb, cfg)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            blv, gpv = model(xv)
            val_loss = float(train_loss(blv, gpv, yv, wv, cfg).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_gate, val_branch = predict_model(model, x[val_idx])
    ext_gate, ext_branch = predict_model(model, x_ext)
    return val_gate, val_branch, ext_gate, ext_branch, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss)}


def crossfit_config(
    cfg: TrainConfig,
    xg: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    oof_gates_all = []
    ext_gates_all = []
    oof_branches_all = []
    ext_branches_all = []
    log_rows = []
    for seed in SEEDS:
        oof_gate = np.zeros(len(y), dtype=float)
        oof_branch = np.zeros((len(y), len(REGIONS)), dtype=float)
        ext_gates = []
        ext_branches = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            vg, vb, eg, eb, info = train_one_fold(cfg, xg, y, sample_weight, tr, va, xs, seed + fold_id * 101, )
            oof_gate[va] = vg
            oof_branch[va] = vb
            ext_gates.append(eg)
            ext_branches.append(eb)
            log_rows.append({"config": cfg.name, "seed": seed, "fold": fold_id, **info})
        oof_gates_all.append(oof_gate)
        oof_branches_all.append(oof_branch)
        ext_gates_all.append(np.mean(ext_gates, axis=0))
        ext_branches_all.append(np.mean(ext_branches, axis=0))
    return (
        np.mean(oof_gates_all, axis=0),
        np.mean(oof_branches_all, axis=0),
        np.mean(ext_gates_all, axis=0),
        np.mean(ext_branches_all, axis=0),
        pd.DataFrame(log_rows),
    )


def finite_summary(rows: list[dict], dataset: str) -> dict:
    sub = [r for r in rows if r["dataset"] == dataset]
    return {
        f"{dataset}_min_p_loss": float(np.nanmin([r["sensitivity_loss_p_exact"] for r in sub])),
        f"{dataset}_max_sens_loss": float(np.nanmax([r["sensitivity_loss"] for r in sub])),
        f"{dataset}_min_spec_gain": float(np.nanmin([r["specificity_gain"] for r in sub])),
        f"{dataset}_mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in sub])),
        f"{dataset}_max_fisher_p": float(np.nanmax([r["deesc_event_fisher_p"] for r in sub])),
        f"{dataset}_min_deesc_n": int(np.nanmin([r["deesc_n"] for r in sub])),
        f"{dataset}_mean_event_rate": float(np.nanmean([r["deesc_event_rate"] for r in sub])),
    }


def evaluate_cutoff(
    config_name: str,
    gate_g: np.ndarray,
    gate_s: np.ndarray,
    cutoff: float,
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict]:
    rows = []
    for dataset, d, c, score in [("g1090_internal", g, c_g, gate_g), ("sdata_external", s, c_s, gate_s)]:
        for op, _ in OPS:
            cpos = c >= thresholds[op]
            deesc = cpos & (score >= cutoff)
            rows.append(
                deesc_metric_row(
                    dataset,
                    f"region_cnn_soft2of4_cutoff_{cutoff:.4f}",
                    config_name,
                    op,
                    d["y"],
                    cpos,
                    deesc,
                )
            )
    return rows


def choose_cutoff_internal(config_name: str, gate_g: np.ndarray, g: dict, c_g: np.ndarray, thresholds: dict[str, float]) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.quantile(gate_g, np.linspace(0.50, 0.95, 46)))
    rows = []
    for cutoff in candidates:
        detail = evaluate_cutoff(config_name, gate_g, gate_g, float(cutoff), g, g, c_g, c_g, thresholds)
        # evaluate_cutoff returns duplicated g/s rows; use g1090 rows only.
        detail = [r for r in detail if r["dataset"] == "g1090_internal"]
        summ = finite_summary(detail, "g1090_internal")
        survives = (
            summ["g1090_internal_min_p_loss"] >= 0.05
            and summ["g1090_internal_min_spec_gain"] > 0
            and summ["g1090_internal_max_fisher_p"] < 0.05
            and summ["g1090_internal_min_deesc_n"] >= 20
            and summ["g1090_internal_max_sens_loss"] <= 0.08
        )
        score = (
            2.6 * summ["g1090_internal_min_spec_gain"]
            + 1.1 * summ["g1090_internal_mean_spec_gain"]
            - 0.8 * summ["g1090_internal_max_sens_loss"]
            - 0.01 * summ["g1090_internal_max_fisher_p"]
        )
        if not survives:
            score -= 10.0
        rows.append({"config": config_name, "cutoff": float(cutoff), "survives_internal": survives, "selection_score": score, **summ})
    df = pd.DataFrame(rows).sort_values(["survives_internal", "selection_score"], ascending=False)
    return float(df.iloc[0]["cutoff"]), df


def clinical_plus_auc(
    y_g: np.ndarray,
    y_s: np.ndarray,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    gate_g: np.ndarray,
    gate_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Higher clinical score means low-SMI risk. Higher gate score means low-risk, so use -gate.
    aec_risk_g = -np.log(np.clip(gate_g, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - gate_g, 1e-6, 1.0))
    aec_risk_s = -np.log(np.clip(gate_s, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - gate_s, 1e-6, 1.0))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260701)
    oof = np.zeros(len(y_g), dtype=float)
    for fold_id, (tr, va) in enumerate(folds.split(np.zeros(len(y_g)), y_g)):
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260701 + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], aec_risk_g[tr]]), y_g[tr])
        oof[va] = model.decision_function(np.column_stack([clinical_oof[va], aec_risk_g[va]]))
    final = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260799)
    final.fit(np.column_stack([clinical_oof, aec_risk_g]), y_g)
    ext = final.decision_function(np.column_stack([clinical_ext, aec_risk_s]))
    return oof, ext


def make_auc_rows(config_name: str, g: dict, s: dict, clinical_oof: np.ndarray, clinical_ext: np.ndarray, gate_g: np.ndarray, gate_s: np.ndarray) -> pd.DataFrame:
    combo_g, combo_s = clinical_plus_auc(g["y"], s["y"], clinical_oof, clinical_ext, gate_g, gate_s)
    rows = []
    for model, sg, ss in [
        ("clinical_only", clinical_oof, clinical_ext),
        (f"{config_name}_region_cnn_aec_only", -gate_g, -gate_s),
        (f"{config_name}_clinical_plus_region_cnn", combo_g, combo_s),
    ]:
        ai, pi = auc_with_p(g["y"], sg)
        ae, pe = auc_with_p(s["y"], ss)
        rows.append({"model": model, "internal_auc": ai, "internal_auc_p": pi, "external_auc": ae, "external_auc_p": pe})
    df = pd.DataFrame(rows)
    df["internal_delta_vs_clinical"] = df["internal_auc"] - df.loc[0, "internal_auc"]
    df["external_delta_vs_clinical"] = df["external_auc"] - df.loc[0, "external_auc"]
    return df


def branch_summary(config_name: str, branch_g: np.ndarray, branch_s: np.ndarray, g: dict, s: dict) -> pd.DataFrame:
    rows = []
    for dataset, branches, d in [("g1090_internal", branch_g, g), ("sdata_external", branch_s, s)]:
        y = d["y"].astype(bool)
        for j, region in enumerate(REGIONS):
            rows.append(
                {
                    "config": config_name,
                    "dataset": dataset,
                    "region": region,
                    "nonlow_mean_vote_prob": float(branches[~y, j].mean()),
                    "low_mean_vote_prob": float(branches[y, j].mean()),
                    "diff_nonlow_minus_low": float(branches[~y, j].mean() - branches[y, j].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_details(detail: pd.DataFrame, branch_df: pd.DataFrame, out_path: Path) -> None:
    labels = [op for op, _ in OPS]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7), constrained_layout=True)
    colors = {"g1090_internal": "#2F6B9A", "sdata_external": "#C54E2C"}
    for dataset in ["g1090_internal", "sdata_external"]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", color=colors[dataset], label=f"{dataset} spec gain")
        axes[0].plot(x, sub["sensitivity_loss"] * 100, marker="x", color=colors[dataset], ls="--", label=f"{dataset} sens loss")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Percentage points")
    axes[0].set_title("De-escalation tradeoff", loc="left", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for dataset in ["g1090_internal", "sdata_external"]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        axes[1].plot(np.arange(len(labels)), sub["deesc_event_rate"] * 100, marker="o", color=colors[dataset], label=dataset)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("De-escalated low-SMI rate (%)")
    axes[1].set_title("Event rate in de-escalated group", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    best = branch_df.copy()
    best["region_short"] = best["region"].str.replace("R", "R", regex=False).str.split("_").str[0]
    x = np.arange(len(REGIONS))
    width = 0.35
    for i, dataset in enumerate(["g1090_internal", "sdata_external"]):
        sub = best[best["dataset"].eq(dataset)]
        axes[2].bar(x + (i - 0.5) * width, sub["diff_nonlow_minus_low"], width=width, label=dataset, color=colors[dataset], alpha=0.85)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"R{i+1}" for i in range(len(REGIONS))])
    axes[2].set_ylabel("Vote prob: non-low minus low")
    axes[2].set_title("Branch vote direction", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False)
    fig.suptitle("Region-constrained CNN with differentiable 2-of-4 gate", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
    sample_weight = clinical_positive_weights(c_g, thresholds)

    all_auc = []
    all_detail = []
    all_cutoffs = []
    all_branch = []
    all_logs = []
    scores_by_config: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for cfg in CONFIGS:
        print(f"training {cfg.name}", flush=True)
        gate_g, branch_g, gate_s, branch_s, log_df = crossfit_config(cfg, xg, g["y"], sample_weight, xs)
        log_df.to_csv(OUT_DIR / f"{cfg.name}_training_log.csv", index=False)
        all_logs.append(log_df)
        cutoff, cutoff_df = choose_cutoff_internal(cfg.name, gate_g, g, c_g, thresholds)
        cutoff_df.to_csv(OUT_DIR / f"{cfg.name}_internal_cutoff_selection.csv", index=False)
        detail = pd.DataFrame(evaluate_cutoff(cfg.name, gate_g, gate_s, cutoff, g, s, c_g, c_s, thresholds))
        detail["config"] = cfg.name
        detail["cnn_cutoff"] = cutoff
        auc_df = make_auc_rows(cfg.name, g, s, clinical_oof, clinical_ext, gate_g, gate_s)
        auc_df["config"] = cfg.name
        branch_df = branch_summary(cfg.name, branch_g, branch_s, g, s)
        all_auc.append(auc_df)
        all_detail.append(detail)
        all_cutoffs.append(cutoff_df.assign(chosen_cutoff=cutoff))
        all_branch.append(branch_df)
        scores_by_config[cfg.name] = (gate_g, branch_g, gate_s, branch_s)

    auc_all = pd.concat(all_auc, ignore_index=True)
    detail_all = pd.concat(all_detail, ignore_index=True)
    cutoff_all = pd.concat(all_cutoffs, ignore_index=True)
    branch_all = pd.concat(all_branch, ignore_index=True)
    logs_all = pd.concat(all_logs, ignore_index=True)
    auc_all.to_csv(OUT_DIR / "region_cnn_auc_summary.csv", index=False)
    detail_all.to_csv(OUT_DIR / "region_cnn_deescalation_details.csv", index=False)
    cutoff_all.to_csv(OUT_DIR / "region_cnn_cutoff_selection_all.csv", index=False)
    branch_all.to_csv(OUT_DIR / "region_cnn_branch_summary.csv", index=False)
    logs_all.to_csv(OUT_DIR / "region_cnn_training_log.csv", index=False)

    # Internal-only model selection.
    summary_rows = []
    for cfg in CONFIGS:
        sub = detail_all[detail_all["config"].eq(cfg.name)]
        gi = sub[sub["dataset"].eq("g1090_internal")]
        se = sub[sub["dataset"].eq("sdata_external")]
        score = (
            2.2 * gi["specificity_gain"].min()
            + gi["specificity_gain"].mean()
            - 0.8 * gi["sensitivity_loss"].max()
            - 0.02 * gi["deesc_event_fisher_p"].max()
        )
        summary_rows.append(
            {
                "config": cfg.name,
                "internal_selection_score": float(score),
                "internal_min_p_loss": float(gi["sensitivity_loss_p_exact"].min()),
                "internal_max_sens_loss": float(gi["sensitivity_loss"].max()),
                "internal_min_spec_gain": float(gi["specificity_gain"].min()),
                "internal_mean_spec_gain": float(gi["specificity_gain"].mean()),
                "external_min_p_loss": float(se["sensitivity_loss_p_exact"].min()),
                "external_max_sens_loss": float(se["sensitivity_loss"].max()),
                "external_min_spec_gain": float(se["specificity_gain"].min()),
                "external_mean_spec_gain": float(se["specificity_gain"].mean()),
                "external_max_fisher_p": float(se["deesc_event_fisher_p"].max()),
            }
        )
    model_summary = pd.DataFrame(summary_rows).sort_values("internal_selection_score", ascending=False)
    model_summary.to_csv(OUT_DIR / "region_cnn_model_selection_summary.csv", index=False)
    best_config = str(model_summary.iloc[0]["config"])
    best_detail = detail_all[detail_all["config"].eq(best_config)].copy()
    best_branch = branch_all[branch_all["config"].eq(best_config)].copy()
    best_detail.to_csv(OUT_DIR / "region_cnn_best_deescalation_details.csv", index=False)
    best_branch.to_csv(OUT_DIR / "region_cnn_best_branch_summary.csv", index=False)
    plot_details(best_detail, best_branch, OUT_DIR / "region_cnn_best_plot.png")

    with (OUT_DIR / "region_cnn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(DEVICE),
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization, row-z curve/slope/curvature channels",
                "regions_1_indexed_inclusive": REGIONS,
                "gate": "Differentiable P(at least 2 of 4 branch votes); cutoff selected on g1090 OOF only and applied to sdata.",
                "configs": [cfg.__dict__ for cfg in CONFIGS],
                "best_config_by_internal_only": best_config,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nMODEL SUMMARY")
    print(model_summary.to_string(index=False))
    print("\nAUC SUMMARY")
    print(auc_all.to_string(index=False))
    print("\nBEST DE-ESCALATION")
    show = [
        "dataset",
        "operating_point",
        "clinical_sensitivity",
        "post_sensitivity",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "clinical_specificity",
        "post_specificity",
        "specificity_gain",
        "specificity_gain_p_exact",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
        "cnn_cutoff",
    ]
    print(best_detail[show].to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    main()
