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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    auc_with_p,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
    make_single_deesc,
)
from aec_region_constrained_cnn_gate import (  # noqa: E402
    DEVICE,
    REGIONS,
    RegionBranch,
    make_channels,
    standardize_channels_train_apply,
    stratified_folds,
)
from aec_region_cnn_teacher_mimic_gate import (  # noqa: E402
    BRANCH_LAMBDA,
    BRANCH_WIDTH,
    locked_targets,
)


OUT_DIR = Path(r"C:\Users\user\OneDrive\Dokumen\radiation\outputs\aec_region_cnn_direct_vote_gate")
SEEDS = [20260701, 20260711]
THRESHOLDS_TO_REPORT = np.round(np.arange(0.35, 0.86, 0.05), 2).tolist()


@dataclass
class VoteConfig:
    name: str
    dropout: float = 0.20
    lr: float = 8.0e-4
    weight_decay: float = 1.0e-3
    consensus_weight: float = 0.65
    non_cpos_weight: float = 0.05
    max_epochs: int = 180
    patience: int = 20
    batch_size: int = 96


CONFIGS = [
    VoteConfig("direct_vote_balanced", dropout=0.20, lr=8.0e-4, weight_decay=1.0e-3, consensus_weight=0.65, non_cpos_weight=0.05),
    VoteConfig("direct_vote_guarded", dropout=0.30, lr=6.0e-4, weight_decay=2.0e-3, consensus_weight=0.85, non_cpos_weight=0.03),
]


def soft_atleast2_prob(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    q = 1.0 - p
    p0 = torch.prod(q, dim=-1)
    p1 = torch.zeros_like(p0)
    for j in range(p.shape[-1]):
        p1 = p1 + p[..., j] * torch.prod(torch.cat([q[..., :j], q[..., j + 1 :]], dim=-1), dim=-1)
    return torch.clamp(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)


class DirectVoteCnn(torch.nn.Module):
    def __init__(self, thresholds: np.ndarray, dropout: float) -> None:
        super().__init__()
        self.regions = list(REGIONS.items())
        self.branches = torch.nn.ModuleList([RegionBranch(dropout=dropout) for _ in self.regions])
        self.head_weight = torch.nn.Parameter(torch.zeros(len(REGIONS), 5))
        self.head_bias = torch.nn.Parameter(torch.zeros(len(REGIONS)))
        # Start near the analytic rule: branch morphology affects vote mostly near the boundary.
        with torch.no_grad():
            self.head_weight[:, 1] = -1.5
            self.head_weight[:, 2] = -2.0
            self.head_weight[:, 4] = 0.5
            self.head_bias[:] = -1.0
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.register_buffer("width", torch.tensor(BRANCH_WIDTH, dtype=torch.float32))

    def forward(self, x: torch.Tensor, clinical_z: torch.Tensor) -> torch.Tensor:
        branch_score = []
        for branch, (_, (start, end)) in zip(self.branches, self.regions):
            branch_score.append(branch(x[:, :, start - 1 : end]))
        morph = torch.stack(branch_score, dim=-1)  # N x 4
        delta = clinical_z[:, None] - self.thresholds[None, :]  # N x O
        boundary = torch.exp(-0.5 * (delta[:, :, None] / self.width[None, None, :]) ** 2)
        cpos = (delta >= 0).float()[:, :, None]
        feats = torch.stack(
            [
                morph[:, None, :].expand(-1, len(OPS), -1),
                morph[:, None, :] * boundary,
                delta[:, :, None].expand(-1, -1, len(REGIONS)),
                boundary,
                cpos.expand(-1, -1, len(REGIONS)),
            ],
            dim=-1,
        )
        return (feats * self.head_weight[None, None, :, :]).sum(dim=-1) + self.head_bias[None, None, :]


def exact_feature_votes(
    y: np.ndarray,
    clinical_z: np.ndarray,
    thresholds: dict[str, float],
    feature_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    votes = np.zeros((len(y), len(OPS), feature_risk.shape[1]), dtype=np.float32)
    cpos = np.zeros((len(y), len(OPS)), dtype=bool)
    for op_idx, (op, _) in enumerate(OPS):
        th = thresholds[op]
        cpos[:, op_idx] = clinical_z >= th
        for j in range(feature_risk.shape[1]):
            votes[:, op_idx, j] = make_single_deesc(
                clinical_z,
                feature_risk[:, j],
                th,
                float(BRANCH_WIDTH[j]),
                float(BRANCH_LAMBDA[j]),
            ).astype(np.float32)
    return votes, cpos


def loss_fn(logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, pos_weight: torch.Tensor, cfg: VoteConfig) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = sample_weight[:, :, None] * (1.0 + (pos_weight[None, None, :] - 1.0) * target)
    branch_loss = (bce * weight).sum() / torch.clamp(weight.sum(), min=1.0)

    prob2 = soft_atleast2_prob(logits)
    target2 = (target.sum(dim=-1) >= 2).float()
    w2 = sample_weight
    consensus_loss = (F.binary_cross_entropy(prob2, target2, reduction="none") * w2).sum() / torch.clamp(w2.sum(), min=1.0)
    return branch_loss + cfg.consensus_weight * consensus_loss


def train_one_fold(
    cfg: VoteConfig,
    x: np.ndarray,
    clinical_z: np.ndarray,
    target: np.ndarray,
    cpos: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    clinical_ext: np.ndarray,
    threshold_vec: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    model = DirectVoteCnn(threshold_vec, cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(clinical_z[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(target[train_idx], dtype=torch.float32, device=DEVICE)
    wt_np = np.where(cpos[train_idx], 1.0, cfg.non_cpos_weight).astype(np.float32)
    wt = torch.tensor(wt_np, dtype=torch.float32, device=DEVICE)

    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    cv = torch.tensor(clinical_z[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(target[val_idx], dtype=torch.float32, device=DEVICE)
    wv_np = np.where(cpos[val_idx], 1.0, cfg.non_cpos_weight).astype(np.float32)
    wv = torch.tensor(wv_np, dtype=torch.float32, device=DEVICE)

    pos = (target[train_idx] * wt_np[:, :, None]).sum(axis=(0, 1))
    neg = ((1.0 - target[train_idx]) * wt_np[:, :, None]).sum(axis=(0, 1))
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 30.0).astype(np.float32)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(xt[batch], ct[batch])
            loss = loss_fn(logits, yt[batch], wt[batch], pos_weight, cfg)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv, cv), yv, wv, pos_weight, cfg).item())
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
    model.eval()
    with torch.no_grad():
        val_logits = model(xv, cv).detach().cpu().numpy()
        ext_logits = model(
            torch.tensor(x_ext, dtype=torch.float32, device=DEVICE),
            torch.tensor(clinical_ext, dtype=torch.float32, device=DEVICE),
        ).detach().cpu().numpy()
    return val_logits, ext_logits, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss), "pos_weight_mean": float(pw.mean())}


def crossfit_config(
    cfg: VoteConfig,
    xg: np.ndarray,
    c_g: np.ndarray,
    target_g: np.ndarray,
    cpos_g: np.ndarray,
    xs: np.ndarray,
    c_s: np.ndarray,
    y: np.ndarray,
    threshold_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    oof_runs = []
    ext_runs = []
    logs = []
    for seed in SEEDS:
        oof = np.zeros_like(target_g, dtype=float)
        ext_folds = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            val_logits, ext_logits, info = train_one_fold(
                cfg, xg, c_g, target_g, cpos_g, tr, va, xs, c_s, threshold_vec, seed + fold_id * 101
            )
            oof[va] = val_logits
            ext_folds.append(ext_logits)
            logs.append({"config": cfg.name, "seed": seed, "fold": fold_id, **info})
        oof_runs.append(oof)
        ext_runs.append(np.mean(ext_folds, axis=0))
    return np.mean(oof_runs, axis=0), np.mean(ext_runs, axis=0), pd.DataFrame(logs)


def evaluate_deesc(
    rule: str,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    votes_g: np.ndarray,
    votes_s: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for dataset, d, cpos, votes in [
        ("g1090_internal", g, cpos_g, votes_g),
        ("sdata_external", s, cpos_s, votes_s),
    ]:
        for op_idx, (op, _) in enumerate(OPS):
            deesc = cpos[:, op_idx] & (votes[:, op_idx, :].sum(axis=1) >= 2)
            rows.append(
                deesc_metric_row(
                    dataset,
                    rule,
                    "direct_branch_votes",
                    op,
                    d["y"].astype(int),
                    cpos[:, op_idx],
                    deesc,
                )
            )
    return pd.DataFrame(rows)


def vote_agreement(
    rule: str,
    target_g: np.ndarray,
    target_s: np.ndarray,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    pred_g: np.ndarray,
    pred_s: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for dataset, target, cpos, pred in [
        ("g1090_internal", target_g, cpos_g, pred_g),
        ("sdata_external", target_s, cpos_s, pred_s),
    ]:
        for op_idx, (op, _) in enumerate(OPS):
            mask = cpos[:, op_idx]
            exact_cons = target[:, op_idx, :].sum(axis=1) >= 2
            pred_cons = pred[:, op_idx, :].sum(axis=1) >= 2
            if mask.any():
                agreement = float((exact_cons[mask] == pred_cons[mask]).mean())
                exact_deesc = exact_cons[mask]
                pred_deesc = pred_cons[mask]
                recall = float((pred_deesc & exact_deesc).sum() / max(exact_deesc.sum(), 1))
                precision = float((pred_deesc & exact_deesc).sum() / max(pred_deesc.sum(), 1))
            else:
                agreement = recall = precision = np.nan
            row = {
                "rule": rule,
                "dataset": dataset,
                "operating_point": op,
                "consensus_agreement_cpos": agreement,
                "consensus_recall_vs_exact": recall,
                "consensus_precision_vs_exact": precision,
                "exact_deesc_n": int(exact_cons[mask].sum()) if mask.any() else 0,
                "pred_deesc_n": int(pred_cons[mask].sum()) if mask.any() else 0,
            }
            for j, region in enumerate(REGIONS):
                if mask.any():
                    row[f"{region}_branch_agreement_cpos"] = float((target[mask, op_idx, j] == pred[mask, op_idx, j]).mean())
                else:
                    row[f"{region}_branch_agreement_cpos"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def clinical_plus_auc(
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    vote_prob_g: np.ndarray,
    vote_prob_s: np.ndarray,
) -> tuple[float, float, float, float]:
    risk_g = -soft_atleast2_np(vote_prob_g).mean(axis=1)
    risk_s = -soft_atleast2_np(vote_prob_s).mean(axis=1)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260701)
    oof = np.zeros(len(g["y"]), dtype=float)
    for fold_id, (tr, va) in enumerate(folds.split(np.zeros(len(g["y"])), g["y"])):
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260701 + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], risk_g[tr]]), g["y"][tr])
        oof[va] = model.decision_function(np.column_stack([clinical_oof[va], risk_g[va]]))
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260799)
    model.fit(np.column_stack([clinical_oof, risk_g]), g["y"])
    ext = model.decision_function(np.column_stack([clinical_ext, risk_s]))
    ig_auc, ig_p = auc_with_p(g["y"], oof)
    es_auc, es_p = auc_with_p(s["y"], ext)
    return ig_auc, ig_p, es_auc, es_p


def soft_atleast2_np(prob: np.ndarray) -> np.ndarray:
    q = 1.0 - prob
    p0 = np.prod(q, axis=-1)
    p1 = np.zeros_like(p0)
    for j in range(prob.shape[-1]):
        p1 += prob[..., j] * np.prod(np.concatenate([q[..., :j], q[..., j + 1 :]], axis=-1), axis=-1)
    return np.clip(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)


def auc_table(
    config_name: str,
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    prob_g: np.ndarray,
    prob_s: np.ndarray,
) -> pd.DataFrame:
    cg_auc, cg_p = auc_with_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_with_p(s["y"], clinical_ext)
    low_risk_g = soft_atleast2_np(prob_g).mean(axis=1)
    low_risk_s = soft_atleast2_np(prob_s).mean(axis=1)
    ag_auc, ag_p = auc_with_p(g["y"], -low_risk_g)
    as_auc, as_p = auc_with_p(s["y"], -low_risk_s)
    pg_auc, pg_p, ps_auc, ps_p = clinical_plus_auc(g, s, clinical_oof, clinical_ext, prob_g, prob_s)
    return pd.DataFrame(
        [
            {
                "config": config_name,
                "model": "clinical_only",
                "internal_auc": cg_auc,
                "internal_auc_p": cg_p,
                "external_auc": cs_auc,
                "external_auc_p": cs_p,
                "internal_delta_vs_clinical": 0.0,
                "external_delta_vs_clinical": 0.0,
            },
            {
                "config": config_name,
                "model": "direct_vote_cnn_lowrisk_score",
                "internal_auc": ag_auc,
                "internal_auc_p": ag_p,
                "external_auc": as_auc,
                "external_auc_p": as_p,
                "internal_delta_vs_clinical": ag_auc - cg_auc,
                "external_delta_vs_clinical": as_auc - cs_auc,
            },
            {
                "config": config_name,
                "model": "clinical_plus_direct_vote_cnn_score",
                "internal_auc": pg_auc,
                "internal_auc_p": pg_p,
                "external_auc": ps_auc,
                "external_auc_p": ps_p,
                "internal_delta_vs_clinical": pg_auc - cg_auc,
                "external_delta_vs_clinical": ps_auc - cs_auc,
            },
        ]
    )


def summarize_internal(detail: pd.DataFrame) -> dict:
    gi = detail[detail["dataset"].eq("g1090_internal")]
    return {
        "internal_min_p_loss": float(gi["sensitivity_loss_p_exact"].min(skipna=True)),
        "internal_max_sens_loss": float(gi["sensitivity_loss"].max(skipna=True)),
        "internal_min_spec_gain": float(gi["specificity_gain"].min(skipna=True)),
        "internal_mean_spec_gain": float(gi["specificity_gain"].mean(skipna=True)),
        "internal_max_fisher_p": float(gi["deesc_event_fisher_p"].max(skipna=True)),
        "internal_min_deesc_n": int(gi["deesc_n"].min(skipna=True)),
        "internal_mean_event_rate": float(gi["deesc_event_rate"].mean(skipna=True)),
    }


def plot_result(best_detail: pd.DataFrame, agree: pd.DataFrame, cnn_rule: str, out_path: Path) -> None:
    labels = [op for op, _ in OPS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {"exact_locked_2of4": "#2c7fb8", cnn_rule: "#d95f02"}
    for rule in ["exact_locked_2of4", cnn_rule]:
        for dataset, ls in [("g1090_internal", "-"), ("sdata_external", "--")]:
            sub = best_detail[best_detail["rule"].eq(rule) & best_detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
            x = np.arange(len(labels))
            axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", ls=ls, color=colors[rule], label=f"{rule} {dataset}")
            axes[1].plot(x, sub["sensitivity_loss"] * 100, marker="x", ls=ls, color=colors[rule], label=f"{rule} {dataset}")
    for ax, title in [(axes[0], "Specificity gain"), (axes[1], "Sensitivity loss")]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    sub = agree[agree["rule"].eq(cnn_rule) & agree["dataset"].eq("sdata_external")].set_index("operating_point").loc[labels].reset_index()
    axes[2].plot(np.arange(len(labels)), sub["consensus_agreement_cpos"] * 100, marker="o", color="#756bb1", label="agreement")
    axes[2].plot(np.arange(len(labels)), sub["consensus_recall_vs_exact"] * 100, marker="x", color="#31a354", label="recall")
    axes[2].plot(np.arange(len(labels)), sub["consensus_precision_vs_exact"] * 100, marker="s", color="#636363", label="precision")
    axes[2].set_xticks(np.arange(len(labels)))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("% among clinical-positive")
    axes[2].set_title("External 2-of-4 mimic", loc="left", fontweight="bold")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    threshold_vec = np.array([thresholds[op] for op, _ in OPS], dtype=np.float32)

    feature_g, feature_s, features = locked_targets(g, s, c_g)
    target_g, cpos_g = exact_feature_votes(g["y"], c_g, thresholds, feature_g)
    target_s, cpos_s = exact_feature_votes(s["y"], c_s, thresholds, feature_s)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))

    exact_detail = evaluate_deesc("exact_locked_2of4", g, s, cpos_g, cpos_s, target_g.astype(bool), target_s.astype(bool))
    all_detail = [exact_detail]
    all_agree = []
    all_auc = []
    all_logs = []
    all_summary = []

    for cfg in CONFIGS:
        print(f"training {cfg.name}", flush=True)
        logits_g, logits_s, logs = crossfit_config(cfg, xg, c_g, target_g, cpos_g, xs, c_s, g["y"], threshold_vec)
        prob_g = 1.0 / (1.0 + np.exp(-logits_g))
        prob_s = 1.0 / (1.0 + np.exp(-logits_s))
        for threshold in THRESHOLDS_TO_REPORT:
            pred_g = prob_g >= threshold
            pred_s = prob_s >= threshold
            rule = f"{cfg.name}_p{threshold:.2f}".replace(".", ".")
            detail = evaluate_deesc(rule, g, s, cpos_g, cpos_s, pred_g, pred_s)
            agree = vote_agreement(rule, target_g.astype(bool), target_s.astype(bool), cpos_g, cpos_s, pred_g, pred_s)
            summary = summarize_internal(detail)
            survives = (
                summary["internal_min_p_loss"] >= 0.05
                and summary["internal_min_spec_gain"] > 0
                and summary["internal_max_fisher_p"] < 0.05
                and summary["internal_min_deesc_n"] >= 25
                and summary["internal_max_sens_loss"] <= 0.08
            )
            score = (
                3.0 * summary["internal_min_spec_gain"]
                + 1.3 * summary["internal_mean_spec_gain"]
                - 0.9 * summary["internal_max_sens_loss"]
                - 0.02 * summary["internal_max_fisher_p"]
            )
            if not survives:
                score -= 10.0
            all_detail.append(detail)
            all_agree.append(agree)
            all_summary.append(
                {
                    "config": cfg.name,
                    "prob_threshold": threshold,
                    "rule": rule,
                    "survives_internal_constraints": survives,
                    "internal_selection_score": score,
                    **summary,
                }
            )
        all_auc.append(auc_table(cfg.name, g, s, clinical_oof, clinical_ext, prob_g, prob_s))
        all_logs.append(logs)

    detail_all = pd.concat(all_detail, ignore_index=True)
    agree_all = pd.concat(all_agree, ignore_index=True)
    auc_all = pd.concat(all_auc, ignore_index=True)
    logs_all = pd.concat(all_logs, ignore_index=True)
    summary_all = pd.DataFrame(all_summary).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    best_rule = str(summary_all.iloc[0]["rule"])
    best_detail = detail_all[detail_all["rule"].isin(["exact_locked_2of4", best_rule])].copy()
    best_agree = agree_all[agree_all["rule"].eq(best_rule)].copy()

    detail_all.to_csv(OUT_DIR / "direct_vote_deescalation_details.csv", index=False)
    agree_all.to_csv(OUT_DIR / "direct_vote_agreement.csv", index=False)
    auc_all.to_csv(OUT_DIR / "direct_vote_auc_summary.csv", index=False)
    logs_all.to_csv(OUT_DIR / "direct_vote_training_log.csv", index=False)
    summary_all.to_csv(OUT_DIR / "direct_vote_model_selection_summary.csv", index=False)
    best_detail.to_csv(OUT_DIR / "direct_vote_best_deescalation_details.csv", index=False)
    best_agree.to_csv(OUT_DIR / "direct_vote_best_agreement.csv", index=False)
    plot_result(best_detail, best_agree, best_rule, OUT_DIR / "direct_vote_best_plot.png")
    with (OUT_DIR / "direct_vote_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "teacher_features": features,
                "regions_1_indexed_inclusive": REGIONS,
                "rule": "Each CNN branch is trained directly against the corresponding locked yes/no de-escalation vote for each operating point. Final gate is hard sum(branch_vote)>=2.",
                "best_rule_by_internal_only": best_rule,
            },
            f,
            indent=2,
        )

    print("\nMODEL SUMMARY")
    print(summary_all.to_string(index=False))
    print("\nAUC SUMMARY")
    print(auc_all.to_string(index=False))
    print("\nBEST DE-ESCALATION")
    show = [
        "rule",
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
    ]
    print(best_detail[show].to_string(index=False))
    print("\nBEST AGREEMENT")
    print(best_agree.to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    main()
