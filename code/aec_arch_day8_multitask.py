from __future__ import annotations

# Day8 — Multi-task 공유 인코더. curve/clinic 인코더 하나를 HTN+DM+CKD 3개 질환이 공유하고
# task-specific head만 분리 — 각 라벨의 양성 수가 적어(CKD n_pos=105) supervised 신호가 부족한 문제를
# 다른 두 질환의 보조 supervision으로 정규화·표현학습을 강화하는 시도. 지금까지 모든 arm은 질환마다
# 독립적으로 학습했고, 이 arm만 하나의 공유 모델이 3개 질환을 동시에 예측한다. docs/aec_architecture_rotation_plan.md 참고.

import copy
import itertools

from aec_fusion_common import (
    AEC_COLS, BACKBONE, DEVICE, EARLY_STOP_PATIENCE, ENSEMBLE_SIZE, EPOCHS, EXTERNAL_XLSX, FEATURES,
    INTERNAL_XLSX, N_FOLDS, PROJECT_ROOT, SEED, VAL_FRACTION, ClinicEncoderMLP, CurveEncoderGAP,
    bootstrap_auc_ci, clinical_matrix, delong_paired_auc_test, load_cohort, plot_auc_grouped, plot_loss_curve, prepare_curve,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arch_day8_multitask"
FUSION_NAME = "multitask"
CHANNELS = BACKBONE["channels"]
KERNEL_SIZES = BACKBONE["kernel_sizes"]
FEATURE_LIST = list(FEATURES.keys())  # ["HTN", "DM", "CKD"]
SEARCH_SPACE = {"embed_dim": [8, 16], "head_hidden": [16, 32]}
N_REFINE_TOP_K = 3


class MultiTaskModel(nn.Module):
    def __init__(self, n_clinic: int, embed_dim: int, head_hidden: int, dropout: float):
        super().__init__()
        self.curve_encoder = CurveEncoderGAP(CHANNELS, KERNEL_SIZES, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.heads = nn.ModuleDict({
            f: nn.Sequential(nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
            for f in FEATURE_LIST
        })

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.curve_encoder(curve), self.clinic_encoder(clinic)], dim=1)
        return torch.cat([self.heads[f](z) for f in FEATURE_LIST], dim=1)  # (B, 3) 열순서=FEATURE_LIST


def train_multitask(curve: np.ndarray, clinic: np.ndarray, y_multi: np.ndarray, n_clinic: int,
                     config: dict, seed: int, history: list[dict] | None = None) -> nn.Module:
    torch.manual_seed(seed)
    strat_key = y_multi[:, 0] * 4 + y_multi[:, 1] * 2 + y_multi[:, 2]
    tr_idx, val_idx = train_test_split(np.arange(len(y_multi)), test_size=VAL_FRACTION, random_state=seed, stratify=strat_key)

    model = MultiTaskModel(n_clinic, config["embed_dim"], config["head_hidden"], BACKBONE["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"])
    pos_weights = []
    for j in range(3):
        n_pos = float(y_multi[tr_idx, j].sum())
        n_neg = float(len(tr_idx)) - n_pos
        pos_weights.append(n_neg / max(n_pos, 1.0))
    pos_weight_t = torch.tensor(pos_weights, device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    curve_t = torch.tensor(curve[tr_idx], dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_multi[tr_idx], dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y_multi[val_idx]
    curve_std = curve_t.std()
    augment_std = BACKBONE["augment_std"]
    batch_size = BACKBONE["batch_size"]

    n = curve_t.shape[0]
    best_mean_auc, best_state, epochs_no_improve = -np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_curve = curve_t[idx]
            if augment_std > 0:
                batch_curve = batch_curve + torch.randn_like(batch_curve) * curve_std * augment_std
            opt.zero_grad()
            loss = bce(model(batch_curve, clinic_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(curve_val_t, clinic_val_t)).cpu().numpy()
        aucs = [roc_auc_score(y_val[:, j], val_pred[:, j]) for j in range(3) if len(np.unique(y_val[:, j])) > 1]
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        if history is not None:
            history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_auc": mean_auc})
        if mean_auc > best_mean_auc:
            best_mean_auc, best_state, epochs_no_improve = mean_auc, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_multitask(model: nn.Module, curve: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t, clinic_t)).cpu().numpy()


def evaluate_config(curve_raw: np.ndarray, clinic: np.ndarray, y_multi: np.ndarray, folds: list,
                     config: dict, n_clinic: int, ensemble_size: int) -> tuple[np.ndarray, np.ndarray]:
    curve = prepare_curve(curve_raw, BACKBONE["curve_prep"])
    oof = np.full((len(y_multi), 3), np.nan)
    for tr_idx, va_idx in folds:
        seed_preds = [
            predict_multitask(train_multitask(curve[tr_idx], clinic[tr_idx], y_multi[tr_idx], n_clinic, config, SEED + s), curve[va_idx], clinic[va_idx])
            for s in range(ensemble_size)
        ]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    aucs = np.array([roc_auc_score(y_multi[:, j], oof[:, j]) for j in range(3)])
    return aucs, oof


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE} fusion={FUSION_NAME}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    clinic_int, scaler = clinical_matrix(meta_int, None)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)
    n_clinic = clinic_int.shape[1]
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))

    y_int_multi = np.column_stack([meta_int[f].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) for f in FEATURE_LIST])
    y_ext_multi = np.column_stack([meta_ext[f].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) for f in FEATURE_LIST])

    clinic4_auc_int = {}
    clinic4_oof = {}
    for j, feature in enumerate(FEATURE_LIST):
        oof = np.full(len(y_int_multi), np.nan)
        for tr_idx, va_idx in folds:
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int_multi[tr_idx, j])
            oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        clinic4_oof[feature] = oof
        clinic4_auc_int[feature] = float(roc_auc_score(y_int_multi[:, j], oof))
        print(f"[baseline/{feature}] clinic4 internal OOF AUC={clinic4_auc_int[feature]:.4f}")

    keys = list(SEARCH_SPACE.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(SEARCH_SPACE[k] for k in keys))]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1 = []
    for i, combo in enumerate(combos):
        aucs, _ = evaluate_config(curve_int_raw, clinic_int, y_int_multi, folds, combo, n_clinic, ensemble_size=1)
        mean_delta = float(np.mean([aucs[j] - clinic4_auc_int[f] for j, f in enumerate(FEATURE_LIST)]))
        print(f"[{FUSION_NAME} stage1 {i+1}/{len(combos)}] {combo} aucs(HTN/DM/CKD)={aucs.round(4)} mean_delta={mean_delta:+.4f}")
        stage1.append({"combo": combo, "aucs": aucs, "mean_delta": mean_delta})

    ranked = sorted(stage1, key=lambda r: r["mean_delta"], reverse=True)[:N_REFINE_TOP_K]
    stage2, best = [], None
    for r in ranked:
        combo = r["combo"]
        aucs, oof = evaluate_config(curve_int_raw, clinic_int, y_int_multi, folds, combo, n_clinic, ensemble_size=ENSEMBLE_SIZE)
        mean_delta = float(np.mean([aucs[j] - clinic4_auc_int[f] for j, f in enumerate(FEATURE_LIST)]))
        print(f"[{FUSION_NAME} stage2-refine] {combo} aucs(HTN/DM/CKD)={aucs.round(4)} mean_delta={mean_delta:+.4f}")
        stage2.append({"combo": combo, "aucs": aucs, "mean_delta": mean_delta, "oof": oof})
        if best is None or mean_delta > best["mean_delta"]:
            best = {"combo": combo, "aucs": aucs, "mean_delta": mean_delta, "oof": oof}
    print(f"[{FUSION_NAME} 선택] {best['combo']} mean_delta={best['mean_delta']:+.4f}")

    curve_int_final = prepare_curve(curve_int_raw, BACKBONE["curve_prep"])
    curve_ext_final = prepare_curve(curve_ext_raw, BACKBONE["curve_prep"])
    loss_histories, ext_preds = [], []
    for s in range(ENSEMBLE_SIZE):
        hist: list[dict] = []
        model = train_multitask(curve_int_final, clinic_int, y_int_multi, n_clinic, best["combo"], SEED + s, history=hist)
        loss_histories.append(hist)
        ext_preds.append(predict_multitask(model, curve_ext_final, clinic_ext))
    model_ext_pred = np.mean(ext_preds, axis=0)  # (n_ext, 3)
    plot_loss_curve(loss_histories, OUTPUT_DIR / "loss_curve_multitask.png", f"{FUSION_NAME} 최종모델 학습곡선(3질환 공유, val_auc=mean)")

    summary_rows, delong_rows = [], []
    for j, feature in enumerate(FEATURE_LIST):
        clf = LogisticRegression(max_iter=2000).fit(clinic_int, y_int_multi[:, j])
        clinic4_ext_pred = clf.predict_proba(clinic_ext)[:, 1]
        clinic4_auc_ext = float(roc_auc_score(y_ext_multi[:, j], clinic4_ext_pred))
        model_auc_ext = float(roc_auc_score(y_ext_multi[:, j], model_ext_pred[:, j]))
        clinic4_ci = bootstrap_auc_ci(y_ext_multi[:, j], clinic4_ext_pred)
        model_ci = bootstrap_auc_ci(y_ext_multi[:, j], model_ext_pred[:, j])
        print(f"[{FUSION_NAME}/{feature} external] clinic4 AUC={clinic4_auc_ext:.4f} / {FUSION_NAME} AUC={model_auc_ext:.4f}")

        summary_rows.extend([
            {"feature": feature, "model": "clinic4", "cohort": "internal", "n": int(len(y_int_multi)), "n_pos": int(y_int_multi[:, j].sum()),
             "auc": clinic4_auc_int[feature], "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": FUSION_NAME, "cohort": "internal", "n": int(len(y_int_multi)), "n_pos": int(y_int_multi[:, j].sum()),
             "auc": best["aucs"][j], "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": "clinic4", "cohort": "external", "n": int(len(y_ext_multi)), "n_pos": int(y_ext_multi[:, j].sum()),
             "auc": clinic4_auc_ext, "auc_ci_lower": clinic4_ci[0], "auc_ci_upper": clinic4_ci[1]},
            {"feature": feature, "model": FUSION_NAME, "cohort": "external", "n": int(len(y_ext_multi)), "n_pos": int(y_ext_multi[:, j].sum()),
             "auc": model_auc_ext, "auc_ci_lower": model_ci[0], "auc_ci_upper": model_ci[1]},
        ])
        for cohort, y, score_a, score_b in (
            ("internal", y_int_multi[:, j], clinic4_oof[feature], best["oof"][:, j]),
            ("external", y_ext_multi[:, j], clinic4_ext_pred, model_ext_pred[:, j]),
        ):
            res = delong_paired_auc_test(y, score_a, score_b)
            print(f"[{FUSION_NAME} vs clinic4 / {feature} / {cohort}] delta_auc={res['diff']:+.4f} p={res['p_value']:.4f}")
            delong_rows.append({"feature": feature, "comparison": f"{FUSION_NAME}_minus_clinic4", "cohort": cohort, **res})

    stage1_rows = [{**r["combo"], "auc_HTN": r["aucs"][0], "auc_DM": r["aucs"][1], "auc_CKD": r["aucs"][2], "mean_delta": r["mean_delta"]} for r in stage1]
    stage2_rows = [{**r["combo"], "auc_HTN": r["aucs"][0], "auc_DM": r["aucs"][1], "auc_CKD": r["aucs"][2], "mean_delta": r["mean_delta"]} for r in stage2]
    pd.DataFrame(stage1_rows).to_csv(OUTPUT_DIR / "search_stage1_grid.csv", index=False)
    pd.DataFrame(stage2_rows).to_csv(OUTPUT_DIR / "search_stage2_refine.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(OUTPUT_DIR / "delong_vs_clinic4.csv", index=False)
    plot_auc_grouped(summary, OUTPUT_DIR / "classification_auc_comparison.png",
                      model_order=["clinic4", FUSION_NAME], colors={"clinic4": "#6b6a66", FUSION_NAME: "#2a78d6"},
                      title=f"AUC 비교 (clinic4 vs {FUSION_NAME})")
    print(f"[{FUSION_NAME}] 결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
