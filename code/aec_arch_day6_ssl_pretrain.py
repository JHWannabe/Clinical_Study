from __future__ import annotations

# Day6 — Self-supervised pretrain -> fine-tune. curve 인코더(CurveEncoderGAP과 동일 구조)를 라벨 없이
# autoencoder reconstruction으로 먼저 학습(각 fold의 train split 곡선만 사용, 라벨 미접촉이므로 CV
# 프로토콜과 무관하게 fold별로 독립 사전학습)한 뒤, 그 가중치로 초기화해 clinic4와 함께 지도학습
# fine-tune. 라벨 수가 적어 supervised 신호가 부족할 가능성에 대응. docs/aec_architecture_rotation_plan.md 참고.

import copy
import itertools
from pathlib import Path

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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arch_day6_ssl_pretrain"
FUSION_NAME = "ssl_pretrain"
CHANNELS = BACKBONE["channels"]
KERNEL_SIZES = BACKBONE["kernel_sizes"]
EMBED_DIM = BACKBONE["embed_dim"]
PRETRAIN_LR = 1e-3
SEARCH_SPACE = {"pretrain_epochs": [15, 30], "head_hidden": [16, 32]}
N_REFINE_TOP_K = 3


class CurveDecoder(nn.Module):
    def __init__(self, embed_dim: int, n_slices: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, n_slices))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConcatFusionCNN(nn.Module):
    def __init__(self, n_clinic: int, head_hidden: int, dropout: float):
        super().__init__()
        self.curve_encoder = CurveEncoderGAP(CHANNELS, KERNEL_SIZES, EMBED_DIM)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, EMBED_DIM)
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def pretrain_encoder(curve_train: np.ndarray, pretrain_epochs: int, seed: int) -> dict:
    torch.manual_seed(seed)
    encoder = CurveEncoderGAP(CHANNELS, KERNEL_SIZES, EMBED_DIM).to(DEVICE)
    decoder = CurveDecoder(EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=PRETRAIN_LR)
    curve_t = torch.tensor(curve_train, dtype=torch.float32, device=DEVICE)
    mse = nn.MSELoss()
    encoder.train(); decoder.train()
    for _ in range(pretrain_epochs):
        opt.zero_grad()
        loss = mse(decoder(encoder(curve_t)), curve_t)
        loss.backward()
        opt.step()
    return copy.deepcopy(encoder.state_dict())


def train_finetune(curve: np.ndarray, clinic: np.ndarray, y_raw: np.ndarray, n_clinic: int,
                    config: dict, seed: int, history: list[dict] | None = None) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y_raw)), test_size=VAL_FRACTION, random_state=seed, stratify=y_raw)

    pretrained_state = pretrain_encoder(curve[tr_idx], config["pretrain_epochs"], seed)
    model = ConcatFusionCNN(n_clinic, config["head_hidden"], BACKBONE["dropout"]).to(DEVICE)
    model.curve_encoder.load_state_dict(pretrained_state)

    opt = torch.optim.Adam(model.parameters(), lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"])
    n_pos = float(y_raw[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve_t = torch.tensor(curve[tr_idx], dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_raw[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y_raw[val_idx]
    curve_std = curve_t.std()
    augment_std = BACKBONE["augment_std"]
    batch_size = BACKBONE["batch_size"]

    n = curve_t.shape[0]
    best_auc, best_state, epochs_no_improve = -np.inf, None, 0
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
            val_pred = torch.sigmoid(model(curve_val_t, clinic_val_t)).cpu().numpy().ravel()
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else float("nan")
        if history is not None:
            history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_auc": float(val_auc)})
        if val_auc > best_auc:
            best_auc, best_state, epochs_no_improve = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_model(model: nn.Module, curve: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t, clinic_t)).cpu().numpy().ravel()


def evaluate_config(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray, folds: list,
                     config: dict, n_clinic: int, ensemble_size: int) -> tuple[float, np.ndarray]:
    curve = prepare_curve(curve_raw, BACKBONE["curve_prep"])
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        seed_preds = [
            predict_model(train_finetune(curve[tr_idx], clinic[tr_idx], y[tr_idx], n_clinic, config, SEED + s),
                           curve[va_idx], clinic[va_idx])
            for s in range(ensemble_size)
        ]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return float(roc_auc_score(y, oof)), oof


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE} fusion={FUSION_NAME}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    clinic_int, scaler = clinical_matrix(meta_int, None)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)
    n_clinic = clinic_int.shape[1]
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))

    keys = list(SEARCH_SPACE.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(SEARCH_SPACE[k] for k in keys))]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stage1, all_stage2, summary_rows, delong_rows = [], [], [], []
    for feature in FEATURES:
        y_int = meta_int[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y_ext = meta_ext[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        clinic4_oof = np.full(len(y_int), np.nan)
        for tr_idx, va_idx in folds:
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int[tr_idx])
            clinic4_oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        clinic4_auc_int = float(roc_auc_score(y_int, clinic4_oof))
        print(f"[baseline/{feature}] clinic4 internal OOF AUC={clinic4_auc_int:.4f}")

        stage1 = []
        for i, combo in enumerate(combos):
            auc, _ = evaluate_config(curve_int_raw, clinic_int, y_int, folds, combo, n_clinic, ensemble_size=1)
            delta = auc - clinic4_auc_int
            print(f"[{FUSION_NAME}/{feature} stage1 {i+1}/{len(combos)}] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
            stage1.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
        all_stage1.extend(stage1)

        ranked = sorted(stage1, key=lambda r: r["auc"], reverse=True)[:N_REFINE_TOP_K]
        best = None
        stage2 = []
        for r in ranked:
            combo = r["combo"]
            auc, oof = evaluate_config(curve_int_raw, clinic_int, y_int, folds, combo, n_clinic, ensemble_size=ENSEMBLE_SIZE)
            delta = auc - clinic4_auc_int
            print(f"[{FUSION_NAME}/{feature} stage2-refine] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
            stage2.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
            if best is None or auc > best["auc"]:
                best = {"combo": combo, "auc": auc, "oof": oof}
        all_stage2.extend(stage2)
        print(f"[{FUSION_NAME}/{feature} 선택] {best['combo']} internal_OOF_AUC={best['auc']:.4f} (delta={best['auc']-clinic4_auc_int:+.4f})")

        curve_int_final = prepare_curve(curve_int_raw, BACKBONE["curve_prep"])
        curve_ext_final = prepare_curve(curve_ext_raw, BACKBONE["curve_prep"])
        loss_histories, ext_preds = [], []
        for s in range(ENSEMBLE_SIZE):
            hist: list[dict] = []
            model = train_finetune(curve_int_final, clinic_int, y_int, n_clinic, best["combo"], SEED + s, history=hist)
            loss_histories.append(hist)
            ext_preds.append(predict_model(model, curve_ext_final, clinic_ext))
        model_ext_pred = np.mean(ext_preds, axis=0)
        plot_loss_curve(loss_histories, OUTPUT_DIR / f"loss_curve_{feature}.png", f"{FUSION_NAME} / {feature} 최종모델 학습곡선(fine-tune)")
        clinic4_full = LogisticRegression(max_iter=2000).fit(clinic_int, y_int)
        clinic4_ext_pred = clinic4_full.predict_proba(clinic_ext)[:, 1]

        model_auc_ext = float(roc_auc_score(y_ext, model_ext_pred))
        clinic4_auc_ext = float(roc_auc_score(y_ext, clinic4_ext_pred))
        model_ci = bootstrap_auc_ci(y_ext, model_ext_pred)
        clinic4_ci = bootstrap_auc_ci(y_ext, clinic4_ext_pred)
        print(f"[{FUSION_NAME}/{feature} external] clinic4 AUC={clinic4_auc_ext:.4f} / {FUSION_NAME} AUC={model_auc_ext:.4f}")

        summary_rows.extend([
            {"feature": feature, "model": "clinic4", "cohort": "internal", "n": int(len(y_int)), "n_pos": int(y_int.sum()),
             "auc": clinic4_auc_int, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": FUSION_NAME, "cohort": "internal", "n": int(len(y_int)), "n_pos": int(y_int.sum()),
             "auc": best["auc"], "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": "clinic4", "cohort": "external", "n": int(len(y_ext)), "n_pos": int(y_ext.sum()),
             "auc": clinic4_auc_ext, "auc_ci_lower": clinic4_ci[0], "auc_ci_upper": clinic4_ci[1]},
            {"feature": feature, "model": FUSION_NAME, "cohort": "external", "n": int(len(y_ext)), "n_pos": int(y_ext.sum()),
             "auc": model_auc_ext, "auc_ci_lower": model_ci[0], "auc_ci_upper": model_ci[1]},
        ])
        for cohort, y, score_a, score_b in (("internal", y_int, clinic4_oof, best["oof"]), ("external", y_ext, clinic4_ext_pred, model_ext_pred)):
            res = delong_paired_auc_test(y, score_a, score_b)
            print(f"[{FUSION_NAME} vs clinic4 / {feature} / {cohort}] delta_auc={res['diff']:+.4f} p={res['p_value']:.4f}")
            delong_rows.append({"feature": feature, "comparison": f"{FUSION_NAME}_minus_clinic4", "cohort": cohort, **res})

    pd.DataFrame([{**r["combo"], "feature": r["feature"], "internal_oof_auc": r["auc"], "delta_vs_clinic4": r["delta"]} for r in all_stage1]).to_csv(OUTPUT_DIR / "search_stage1_grid.csv", index=False)
    pd.DataFrame([{**r["combo"], "feature": r["feature"], "internal_oof_auc": r["auc"], "delta_vs_clinic4": r["delta"]} for r in all_stage2]).to_csv(OUTPUT_DIR / "search_stage2_refine.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(OUTPUT_DIR / "delong_vs_clinic4.csv", index=False)
    plot_auc_grouped(summary, OUTPUT_DIR / "classification_auc_comparison.png",
                      model_order=["clinic4", FUSION_NAME], colors={"clinic4": "#6b6a66", FUSION_NAME: "#2a78d6"},
                      title=f"AUC 비교 (clinic4 vs {FUSION_NAME})")
    print(f"[{FUSION_NAME}] 결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
