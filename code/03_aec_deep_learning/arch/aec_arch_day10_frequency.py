from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# reorg bootstrap: aec_fusion_common은 ../fusion/에 있음(code/03_aec_deep_learning 재편, 2026-09-03)
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "fusion"))

# Day10 — 주파수영역 특징. raw 128포인트 curve를 rFFT로 변환해 magnitude spectrum(65개 주파수 성분,
# log1p 안정화)을 소형 MLP에 넣고 clinic4와 concat한다. 시간영역 CNN/RNN/Transformer와 달리 curve의
# 주기성·변조 패턴(CT tube-current z축 반복 구조)을 직접 특징화 — z-score 정규화는 스펙트럼 의미를
# 해치므로 적용하지 않고 raw curve만 사용. docs/aec_architecture_rotation_plan.md 참고.

import copy
import itertools

from aec_fusion_common import (
    AEC_COLS, BACKBONE, DEVICE, EARLY_STOP_PATIENCE, ENSEMBLE_SIZE, EPOCHS, EXTERNAL_XLSX, FEATURES,
    INTERNAL_XLSX, N_FOLDS, PROJECT_ROOT, SEED, VAL_FRACTION, ClinicEncoderMLP,
    bootstrap_auc_ci, clinical_matrix, delong_paired_auc_test, load_cohort, plot_auc_grouped, plot_loss_curve,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "arch" / "day10_frequency"
FUSION_NAME = "frequency"
N_FREQ = 65  # rfft(128) -> 65개 주파수 성분
SEARCH_SPACE = {"freq_hidden": [16, 32], "head_hidden": [16, 32]}
N_REFINE_TOP_K = 3


def curve_to_freq(curve_raw: np.ndarray) -> np.ndarray:
    fft = np.fft.rfft(curve_raw, axis=1)
    return np.log1p(np.abs(fft)).astype(np.float32)  # (n, 65)


class FreqEncoderMLP(nn.Module):
    def __init__(self, freq_hidden: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FREQ, freq_hidden), nn.ReLU(), nn.Linear(freq_hidden, embed_dim), nn.ReLU())

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        return self.net(freq)


class ConcatFusionFreq(nn.Module):
    def __init__(self, n_clinic: int, freq_hidden: int, embed_dim: int, head_hidden: int, dropout: float):
        super().__init__()
        self.freq_encoder = FreqEncoderMLP(freq_hidden, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, freq: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        f = self.freq_encoder(freq)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([f, t], dim=1))


def train_model(freq: np.ndarray, clinic: np.ndarray, y_raw: np.ndarray, n_clinic: int, config: dict, seed: int,
                 history: list[dict] | None = None) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y_raw)), test_size=VAL_FRACTION, random_state=seed, stratify=y_raw)

    model = ConcatFusionFreq(n_clinic, config["freq_hidden"], BACKBONE["embed_dim"], config["head_hidden"], BACKBONE["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"])
    n_pos = float(y_raw[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    freq_t = torch.tensor(freq[tr_idx], dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_raw[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    freq_val_t = torch.tensor(freq[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y_raw[val_idx]
    batch_size = BACKBONE["batch_size"]  # 주파수 성분은 augmentation(시간영역 noise) 미적용

    n = freq_t.shape[0]
    best_auc, best_state, epochs_no_improve = -np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = bce(model(freq_t[idx], clinic_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(freq_val_t, clinic_val_t)).cpu().numpy().ravel()
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


def predict_model(model: nn.Module, freq: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        freq_t = torch.tensor(freq, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(freq_t, clinic_t)).cpu().numpy().ravel()


def evaluate_config(freq: np.ndarray, clinic: np.ndarray, y: np.ndarray, folds: list, n_clinic: int, config: dict, ensemble_size: int) -> tuple[float, np.ndarray]:
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        seed_preds = [predict_model(train_model(freq[tr_idx], clinic[tr_idx], y[tr_idx], n_clinic, config, SEED + s), freq[va_idx], clinic[va_idx]) for s in range(ensemble_size)]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return float(roc_auc_score(y, oof)), oof


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE} fusion={FUSION_NAME}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    freq_int, freq_ext = curve_to_freq(curve_int_raw), curve_to_freq(curve_ext_raw)
    clinic_int, scaler = clinical_matrix(meta_int, None)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)
    n_clinic = clinic_int.shape[1]
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(freq_int))

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
            auc, _ = evaluate_config(freq_int, clinic_int, y_int, folds, n_clinic, combo, ensemble_size=1)
            delta = auc - clinic4_auc_int
            print(f"[{FUSION_NAME}/{feature} stage1 {i+1}/{len(combos)}] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
            stage1.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
        all_stage1.extend(stage1)

        ranked = sorted(stage1, key=lambda r: r["auc"], reverse=True)[:N_REFINE_TOP_K]
        best, stage2 = None, []
        for r in ranked:
            combo = r["combo"]
            auc, oof = evaluate_config(freq_int, clinic_int, y_int, folds, n_clinic, combo, ensemble_size=ENSEMBLE_SIZE)
            delta = auc - clinic4_auc_int
            print(f"[{FUSION_NAME}/{feature} stage2-refine] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
            stage2.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
            if best is None or auc > best["auc"]:
                best = {"combo": combo, "auc": auc, "oof": oof}
        all_stage2.extend(stage2)
        print(f"[{FUSION_NAME}/{feature} 선택] {best['combo']} internal_OOF_AUC={best['auc']:.4f} (delta={best['auc']-clinic4_auc_int:+.4f})")

        loss_histories, ext_preds = [], []
        for s in range(ENSEMBLE_SIZE):
            hist: list[dict] = []
            m = train_model(freq_int, clinic_int, y_int, n_clinic, best["combo"], SEED + s, history=hist)
            loss_histories.append(hist)
            ext_preds.append(predict_model(m, freq_ext, clinic_ext))
        model_ext_pred = np.mean(ext_preds, axis=0)
        plot_loss_curve(loss_histories, OUTPUT_DIR / f"loss_curve_{feature}.png", f"{FUSION_NAME} / {feature} 최종모델 학습곡선")
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
