from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# reorg bootstrap: aec_fusion_common은 ../fusion/에 있음(code/03_aec_deep_learning 재편, 2026-09-03)
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "fusion"))

# U-Net(1D CNN), clinic4 미사용 — AEC-128 raw curve만 입력으로 HTN/DM/CKD를 예측한다
# (https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch, VSB Power Line 대회에서
# 1D 신호를 U-Net 인코더-디코더로 통과시켜 분류에 쓴 노트북 — segmentation이 아니라 encoder-decoder를
# feature extractor로 쓰는 구조를 그대로 차용). 10일 로테이션(day1~10, [[project_aec_10day_architecture_rotation_result]])은
# 전부 clinic4+AEC fusion이었고 전부 목표 미달로 결론났지만, 이번 요청은 clinic4를 아예 빼고 AEC-128
# 단독으로 U-Net이 얼마나 예측력을 갖는지를 별도로 확인하는 것이라 fusion 프레임과 다르다. clinic4
# logistic baseline은 비교 기준으로만 남긴다(모델 입력에는 안 들어감).
#
# Encoder: ConvBlock(conv-BN-ReLU x2) -> MaxPool1d(2)를 depth번 반복, 채널은 base_width*2^level로 배증.
# Bottleneck: 마지막 pooled feature에 ConvBlock 1개.
# Decoder: ConvTranspose1d(kernel=2,stride=2)로 업샘플 -> 같은 레벨 encoder skip과 concat -> ConvBlock.
# 128=2^7이라 depth<=4에서는 pooling/upsampling 모두 나머지 없이 정확히 맞아떨어져 크롭 불필요.
# 최종 decoder 출력(B, base_width, 128)을 GAP+FC로 풀어 단일 로짓을 낸다.
#
# 평가 프로토콜은 나머지 아키텍처 실험과 동일: internal 5-fold CV로만 그리드 탐색·모델 선택하고
# external은 확정 설정으로 1회만 동결 평가([[feedback_internal_external_validation_discipline]]).
# curve 전처리는 BACKBONE 고정값(raw 대비 patient_zscore 등, [[feedback_aec_preprocessing_methods]])을 그대로 쓴다.

import copy
import itertools

from aec_fusion_common import (
    AEC_COLS, BACKBONE, DEVICE, EARLY_STOP_PATIENCE, ENSEMBLE_SIZE, EPOCHS, EXTERNAL_XLSX, FEATURES,
    INTERNAL_XLSX, N_FOLDS, PROJECT_ROOT, SEED, VAL_FRACTION,
    bootstrap_auc_ci, clinical_matrix, delong_paired_auc_test, load_cohort, plot_auc_grouped, plot_loss_curve, prepare_curve,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "arch" / "unet1d_curveonly"
MODEL_NAME = "unet1d_aeconly"
HEAD_HIDDEN = 32  # 다른 arch 실험들과 동일 고정값
N_REFINE_TOP_K = 3
SEARCH_SPACE = {
    "depth": [3, 4],
    "base_width": [16, 32],
    "kernel_size": [3, 5],
}


# conv-BN-ReLU x2, 두 번째 conv 뒤에 dropout(ResBlock1D와 동일 위치 관례)
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# U-Net 1D 인코더-디코더 본체(분류 head는 밖에서 붙임). 입력 (B,1,128) -> 출력 (B, base_width, 128)
class UNet1DBackbone(nn.Module):
    def __init__(self, depth: int, base_width: int, kernel_size: int, dropout: float):
        super().__init__()
        widths = [base_width * (2 ** i) for i in range(depth + 1)]
        enc_widths, bottleneck_out = widths[:-1], widths[-1]

        self.enc_blocks = nn.ModuleList()
        in_ch = 1
        for w in enc_widths:
            self.enc_blocks.append(ConvBlock1D(in_ch, w, kernel_size, dropout))
            in_ch = w
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock1D(enc_widths[-1], bottleneck_out, kernel_size, dropout)

        rev = list(reversed(widths))  # [bottleneck_out, enc_widths[-1], ..., enc_widths[0]]
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch, out_ch = rev[i], rev[i + 1]
            self.ups.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock1D(out_ch * 2, out_ch, kernel_size, dropout))

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve.unsqueeze(1)  # (B, 1, 128)
        skips = []
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i, (up, block) in enumerate(zip(self.ups, self.dec_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x  # (B, base_width, 128)


class UNet1DClassifier(nn.Module):
    def __init__(self, depth: int, base_width: int, kernel_size: int, dropout: float, head_hidden: int):
        super().__init__()
        self.backbone = UNet1DBackbone(depth, base_width, kernel_size, dropout)
        self.head = nn.Sequential(
            nn.Linear(base_width, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.backbone(curve)
        pooled = x.mean(dim=-1)  # GAP
        return self.head(pooled)


def build_model(config: dict) -> nn.Module:
    return UNet1DClassifier(
        depth=config["depth"], base_width=config["base_width"], kernel_size=config["kernel_size"],
        dropout=BACKBONE["dropout"], head_hidden=HEAD_HIDDEN,
    )


# AEC curve 단독 학습 루프(clinic 입력 없음). early stopping·augmentation은 다른 arch 실험과 동일 관례.
def train_unet(curve: np.ndarray, y_raw: np.ndarray, config: dict, seed: int, history: list[dict] | None = None) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y_raw)), test_size=VAL_FRACTION, random_state=seed, stratify=y_raw)

    model = build_model(config).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=BACKBONE["lr"], weight_decay=BACKBONE["weight_decay"])
    n_pos = float(y_raw[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve_t = torch.tensor(curve[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_raw[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve[val_idx], dtype=torch.float32, device=DEVICE)
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
            loss = bce(model(batch_curve), y_t[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(curve_val_t)).cpu().numpy().ravel()
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


def predict_unet(model: nn.Module, curve: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t)).cpu().numpy().ravel()


def evaluate_config(curve_raw: np.ndarray, y: np.ndarray, folds: list, config: dict, ensemble_size: int) -> tuple[float, np.ndarray]:
    curve = prepare_curve(curve_raw, BACKBONE["curve_prep"])
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        seed_preds = [predict_unet(train_unet(curve[tr_idx], y[tr_idx], config, SEED + s), curve[va_idx]) for s in range(ensemble_size)]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return float(roc_auc_score(y, oof)), oof


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE} model={MODEL_NAME}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    print(f"internal n={len(meta_int)}, external n={len(meta_ext)}")

    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    clinic_int, scaler = clinical_matrix(meta_int, None)  # clinic4는 baseline 비교용으로만 사용, 모델 입력 아님
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)

    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))
    keys = list(SEARCH_SPACE.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(SEARCH_SPACE[k] for k in keys))]
    print(f"[{MODEL_NAME}] grid: {len(combos)}개 조합 (backbone dropout/lr/wd/curve_prep는 BACKBONE 고정)")

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
            auc, _ = evaluate_config(curve_int_raw, y_int, folds, combo, ensemble_size=1)
            delta = auc - clinic4_auc_int
            print(f"[{MODEL_NAME}/{feature} stage1 {i + 1}/{len(combos)}] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
            stage1.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
        all_stage1.extend(stage1)

        ranked = sorted(stage1, key=lambda r: r["auc"], reverse=True)[:N_REFINE_TOP_K]
        best, stage2 = None, []
        for r in ranked:
            combo = r["combo"]
            auc, oof = evaluate_config(curve_int_raw, y_int, folds, combo, ensemble_size=ENSEMBLE_SIZE)
            delta = auc - clinic4_auc_int
            print(f"[{MODEL_NAME}/{feature} stage2-refine] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f}, ensemble={ENSEMBLE_SIZE})")
            stage2.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
            if best is None or auc > best["auc"]:
                best = {"combo": combo, "auc": auc, "oof": oof}
        all_stage2.extend(stage2)
        print(f"[{MODEL_NAME}/{feature} 선택] {best['combo']} internal_OOF_AUC={best['auc']:.4f} "
              f"(delta={best['auc'] - clinic4_auc_int:+.4f} vs clinic4)")

        # ---- external: 확정 설정으로 딱 1번만 동결 평가 ----
        curve_int_final = prepare_curve(curve_int_raw, BACKBONE["curve_prep"])
        curve_ext_final = prepare_curve(curve_ext_raw, BACKBONE["curve_prep"])
        loss_histories, ext_preds = [], []
        for s in range(ENSEMBLE_SIZE):
            hist: list[dict] = []
            model = train_unet(curve_int_final, y_int, best["combo"], SEED + s, history=hist)
            loss_histories.append(hist)
            ext_preds.append(predict_unet(model, curve_ext_final))
        model_ext_pred = np.mean(ext_preds, axis=0)
        plot_loss_curve(loss_histories, OUTPUT_DIR / f"loss_curve_{feature}.png", f"{MODEL_NAME} / {feature} 최종모델 학습곡선")

        clinic4_full = LogisticRegression(max_iter=2000).fit(clinic_int, y_int)
        clinic4_ext_pred = clinic4_full.predict_proba(clinic_ext)[:, 1]

        model_auc_ext = float(roc_auc_score(y_ext, model_ext_pred))
        clinic4_auc_ext = float(roc_auc_score(y_ext, clinic4_ext_pred))
        model_ci = bootstrap_auc_ci(y_ext, model_ext_pred)
        clinic4_ci = bootstrap_auc_ci(y_ext, clinic4_ext_pred)
        print(f"[{MODEL_NAME}/{feature} external, 1회 동결평가] clinic4 AUC={clinic4_auc_ext:.4f} "
              f"95%CI=[{clinic4_ci[0]:.4f}, {clinic4_ci[1]:.4f}] / {MODEL_NAME} AUC={model_auc_ext:.4f} "
              f"95%CI=[{model_ci[0]:.4f}, {model_ci[1]:.4f}]")

        summary_rows.extend([
            {"feature": feature, "model": "clinic4", "cohort": "internal", "n": int(len(y_int)),
             "n_pos": int(y_int.sum()), "auc": clinic4_auc_int, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": MODEL_NAME, "cohort": "internal", "n": int(len(y_int)),
             "n_pos": int(y_int.sum()), "auc": best["auc"], "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": "clinic4", "cohort": "external", "n": int(len(y_ext)),
             "n_pos": int(y_ext.sum()), "auc": clinic4_auc_ext, "auc_ci_lower": clinic4_ci[0], "auc_ci_upper": clinic4_ci[1]},
            {"feature": feature, "model": MODEL_NAME, "cohort": "external", "n": int(len(y_ext)),
             "n_pos": int(y_ext.sum()), "auc": model_auc_ext, "auc_ci_lower": model_ci[0], "auc_ci_upper": model_ci[1]},
        ])

        for cohort, y, score_a, score_b in (
            ("internal", y_int, clinic4_oof, best["oof"]),
            ("external", y_ext, clinic4_ext_pred, model_ext_pred),
        ):
            res = delong_paired_auc_test(y, score_a, score_b)
            print(f"[{MODEL_NAME} vs clinic4 / {feature} / {cohort}] delta_auc={res['diff']:+.4f} z={res['z']:.4f} p={res['p_value']:.4f}")
            delong_rows.append({"feature": feature, "comparison": f"{MODEL_NAME}_minus_clinic4", "cohort": cohort, **res})

    pd.DataFrame([{**r["combo"], "feature": r["feature"], "internal_oof_auc": r["auc"], "delta_vs_clinic4": r["delta"]} for r in all_stage1]).to_csv(
        OUTPUT_DIR / "search_stage1_grid.csv", index=False,
    )
    pd.DataFrame([{**r["combo"], "feature": r["feature"], "internal_oof_auc": r["auc"], "delta_vs_clinic4": r["delta"]} for r in all_stage2]).to_csv(
        OUTPUT_DIR / "search_stage2_refine.csv", index=False,
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(OUTPUT_DIR / "delong_vs_clinic4.csv", index=False)
    plot_auc_grouped(
        summary, OUTPUT_DIR / "classification_auc_comparison.png",
        model_order=["clinic4", MODEL_NAME], colors={"clinic4": "#6b6a66", MODEL_NAME: "#2a78d6"},
        title=f"AUC 비교 (clinic4 vs {MODEL_NAME}, AEC-128 단독 입력)",
    )
    print(f"[{MODEL_NAME}] 결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
