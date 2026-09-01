from __future__ import annotations

# 0단계 ablation — "지금까지 본 concat/gated/attnpool의 +0.003~+0.015 delta가 진짜 곡선 신호인가,
# 아니면 신경망 자체(비선형 head+ensemble)가 로지스틱 baseline보다 원래 조금 더 잘 맞기 때문인가"를
# 분리한다. 세 arm을 동일 internal 5-fold CV·동일 head_hidden=32(depth ablation과 동일 고정값)로 비교:
#   1) clinic4 logistic regression baseline
#   2) real curve concat (실제 AEC-128을 curve encoder에 넣음)
#   3) zero curve concat  (curve 입력을 매 patient 0-벡터로 대체 — 파라미터 수·구조는 real과 완전히 동일,
#      학습 가능한 curve_encoder도 그대로 두되 입력에서 정보만 지움. 즉 "capacity는 같은데 곡선 정보만 없는" 모델)
# real이 zero보다 유의하게 높아야 "곡선 자체에 clinic4를 넘는 신호가 있다"고 말할 수 있다. zero가 이미
# clinic4 logistic보다 높다면, 지금까지의 개선분 중 상당수는 곡선이 아니라 "신경망이라서" 생긴 것이다.
# external은 건드리지 않는다([[feedback_internal_external_validation_discipline]] — 이 스크립트는 순수 진단).

from pathlib import Path

from aec_fusion_common import (
    BACKBONE, ClinicEncoderMLP, CurveEncoderGAP, ENSEMBLE_SIZE, FEATURES, INTERNAL_XLSX,
    AEC_COLS, N_FOLDS, SEED, PROJECT_ROOT, clinical_matrix, delong_paired_auc_test, evaluate_config, load_cohort,
)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "curve_ablation"
HEAD_HIDDEN = 32  # aec_fusion_concat.py stage2 우승 설정과 동일값(depth ablation과도 통일)


class ConcatFusionCNN(nn.Module):
    """aec_fusion_concat.py의 ConcatFusionCNN과 동일 구조(재정의만 함, 로직 차이 없음)."""

    def __init__(self, n_clinic: int, channels, kernel_sizes, dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderGAP(channels, kernel_sizes, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


# curve 입력을 매 forward마다 0-벡터로 덮어써 내부로 넘김 — 파라미터·구조는 real과 완전히 동일하고
# 정보만 지운다(augmentation 노이즈도 0-벡터엔 무의미하므로 자동으로 아무 효과 없음).
class ZeroCurveWrapper(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        return self.inner(torch.zeros_like(curve), clinic)


def build_real(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionCNN(
        n_clinic=n_clinic, channels=config["channels"], kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"], embed_dim=config["embed_dim"], head_hidden=HEAD_HIDDEN,
    )


def build_zero(config: dict, n_clinic: int) -> nn.Module:
    return ZeroCurveWrapper(build_real(config, n_clinic))


def main() -> None:
    meta_int = load_cohort(INTERNAL_XLSX)
    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    clinic_int, _ = clinical_matrix(meta_int, None)
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))
    cfg = {**BACKBONE}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, delong_rows = [], []
    for feature in FEATURES:
        y_int = meta_int[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        clinic4_oof = np.full(len(y_int), np.nan)
        for tr_idx, va_idx in folds:
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int[tr_idx])
            clinic4_oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        clinic4_auc = float(roc_auc_score(y_int, clinic4_oof))

        real_auc, real_oof = evaluate_config(curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=ENSEMBLE_SIZE, build_model_fn=build_real)
        zero_auc, zero_oof = evaluate_config(curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=ENSEMBLE_SIZE, build_model_fn=build_zero)

        print(f"[{feature}] clinic4={clinic4_auc:.4f}  zero_curve={zero_auc:.4f} (delta={zero_auc - clinic4_auc:+.4f})  "
              f"real_curve={real_auc:.4f} (delta={real_auc - clinic4_auc:+.4f})")

        summary_rows.extend([
            {"feature": feature, "model": "clinic4", "internal_oof_auc": clinic4_auc, "delta_vs_clinic4": 0.0},
            {"feature": feature, "model": "zero_curve", "internal_oof_auc": zero_auc, "delta_vs_clinic4": zero_auc - clinic4_auc},
            {"feature": feature, "model": "real_curve", "internal_oof_auc": real_auc, "delta_vs_clinic4": real_auc - clinic4_auc},
        ])

        for comparison, score_a, score_b in (
            ("zero_minus_clinic4", clinic4_oof, zero_oof),
            ("real_minus_clinic4", clinic4_oof, real_oof),
            ("real_minus_zero", zero_oof, real_oof),
        ):
            res = delong_paired_auc_test(y_int, score_a, score_b)
            print(f"  [{feature} DeLong {comparison}] diff={res['diff']:+.4f} z={res['z']:.4f} p={res['p_value']:.4f}")
            delong_rows.append({"feature": feature, "comparison": comparison, **res})

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "curve_ablation_summary.csv", index=False)
    pd.DataFrame(delong_rows).to_csv(OUTPUT_DIR / "curve_ablation_delong.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
