from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    OUT_DIR as COND_OUT_DIR,
    SEED,
    aec_estimator,
    binary_metrics,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    make_folds,
    oof_and_external,
    threshold_youden,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec_signal_shape"


def segment_summary(mat: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, prefix: str) -> pd.DataFrame:
    """128구간 곡선을 8개 구간으로 나눠, 두 그룹(mask_a vs mask_b)의 구간별 평균과 차이를 표로 만듦."""
    rows = []
    edges = np.linspace(0, 128, 9).astype(int)
    for i in range(8):
        a, b = edges[i], edges[i + 1]
        va = mat[mask_a, a:b].mean(axis=1)
        vb = mat[mask_b, a:b].mean(axis=1)
        rows.append(
            {
                "curve": prefix,
                "segment": i + 1,
                "position_start": int(a + 1),
                "position_end": int(b),
                "mean_aec_gate_positive": float(np.mean(va)) if len(va) else np.nan,
                "mean_aec_gate_negative": float(np.mean(vb)) if len(vb) else np.nan,
                "difference_positive_minus_negative": float(np.mean(va) - np.mean(vb)) if len(va) and len(vb) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def contiguous_blocks(indices: np.ndarray) -> list[tuple[int, int]]:
    """정렬된 위치 인덱스 배열을 연속 구간(start, end) 리스트로 묶음."""
    if len(indices) == 0:
        return []
    blocks = []
    start = prev = int(indices[0])
    for ix in indices[1:]:
        ix = int(ix)
        if ix == prev + 1:
            prev = ix
        else:
            blocks.append((start, prev))
            start = prev = ix
    blocks.append((start, prev))
    return blocks


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상 양성으로 분류된 환자 중, AEC 게이트가 갈라내는 두 그룹은
    실제 AEC 곡선 모양이 어떻게 다른가?):

    1. aec_conditional_value에서 데이터 로딩·임상모델·AEC모델·게이트 관련 함수를 그대로 가져와
       train(g1090)/test(sdata)를 로드하고, 임상 단독 모델과 AEC 단독(SVM) 모델을 학습한다.
    2. 두 점수를 z-표준화하고, 임상 점수의 Youden 임계값(clinical_th) 및 그 경계 근처 여성에게만
       AEC를 가중치로 얹는 gate_score(female_boundary 게이트)를 계산한다.
    3. 외부(sdata)에서 "임상 양성" 환자를 AEC 게이트 양성/음성으로 나누고(전체 및 여성만),
       segment_summary로 a128/crop 두 곡선을 8구간으로 나눠 그룹별 평균 차이를 계산해 CSV로 저장.
    4. 전체 train으로 AEC SVM을 다시 학습해, SelectKBest로 선택된 128+128개 위치 중 실제로 쓰인
       특징들의 SVM 계수를 뽑아 절대값 기준 정렬한 뒤 CSV로 저장 (어느 위치가 모델에 중요한지).
    5. 여성 임상양성군에서 게이트 양성 vs 음성의 위치별(raw position) 평균 차이가 큰 상위 30개를 추출.
    6. 4개 패널(a128/crop x 전체/여성) 그래프로 게이트 양성/음성 그룹의 평균 곡선 모양을 시각화해
       PNG로 저장 (곡선이 실제로 다르게 생겼는지 눈으로 확인하기 위함).
    7. 임계값, 그룹별 표본수/이벤트수, 구간별/위치별 차이, SVM 계수, 계수가 연속으로 양/음인 구간
       (coefficient blocks)을 모두 요약 JSON으로 저장하고 콘솔에 출력한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]
    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(ytr, 5)

    clinical_oof, clinical_test = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)
    aec_oof, aec_test = oof_and_external(
        lambda seed: aec_estimator(train["aec"].shape[1], seed),
        train["aec"],
        ytr,
        test["aec"],
        folds,
    )
    c_z, c_te_z, _, _ = zfit_apply(clinical_oof, clinical_test)
    a_z, a_te_z, _, _ = zfit_apply(aec_oof, aec_test)

    clinical_th = threshold_youden(ytr, clinical_oof)
    clinical_z_th = (clinical_th - np.mean(clinical_oof)) / np.std(clinical_oof)
    clinical_pred = clinical_test >= clinical_th

    female_tr = train["sex"] == "F"
    female_te = test["sex"] == "F"
    boundary_tr = np.exp(-0.5 * ((c_z - clinical_z_th) / 0.75) ** 2)
    boundary_te = np.exp(-0.5 * ((c_te_z - clinical_z_th) / 0.75) ** 2)
    female_boundary_tr = boundary_tr * female_tr
    female_boundary_te = boundary_te * female_te
    gate_score_tr = c_z + 0.25 * female_boundary_tr * a_z
    gate_score_te = c_te_z + 0.25 * female_boundary_te * a_te_z
    gate_th = threshold_youden(ytr, gate_score_tr)
    gate_pred = gate_score_te >= gate_th

    cp = clinical_pred
    cp_gate_pos = cp & gate_pred
    cp_gate_neg = cp & ~gate_pred
    cp_female = cp & female_te
    cp_female_gate_pos = cp_female & gate_pred
    cp_female_gate_neg = cp_female & ~gate_pred

    a128 = test["aec"][:, :128]
    crop = test["aec"][:, 128:]
    seg_all = pd.concat(
        [
            segment_summary(a128, cp_gate_pos, cp_gate_neg, "a128_clinical_positive"),
            segment_summary(crop, cp_gate_pos, cp_gate_neg, "crop_clinical_positive"),
            segment_summary(a128, cp_female_gate_pos, cp_female_gate_neg, "a128_female_clinical_positive"),
            segment_summary(crop, cp_female_gate_pos, cp_female_gate_neg, "crop_female_clinical_positive"),
        ],
        ignore_index=True,
    )
    seg_all["abs_difference"] = seg_all["difference_positive_minus_negative"].abs()
    seg_all.to_csv(OUT_DIR / "segment_mean_differences.csv", index=False)

    final_aec = aec_estimator(train["aec"].shape[1], SEED + 99)
    final_aec.fit(train["aec"], ytr)
    support = final_aec.named_steps["select"].get_support(indices=True)
    coefs = final_aec.named_steps["svm"].coef_[0]
    names = np.array([f"a128_pos_{i+1:03d}" for i in range(128)] + [f"crop_pos_{i+1:03d}" for i in range(128)])
    coef_df = pd.DataFrame(
        {
            "feature_index": support,
            "feature": names[support],
            "curve": np.where(support < 128, "a128", "crop"),
            "position": np.where(support < 128, support + 1, support - 128 + 1),
            "linear_svm_coef_on_scaled_feature": coefs,
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(OUT_DIR / "aec_expert_top_coefficients.csv", index=False)

    top_pos = coef_df[coef_df["linear_svm_coef_on_scaled_feature"] > 0].head(25)
    top_neg = coef_df[coef_df["linear_svm_coef_on_scaled_feature"] < 0].head(25)

    # Which raw positions show the largest clinical-positive AEC+ vs AEC- difference?
    diff_rows = []
    for curve_name, mat in [("a128", a128), ("crop", crop)]:
        if cp_female_gate_pos.sum() and cp_female_gate_neg.sum():
            d = mat[cp_female_gate_pos].mean(axis=0) - mat[cp_female_gate_neg].mean(axis=0)
            for ix in np.argsort(np.abs(d))[::-1][:30]:
                diff_rows.append(
                    {
                        "curve": curve_name,
                        "position": int(ix + 1),
                        "female_clinical_positive_gate_pos_minus_gate_neg": float(d[ix]),
                    }
                )
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT_DIR / "largest_position_differences_female_clinical_positive.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    x = np.arange(1, 129)
    panels = [
        (axes[0, 0], a128, cp_gate_pos, cp_gate_neg, "a128: clinical-positive"),
        (axes[0, 1], crop, cp_gate_pos, cp_gate_neg, "cropped: clinical-positive"),
        (axes[1, 0], a128, cp_female_gate_pos, cp_female_gate_neg, "a128: female clinical-positive"),
        (axes[1, 1], crop, cp_female_gate_pos, cp_female_gate_neg, "cropped: female clinical-positive"),
    ]
    for ax, mat, mask_pos, mask_neg, title in panels:
        ax.plot(x, mat[mask_pos].mean(axis=0), lw=2, label=f"AEC gate+ n={mask_pos.sum()}")
        ax.plot(x, mat[mask_neg].mean(axis=0), lw=2, label=f"AEC gate- n={mask_neg.sum()}")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("normalized AEC - 1")
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("resampled z-axis position")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec_gate_shape_profiles.png", dpi=180)
    plt.close(fig)

    summary = {
        "thresholds": {
            "clinical_raw_threshold": float(clinical_th),
            "clinical_z_threshold": float(clinical_z_th),
            "gate_threshold": float(gate_th),
        },
        "external_gate_metrics": binary_metrics(yte, gate_score_te, gate_th),
        "clinical_positive_counts": {
            "clinical_positive": int(cp.sum()),
            "clinical_positive_events": int(yte[cp].sum()),
            "clinical_positive_gate_positive": int(cp_gate_pos.sum()),
            "clinical_positive_gate_positive_events": int(yte[cp_gate_pos].sum()),
            "clinical_positive_gate_negative": int(cp_gate_neg.sum()),
            "clinical_positive_gate_negative_events": int(yte[cp_gate_neg].sum()),
            "female_clinical_positive": int(cp_female.sum()),
            "female_clinical_positive_gate_positive": int(cp_female_gate_pos.sum()),
            "female_clinical_positive_gate_negative": int(cp_female_gate_neg.sum()),
        },
        "largest_segment_differences": seg_all.sort_values("abs_difference", ascending=False).head(12).to_dict(orient="records"),
        "top_positive_aec_expert_coefficients": top_pos[["feature", "curve", "position", "linear_svm_coef_on_scaled_feature"]].to_dict(orient="records"),
        "top_negative_aec_expert_coefficients": top_neg[["feature", "curve", "position", "linear_svm_coef_on_scaled_feature"]].to_dict(orient="records"),
        "largest_position_differences_female_clinical_positive": diff_df.head(20).to_dict(orient="records"),
        "coefficient_positive_blocks": {
            "a128": contiguous_blocks(coef_df[(coef_df["curve"] == "a128") & (coef_df["linear_svm_coef_on_scaled_feature"] > 0)]["position"].sort_values().to_numpy()),
            "crop": contiguous_blocks(coef_df[(coef_df["curve"] == "crop") & (coef_df["linear_svm_coef_on_scaled_feature"] > 0)]["position"].sort_values().to_numpy()),
        },
        "coefficient_negative_blocks": {
            "a128": contiguous_blocks(coef_df[(coef_df["curve"] == "a128") & (coef_df["linear_svm_coef_on_scaled_feature"] < 0)]["position"].sort_values().to_numpy()),
            "crop": contiguous_blocks(coef_df[(coef_df["curve"] == "crop") & (coef_df["linear_svm_coef_on_scaled_feature"] < 0)]["position"].sort_values().to_numpy()),
        },
    }
    with open(OUT_DIR / "aec_signal_shape_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
