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
    SEED,
    aec_estimator,
    make_folds,
    matrix_from_sheet,
    oof_and_external,
    row_norm,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_svm_feature_interpretation"
MID = (63, 92)
TAIL = (112, 128)


def load_aec128(path: Path) -> dict:
    """엑셀에서 aec_128 행정규화 곡선과 저근감소증 라벨을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "norm": norm, "y": y}


def visual_score(x: np.ndarray, mid: tuple[int, int] = MID, tail: tuple[int, int] = TAIL) -> np.ndarray:
    """지정된 후반 구간 평균에서 중간 구간 평균을 빼 "후반 반등 강도" 점수를 계산."""
    m0, m1 = mid
    t0, t1 = tail
    return x[:, t0 - 1 : t1].mean(axis=1) - x[:, m0 - 1 : m1].mean(axis=1)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """두 배열의 피어슨 상관계수를 계산."""
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 계산."""
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or np.nan))


def final_svm_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """전체 train으로 AEC SVM을 학습해, 선택된 특징들의 계수를 표준화 스케일과 원본(비표준화) 스케일 두 가지로 128차원 전체에 복원."""
    model = aec_estimator(x.shape[1], SEED + 99)
    model.fit(x, y)
    scaler = model.named_steps["scaler"]
    selector = model.named_steps["select"]
    svm = model.named_steps["svm"]
    selected = selector.get_support(indices=True)
    coef_scaled_full = np.zeros(x.shape[1], dtype=float)
    coef_orig_full = np.zeros(x.shape[1], dtype=float)
    coef_scaled_full[selected] = svm.coef_.ravel()
    coef_orig_full[selected] = svm.coef_.ravel() / scaler.scale_[selected]
    return coef_scaled_full, coef_orig_full


def interval_summary(coef: np.ndarray) -> pd.DataFrame:
    """SVM 계수를 초반/중간/공백/후반 4개 구간으로 나눠 각 구간의 평균·합·양수합·음수합·절댓값합을 계산."""
    zones = {
        "early_1_31": (1, 31),
        "mid_63_92": MID,
        "gap_93_111": (93, 111),
        "tail_112_128": TAIL,
    }
    rows = []
    for name, (a, b) in zones.items():
        z = coef[a - 1 : b]
        rows.append(
            {
                "zone": name,
                "start": a,
                "end": b,
                "mean_coef": float(np.mean(z)),
                "sum_coef": float(np.sum(z)),
                "positive_weight_sum": float(np.sum(z[z > 0])),
                "negative_weight_sum": float(np.sum(z[z < 0])),
                "abs_weight_sum": float(np.sum(np.abs(z))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 블랙박스인 AEC128 SVM이 실제로 학습한 계수 패턴이, 사람이
    손으로 정의한 "후반-중간 반등" 시각적 특징과 얼마나 비슷한가? — 모델 해석 가능성 점검):

    1. g1090/sdata를 로드하고 AEC128 SVM의 5-fold OOF/외부 점수를 표준화, g1090에서 방향(부호) 고정.
    2. 같은 데이터로 visual_score(후반-중간 반등)도 계산해 동일하게 표준화·부호 고정.
    3. 전체 g1090으로 SVM을 다시 학습해 128차원 전체에 대한 계수(표준화 스케일, 원본 스케일)를 복원.
    4. "중간 구간은 -1, 후반 구간은 +1"인 이상적인 시각적 특징 템플릿을 만들고, 실제 SVM 계수와
       위치별로 나란히 표로 저장.
    5. SVM 계수를 초반/중간/공백/후반 4구간으로 나눠 각 구간의 가중치 총합을 요약.
    6. SVM 점수 vs 시각적 점수의 상관, 각각과 라벨의 상관, 그리고 SVM 계수와 이상적 템플릿 간의
       코사인 유사도·피어슨 상관을 모두 계산해 "SVM이 정말 이 시각적 패턴을 학습했는지" 정량화.
    7. SVM 계수 곡선 위에 중간/후반 구간을 색으로 표시한 그래프를 저장하고, 전체 요약을 JSON으로
       저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    folds = make_folds(g["y"], 5)

    svm_oof, svm_ext = oof_and_external(lambda seed: aec_estimator(g["norm"].shape[1], seed), g["norm"], g["y"], s["norm"], folds)
    svm_g, svm_s, _, _ = zfit_apply(svm_oof, svm_ext)
    if pearson(svm_g, g["y"]) < 0:
        svm_g = -svm_g
        svm_s = -svm_s

    vis_g_raw = visual_score(g["norm"])
    vis_s_raw = visual_score(s["norm"])
    v_mu = float(np.mean(vis_g_raw))
    v_sd = float(np.std(vis_g_raw)) or 1.0
    vis_g = (vis_g_raw - v_mu) / v_sd
    vis_s = (vis_s_raw - v_mu) / v_sd
    if pearson(vis_g, g["y"]) < 0:
        vis_g = -vis_g
        vis_s = -vis_s

    coef_scaled, coef_orig = final_svm_coefficients(g["norm"], g["y"])
    if pearson(svm_g, g["y"]) < 0:
        coef_scaled = -coef_scaled
        coef_orig = -coef_orig

    template = np.zeros(g["norm"].shape[1], dtype=float)
    template[MID[0] - 1 : MID[1]] = -1.0 / (MID[1] - MID[0] + 1)
    template[TAIL[0] - 1 : TAIL[1]] = 1.0 / (TAIL[1] - TAIL[0] + 1)

    point = pd.DataFrame(
        {
            "point": np.arange(1, g["norm"].shape[1] + 1),
            "svm_coef_scaled": coef_scaled,
            "svm_coef_original_scale": coef_orig,
            "visual_tail_minus_mid_template": template,
        }
    )
    point.to_csv(OUT_DIR / "svm_coefficients_vs_visual_template.csv", index=False)
    zone = interval_summary(coef_orig)
    zone.to_csv(OUT_DIR / "svm_coefficient_zone_summary.csv", index=False)

    summary = {
        "visual_feature": f"mean(AEC {TAIL[0]}-{TAIL[1]}) - mean(AEC {MID[0]}-{MID[1]}), sign oriented so higher means higher low-SMI risk in g1090",
        "score_correlations": {
            "g1090_oof_svm_vs_visual_pearson": pearson(svm_g, vis_g),
            "sdata_external_svm_vs_visual_pearson": pearson(svm_s, vis_s),
            "g1090_oof_svm_vs_y_pearson": pearson(svm_g, g["y"]),
            "g1090_oof_visual_vs_y_pearson": pearson(vis_g, g["y"]),
            "sdata_external_svm_vs_y_pearson": pearson(svm_s, s["y"]),
            "sdata_external_visual_vs_y_pearson": pearson(vis_s, s["y"]),
        },
        "coefficient_template_similarity": {
            "cosine_original_coef_vs_visual_template": cosine(coef_orig, template),
            "pearson_original_coef_vs_visual_template": pearson(coef_orig, template),
            "cosine_scaled_coef_vs_visual_template": cosine(coef_scaled, template),
            "pearson_scaled_coef_vs_visual_template": pearson(coef_scaled, template),
        },
        "zone_summary": zone.to_dict(orient="records"),
    }
    (OUT_DIR / "svm_feature_interpretation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    x = point["point"]
    ax1.plot(x, point["svm_coef_original_scale"], color="#4C78A8", lw=2.0, label="SVM coefficient")
    ax1.axhline(0, color="#666666", ls="--", lw=1)
    ax1.axvspan(MID[0], MID[1], color="#F58518", alpha=0.16, label=f"mid {MID[0]}-{MID[1]}")
    ax1.axvspan(TAIL[0], TAIL[1], color="#54A24B", alpha=0.18, label=f"tail {TAIL[0]}-{TAIL[1]}")
    ax1.set_xlabel("AEC128 point")
    ax1.set_ylabel("SVM coefficient, original normalized-AEC scale")
    ax1.set_title("Does AEC128 SVM align with the visual tail-minus-mid feature?", loc="left", fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "svm_coefficients_vs_visual_feature.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
