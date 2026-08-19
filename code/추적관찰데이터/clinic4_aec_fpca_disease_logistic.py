from __future__ import annotations

# clinic4_disease_logistic.py의 clinic4(성별/나이/신장/체중) 단일 모델에 AEC-128 곡선 형태 feature
# 4종(SD/Skewness/상하위50%비율/FPCA)을 전부 하나의 모델로 결합해 질병 6종(당뇨병/고혈압/이상지질혈증/
# 골다공증/심근경색/뇌졸중) 유무를 예측하는 logistic regression. 이전 버전은 질환마다 5개 후보(SD/
# Skewness/상하위비율/FPCA/전체결합) 중 internal OOF AUC가 가장 높은 조합을 독립적으로 선택했으나,
# 사용자 확인 2026-08-19: "5개를 모두 결합하고 추가로 fpca 값도 다 더해서 분류 진행" - 질환별 선택 없이
# SD+Skewness+상하위비율+FPCA를 항상 전부 결합한 단일 모델로 통일한다. FPCA n_components는 3(elbow, 유지
# - 사용자 확인: "elbow 값(n=3) 유지"), aec_128_contrast 전체 코호트 scree curve의 elbow(Kneedle 방식:
# 축 정규화 후 첫점-끝점 직선까지 수직거리 최대 지점)로 산출한다.
#
# aec_128_contrast 시트는 환자당 조영 시리즈가 여러 개(최대 6개, Arterial/Portal/Delay/일반 With Contrast
# 등 혼재)라 형태 feature/FPCA에 넣을 환자당 1개 곡선이 필요함 - Portal phase가 있으면 그것을, 없으면
# 원본 순서상 첫 시리즈를 사용한다(사용자 확인 2026-08-19: "Portal phase 우선 선택").
#
# clinic4_disease_logistic.py와 달리 이 코호트는 internal/external 분리가 없는 단일 소스이므로, clinic4
# 단독과 clinic4+AEC-결합 두 모델을 같은 5-fold CV split으로 학습해 OOF AUC를 DeLong paired test로
# 비교한다. PCA는 fold의 train 구간에서만 fit해(fold별 refit) 검증 fold 정보가 고유함수 추정에 새는
# leakage를 방지한다(질병예측/step3의 _fpca_oof_proba_predict와 동일 원칙). SD/Skewness/상하위비율은
# 환자별로 독립 계산되는 값이라 fold와 무관해 refit이 필요 없다.

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "followup_clinic4_aec_fpca_disease_logistic"

AEC_XLSX = DATA_DIR / "aec_cropped.xlsx"
METADATA_SHEET = "metadata_cleaned"
AEC_CONTRAST_SHEET = "aec_128_contrast"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
CLINICAL_BASE_COLS = ["나이", "신장", "체중"]  # clinic4의 성별 제외 나머지 3개(표준화 대상)
FPCA_COMPONENT_CANDIDATES_MAX = 20
MIN_POSITIVES = 2  # 이 미만이면 ROC/logistic 자체가 정의되지 않아 해당 질병을 skip
AEC_COMBINED_LABEL = "SD+Skewness+상하위50%비율+FPCA 결합"

DISEASES: dict[str, str] = {
    "당뇨병_여부": "dm",
    "고혈압_여부": "htn",
    "이상지질혈증_여부": "dyslipidemia",
    "골다공증_여부": "osteoporosis",
    "심근경색_여부": "mi",
    "뇌졸중_여부": "stroke",
}


# 환자당 여러 조영 시리즈 중 Portal phase(series_desc에 "portal" 포함)를 우선 선택, 없으면 원본 순서상
# 첫 시리즈를 사용해 환자당 1행으로 축소
def select_series_per_patient(aec: pd.DataFrame) -> pd.DataFrame:
    df = aec.copy()
    df["_portal_priority"] = (~df["series_desc"].astype(str).str.contains("portal", case=False, na=False)).astype(int)
    df = df.sort_values(["PatientID", "_portal_priority"], kind="stable")
    result = df.drop_duplicates(subset="PatientID", keep="first").drop(columns="_portal_priority")
    return result.reset_index(drop=True)


# metadata_cleaned(clinic4+질병 6종)와 aec_128_contrast(환자당 1행으로 축소한 AEC-128 raw)를 PatientID
# 기준 inner join. 성별이 M/F가 아니거나 clinic4 입력이 결측인 행은 제외
def load_cohort() -> pd.DataFrame:
    meta = pd.read_excel(AEC_XLSX, sheet_name=METADATA_SHEET, engine="openpyxl").reset_index(drop=True)
    valid_sex = meta["성별"].astype(str).str.upper().isin(["M", "F"])
    valid_clinic = meta[CLINICAL_BASE_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    mask = valid_sex & valid_clinic
    print(f"clinic4 입력 결측/성별 이상 제외: {(~mask).sum()}/{len(mask)}명")
    meta = meta[mask].reset_index(drop=True)

    aec_raw = pd.read_excel(AEC_XLSX, sheet_name=AEC_CONTRAST_SHEET, engine="openpyxl")
    is_portal = aec_raw["series_desc"].astype(str).str.contains("portal", case=False, na=False)
    n_patients_with_portal = aec_raw.loc[is_portal, "PatientID"].nunique()
    aec = select_series_per_patient(aec_raw)
    print(f"aec_128_contrast {len(aec_raw)}행(고유 {aec_raw['PatientID'].nunique()}명) -> 환자당 1행으로 축소 "
          f"{len(aec)}행 (Portal phase 보유 {n_patients_with_portal}명, 나머지는 원본 순서상 첫 시리즈 사용)")

    merged = meta.merge(aec[["PatientID"] + AEC_COLS], left_on="patientID", right_on="PatientID", how="inner")
    print(f"metadata_cleaned {len(meta)}명 -> aec_128_contrast 교집합 {len(merged)}명")
    return merged.reset_index(drop=True)


# 성별(M=1/F=0) + 표준화된 나이/신장/체중으로 clinic4 입력 행렬을 구성
def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[CLINICAL_BASE_COLS].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    sex_m = (meta["성별"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


# raw AEC-128 행렬(n x 128)에서 환자별 SD, Skewness, 슬라이스 위치 기준 상위/하위 50% 평균 비율을 산출.
# 128개 값이 거의 동일해 SD=0에 가까운 환자(9,914명 중 699명)는 skewness 계산식이 0/0이 되어 NaN이
# 나오는데, 곡선이 평탄하면 비대칭도 자체가 없는 게 정상이므로 0으로 채운다(NaN 그대로 두면 logistic
# regression 입력에 NaN이 섞여 학습이 실패함)
def shape_features(aec_matrix: np.ndarray) -> dict[str, np.ndarray]:
    sd = aec_matrix.std(axis=1, ddof=1)
    skewness = np.nan_to_num(stats.skew(aec_matrix, axis=1), nan=0.0)
    half = N_SLICES // 2
    upper_mean = aec_matrix[:, :half].mean(axis=1)
    lower_mean = aec_matrix[:, half:].mean(axis=1)
    upper_lower_ratio = upper_mean / lower_mean
    return {"sd": sd, "skew": skewness, "upper_lower_ratio": upper_lower_ratio}


# scree curve(컴포넌트별 개별 explained variance ratio)를 축 정규화 후 첫점-끝점 직선까지 수직거리로 계산
# (Satopaa et al. 2011 Kneedle 알고리즘, 질병예측/step2·3의 _scree_and_elbow_distance와 동일 로직)
def _scree_and_elbow_distance(cum_var: pd.Series) -> tuple[pd.Series, pd.Series]:
    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    return scree, pd.Series(dist, index=scree.index)


# 전체 코호트 AEC-128 곡선의 scree curve에서 elbow n_components를 계산해 그대로 채택(고정값으로 override
# 하지 않음 - 사용자 확인 2026-08-19: "elbow 값(n=3) 유지")
def select_elbow_fpca_n(aec_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_raw.shape[0], aec_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))
    _, dist = _scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] elbow n_components = {elbow_n} (누적분산비율={cum_var[elbow_n]:.4f}, chord-거리={dist[elbow_n]:.4f})")
    return elbow_n, cum_var


# n_components별 누적/개별 explained variance ratio, elbow 판단에 쓴 chord-거리를 엑셀로 저장(그래프의 원자료)
def save_cum_var_excel(cum_var: pd.Series, elbow_n: int, out_path: Path) -> None:
    scree, dist = _scree_and_elbow_distance(cum_var)
    df = pd.DataFrame({"n_components": cum_var.index, "cumulative_variance_ratio": cum_var.values,
                        "individual_variance_ratio": scree.values, "elbow_chord_distance": dist.values})
    df["is_elbow_selected"] = df["n_components"] == elbow_n
    df.to_excel(out_path, index=False)
    print(f"Saved FPCA cumulative variance ratio to {out_path}")


# elbow n_components별 scree curve(막대: 개별 explained variance ratio, 선: 누적)와 elbow 지점을 시각화
def plot_scree_elbow(cum_var: pd.Series, elbow_n: int, out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    scree, _ = _scree_and_elbow_distance(cum_var)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(scree.index, scree.values, color="#9fb8d9", label="개별 explained variance ratio")
    ax1.set_xlabel("n_components", fontsize=15)
    ax1.set_ylabel("개별 explained variance ratio", fontsize=15, color="#2a78d6")
    ax1.tick_params(axis="both", labelsize=13)

    ax2 = ax1.twinx()
    ax2.plot(cum_var.index, cum_var.values, color="#d30909", marker="o", linewidth=2, label="누적 explained variance ratio")
    ax2.axvline(elbow_n, color="#161616", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("누적 explained variance ratio", fontsize=15, color="#d30909")
    ax2.tick_params(axis="y", labelsize=13)

    ax1.set_title(f"AEC-128 FPCA scree curve (elbow n_components={elbow_n})", fontsize=17,
                  fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scree/elbow plot to {out_path}")


# 질병 유병 여부 클래스 균형에 맞춘 StratifiedKFold fold 수(최소 2, 최대 N_FOLDS)
def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


# clinic4 단독 OOF 확률 산출(비교 기준선)
def oof_predict_clinic4(meta: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(meta, y):
        x_train, scaler = clinical_matrix(meta.iloc[train_idx])
        x_test, _ = clinical_matrix(meta.iloc[test_idx], scaler)
        model = LogisticRegression(max_iter=2000).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


# clinic4 + AEC 결합(SD+Skewness+상하위비율+FPCA) 모델의 OOF 확률을 산출. SD/Skewness/상하위비율은 환자별로
# 독립 계산되는 값이라 fold와 무관하지만, FPCA는 fold의 train 구간에서만 PCA를 fit해(fold별 refit) 검증
# fold 정보가 고유함수 추정에 새는 leakage를 방지한다
def oof_predict_combined(meta: pd.DataFrame, aec_raw: np.ndarray, shape_stack: np.ndarray, y: np.ndarray,
                          cv: StratifiedKFold, n_fpca: int) -> np.ndarray:
    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(aec_raw, y):
        x4_train, scaler4 = clinical_matrix(meta.iloc[train_idx])
        x4_test, _ = clinical_matrix(meta.iloc[test_idx], scaler4)

        pca = PCA(n_components=n_fpca, random_state=SEED).fit(aec_raw[train_idx])
        fpca_train, fpca_test = pca.transform(aec_raw[train_idx]), pca.transform(aec_raw[test_idx])
        extra_train = np.column_stack([shape_stack[train_idx], fpca_train])
        extra_test = np.column_stack([shape_stack[test_idx], fpca_test])

        extra_scaler = StandardScaler().fit(extra_train)
        x_train = np.column_stack([x4_train, extra_scaler.transform(extra_train)])
        x_test = np.column_stack([x4_test, extra_scaler.transform(extra_test)])

        model = LogisticRegression(max_iter=2000).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


# DeLong midrank (동순위는 평균 순위로 처리)
def _delong_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


# scores: (n_scores, n_samples), 앞 n_pos개 열이 양성. DeLong et al.(1988)/Sun & Xu(2014) 알고리즘
def _delong_covariance(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    n_neg = scores.shape[1] - n_pos
    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    k = scores.shape[0]
    tx = np.vstack([_delong_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_delong_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_delong_midrank(scores[r]) for r in range(k)])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


# 같은 환자 집합(같은 y)에서 나온 두 예측 점수의 AUC가 서로 다른지 검정하는 paired DeLong test
def delong_paired_auc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
                "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


# OOF ROC에서 Youden's J(sensitivity+specificity-1)를 최대화하는 threshold
def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


# 확률 점수와 고정 threshold로 sensitivity/specificity/accuracy 산출
def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


def _log(disease: str, model_name: str, s: dict) -> None:
    print(f"[{disease} / {model_name}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
          f"AUC={s['auc']:.3f} Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} Acc={s['accuracy']:.3f}")


def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# 질병 하나에 대해 clinic4 vs clinic4+AEC-결합의 OOF ROC curve를 겹쳐 그림, 제목에 DeLong test p-value 표기
def plot_roc_pair(disease: str, y: np.ndarray, score_c4: np.ndarray, score_comb: np.ndarray,
                   s_c4: dict, s_comb: dict, delong: dict, out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, score, s, color in (
        ("clinic4", score_c4, s_c4, "#2a78d6"),
        (f"clinic4 + AEC({AEC_COMBINED_LABEL})", score_comb, s_comb, "#1baf7a"),
    ):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{label} AUC={s['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)

    p = delong["p_value"]
    p_label = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.set_title(f"{disease} (DeLong {p_label})", fontsize=17, fontweight="bold", color=INK_PRIMARY)
    ax.set_xlabel("1 - Specificity", fontsize=15)
    ax.set_ylabel("Sensitivity", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, loc="lower right", frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


# 질병 6개 x clinic4/clinic4+AEC-결합의 AUC를 나란히 비교하는 막대그래프
def plot_auc_summary(summary: pd.DataFrame, out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    colors = {"clinic4": "#6b6a66", "clinic4_aec_combined": "#1baf7a"}
    labels = {"clinic4": "clinic4", "clinic4_aec_combined": f"clinic4 + AEC({AEC_COMBINED_LABEL})"}
    diseases = [d for d in DISEASES if d in summary["disease"].unique()]
    slugs = [DISEASES[d] for d in diseases]
    x = np.arange(len(diseases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.2 * len(diseases) + 2, 6))
    for i, model_name in enumerate(("clinic4", "clinic4_aec_combined")):
        rows = summary[summary["model"] == model_name].set_index("disease").reindex(diseases)
        offset = (i - 0.5) * width
        ax.bar(x + offset, rows["auc"], width, label=labels[model_name], color=colors[model_name])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, fontsize=16, rotation=0)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"clinic4 vs clinic4+AEC({AEC_COMBINED_LABEL}) 질병 예측 AUC (추적관찰 코호트, internal OOF)",
                 fontsize=15, fontweight="bold", color=INK_PRIMARY)
    ax.set_ylabel("AUC", fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(fontsize=13, frameon=False)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


# 전체 코호트로 clinic4+AEC-결합(SD+Skewness+상하위비율+FPCA)을 fit한 최종 모델의 계수/오즈비 표를 만든다
def build_final_coef_table(meta: pd.DataFrame, aec_raw: np.ndarray, shape_stack: np.ndarray, y: np.ndarray,
                            n_fpca: int) -> pd.DataFrame:
    x4_full, _ = clinical_matrix(meta)
    pca_full = PCA(n_components=n_fpca, random_state=SEED).fit(aec_raw)
    fpca_full = pca_full.transform(aec_raw)
    extra_full = np.column_stack([shape_stack, fpca_full])
    extra_terms = ["aec_sd", "aec_skew", "aec_uplow_ratio"] + [f"FPCA{i}" for i in range(1, n_fpca + 1)]

    extra_scaler = StandardScaler().fit(extra_full)
    x_full = np.column_stack([x4_full, extra_scaler.transform(extra_full)])
    model = LogisticRegression(max_iter=2000).fit(x_full, y)

    coef_df = pd.DataFrame({
        "term": ["sex_M", "age", "height", "weight"] + extra_terms + ["intercept"],
        "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
    })
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    return coef_df.round(4)


# clinic4 단독과 clinic4+AEC-결합(SD+Skewness+상하위50%비율+FPCA를 항상 전부 결합, 질환별 선택 없음) 두
# 모델로 질병 6종 유무를 각각 독립적인 logistic regression으로 예측한다. 두 모델은 같은 5-fold CV split을
# 공유해 OOF AUC를 DeLong paired test로 비교
def run(meta: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aec_raw = meta[AEC_COLS].astype(float).to_numpy()
    elbow_n, cum_var = select_elbow_fpca_n(aec_raw)
    save_cum_var_excel(cum_var, elbow_n, output_dir / "fpca_cumulative_variance.xlsx")
    plot_scree_elbow(cum_var, elbow_n, output_dir / "fpca_scree_elbow.png")

    shape_all = shape_features(aec_raw)
    shape_stack_all = np.column_stack([shape_all["sd"], shape_all["skew"], shape_all["upper_lower_ratio"]])

    valid_diseases: list[str] = []
    skipped = []
    for disease in DISEASES:
        y_all = pd.to_numeric(meta[disease], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        n_pos = int(y_all[mask].sum())
        n_neg = int(mask.sum() - n_pos)
        print(f"[{disease}] n_pos={n_pos}/{mask.sum()}")
        if min(n_pos, n_neg) < MIN_POSITIVES:
            msg = (f"[{disease}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만이라 logistic regression/ROC 산출이 불가함 "
                   f"(pos={n_pos} neg={n_neg})")
            print(msg)
            skipped.append({"disease": disease, "reason": msg, "n_pos": n_pos, "n_neg": n_neg})
            continue
        valid_diseases.append(disease)

    summary_rows = []
    delong_rows = []
    predictions_rows = []

    for disease in valid_diseases:
        slug = DISEASES[disease]
        y_full = pd.to_numeric(meta[disease], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_full)
        meta_masked = meta.loc[mask].reset_index(drop=True)
        aec_masked = aec_raw[mask]
        shape_stack_masked = shape_stack_all[mask]
        y = y_full[mask].astype(int)

        cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)
        oof_c4 = oof_predict_clinic4(meta_masked, y, cv)
        oof_comb = oof_predict_combined(meta_masked, aec_masked, shape_stack_masked, y, cv, elbow_n)

        threshold_c4 = youden_threshold(y, oof_c4)
        threshold_comb = youden_threshold(y, oof_comb)

        stats_by_model = {}
        for model_name, score, threshold in (
            ("clinic4", oof_c4, threshold_c4), ("clinic4_aec_combined", oof_comb, threshold_comb),
        ):
            auc = float(roc_auc_score(y, score))
            cls_stats = classification_stats(y, score, threshold)
            s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
                 "auc": auc, "threshold": threshold, **cls_stats}
            _log(disease, model_name, s)
            stats_by_model[model_name] = s
            summary_rows.append({"disease": disease, "model": model_name, **s})
            predictions_rows.append(pd.DataFrame({
                "disease": disease, "model": model_name,
                "patient_id": meta_masked["patientID"].to_numpy(),
                "y": y, "score": score, "threshold": threshold,
            }))

        delong = delong_paired_auc_test(y, oof_c4, oof_comb)
        print(f"[{disease}] DeLong clinic4 vs clinic4+AEC(결합): AUC diff={delong['diff']:+.4f} "
              f"z={delong['z']:.3f} p={delong['p_value']:.4f}")
        delong_rows.append({"disease": disease, "auc_clinic4": delong["auc_a"],
                             "auc_clinic4_aec_combined": delong["auc_b"], "auc_diff": -delong["diff"],
                             "z": -delong["z"], "p_value": delong["p_value"]})

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)
        plot_roc_pair(disease, y, oof_c4, oof_comb, stats_by_model["clinic4"],
                      stats_by_model["clinic4_aec_combined"], delong, feat_dir / f"{slug}_roc_curve.png")

        coef_df = build_final_coef_table(meta_masked, aec_masked, shape_stack_masked, y, elbow_n)
        write_sheets(feat_dir / f"{slug}_logistic_coefficients.xlsx", {"clinic4_aec_combined": coef_df})

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_diseases.csv", index=False)
        print(f"Saved skipped disease log to {output_dir / 'skipped_diseases.csv'}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    predictions = pd.concat(predictions_rows, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved per-patient predictions to {output_dir / 'predictions.csv'}")

    if not summary.empty:
        plot_auc_summary(summary, output_dir / "logistic_regression_auc_summary.png")


def main() -> None:
    meta = load_cohort()
    run(meta, OUTPUT_DIR)


if __name__ == "__main__":
    main()
