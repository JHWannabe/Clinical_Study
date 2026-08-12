from __future__ import annotations

# 임상변수 4개(sex/age/height/weight, clinic4)와 clinic4+AEC-128을 8구간으로 나눈 구간별 평균(8개값)으로
# 체성분 feature 7종의 이상 여부(성별 mean±1SD cutoff 이분화)를 예측하는 logistic regression 파이프라인.
# internal(Gangnam)에서 cutoff을 산출/고정하고 external(Sinchon)에는 그대로 적용하며,
# clinic4 vs clinic4+AEC의 AUC 차이를 같은 환자 집합에 대한 paired DeLong test로 비교한다.

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step4"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
N_SEG = 8  # 128슬라이스를 나눌 구간 수
MIN_POSITIVES = 2  # 이 미만이면 ROC/logistic 자체가 정의되지 않아 해당 feature/cohort를 skip

# 이분화 대상 체성분 feature -> (파일명 slug, cutoff 방향).
# "low" = mean-1SD 미만이면 이상(근육량 저하), "high" = mean+1SD 초과면 이상(지방/근육내지방 과다)
# LAMA(Low Attenuation Muscle Area)는 해부학적으로는 근육 영역(TAMA=NAMA+LAMA)이지만
# 낮은 감쇠값 자체가 근육 내 지방침윤(myosteatosis)을 의미하므로 "높을수록 이상" -> 지방계열과 같은 high 방향.
FEATURES: dict[str, tuple[str, str]] = {
    "TAMA_SUM": ("tama", "low"),
    "NAMA_SUM": ("nama", "low"),
    "LAMA_SUM": ("lama", "high"),
    "IMATA_SUM": ("imata", "high"),
    "SAT(피하지방)_SUM": ("sat", "high"),
    "VAT(내장지방)_SUM": ("vat", "high"),
    "Total Fat_SUM": ("total_fat", "high"),
}


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# 구간 수 n_seg별 컬럼명 생성 (예: n_seg=8 -> aec_seg1..aec_seg8), coef 라벨링용
def segment_col_names(n_seg: int) -> list[str]:
    return [f"aec_seg{i}" for i in range(1, n_seg + 1)]


# raw AEC-128 행렬(n x 128)을 n_seg개 구간으로 나눠 구간별 평균 행렬(n x n_seg)을 산출
def segment_means(aec_matrix: np.ndarray, n_seg: int) -> np.ndarray:
    chunks = np.array_split(aec_matrix, n_seg, axis=1)
    return np.column_stack([c.mean(axis=1) for c in chunks])


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 + (있으면) AEC 구간평균(n x n_seg) 결합.
# 성별을 고정한 남/여 개별 실행에서는 sex가 상수가 되어 계수가 무의미해지므로 include_sex=False로 제외
def clinical_matrix(meta: pd.DataFrame, aec_seg: np.ndarray | None = None,
                     scaler: StandardScaler | None = None, include_sex: bool = True) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = scaled if not include_sex else np.column_stack(
        [(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    x = clinic if aec_seg is None else np.column_stack([clinic, aec_seg])
    return x, scaler


# feature별 라벨 산출에 쓸 연속값: 모든 feature에 대해 raw SUM 값을 그대로 사용
def label_source_values(meta: pd.DataFrame, feat: str) -> np.ndarray:
    return pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)


# internal 코호트의 성별(M/F) 그룹별로 mean±1SD cutoff을 산출 ("low"->mean-1SD, "high"->mean+1SD).
# 남/여 개별 실행에서는 데이터에 한쪽 성별만 존재하므로 실제 존재하는 성별에 대해서만 산출
def sex_specific_cutoffs(values: np.ndarray, sex: np.ndarray, direction: str) -> dict[str, float]:
    cutoffs = {}
    for s in ("M", "F"):
        v = values[(sex == s) & np.isfinite(values)]
        if len(v) == 0:
            continue
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        cutoffs[s] = mean - 1 * sd if direction == "low" else mean + 1 * sd
    return cutoffs


# 성별 cutoff으로 이분형 라벨(이상=1)을 산출. cutoff은 항상 internal에서 고정한 절대값을 그대로 적용(재계산 금지)
def apply_cutoff_label(values: np.ndarray, sex: np.ndarray, cutoffs: dict[str, float], direction: str) -> np.ndarray:
    th = np.full(len(sex), np.nan)
    for s, c in cutoffs.items():
        th[sex == s] = c
    return ((values < th) if direction == "low" else (values > th)).astype(int)


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


# scores: (n_scores, n_samples), 앞 n_pos개 열이 양성. DeLong et al.(1988)/Sun & Xu(2014) 알고리즘으로
# 각 score의 AUC와 그 공분산 행렬을 산출한다 (같은 표본으로 계산된 두 AUC를 비교하려면 이 공분산이 필요)
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


# 같은 환자 집합(같은 y)에서 나온 두 예측 점수(score_a vs score_b)의 AUC가 서로 다른지 검정하는 paired DeLong test.
# clinic4와 clinic4+AEC는 동일 환자에 대해 평가되므로 독립 two-sample test가 아니라 이 paired test를 써야 함
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


# AUC의 bootstrap 95% CI 산출. 양성 비율이 낮아 일부 resample은 한쪽 클래스가 비므로 그런 반복은 제외
def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 3000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for bi in rng.integers(0, n, size=(n_boot, n)):
        y_bi = y[bi]
        if len(np.unique(y_bi)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_bi, score[bi]))
    if len(boot_aucs) < n_boot * 0.5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


# internal OOF ROC에서 Youden's J(sensitivity+specificity-1)를 최대화하는 threshold를 선택 (external에는 이 값을 고정 적용)
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


# feat/model/cohort별 핵심 통계를 한 줄로 출력
def _log(feat: str, model_name: str, cohort: str, s: dict) -> None:
    ci_lo, ci_hi = s["auc_ci_lower"], s["auc_ci_upper"]
    print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
          f"AUC={s['auc']:.3f} 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}] "
          f"Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} Acc={s['accuracy']:.3f}")


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# feature 하나에 대해 internal(OOF)/external(frozen) ROC curve를 clinic4 vs clinic4+AEC 겹쳐 그림.
# 범례에 AUC(95%CI)와 DeLong p-value를 함께 표시
def plot_roc_dual(feat: str, curves: dict[str, dict[str, np.ndarray]],
                   stats_by_model: dict[str, dict[str, dict]], delong: dict[str, dict], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    colors = {"clinic4": "#2a78d6", "clinic4_aec_mean": "#e2622e"}
    labels = {"clinic4": "clinic4", "clinic4_aec_mean": f"clinic4 + AEC {N_SEG}구간평균"}
    cohorts = ["internal", "external"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, cohort in zip(axes, cohorts):
        for model_name in ("clinic4", "clinic4_aec_mean"):
            y = curves[cohort]["y"]
            score = curves[cohort][model_name]
            fpr, tpr, _ = roc_curve(y, score)
            s = stats_by_model[model_name][cohort]
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=1.8,
                    label=f"{labels[model_name]} AUC={s['auc']:.3f} "
                          f"[{s['auc_ci_lower']:.3f}, {s['auc_ci_upper']:.3f}]")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        p = delong[cohort]["p_value"]
        p_str = "p<0.001" if np.isfinite(p) and p < 0.001 else f"p={p:.3f}" if np.isfinite(p) else "p=n/a"
        ax.set_title(f"{feat} ({cohort})\nDeLong (clinic4 vs +AEC) {p_str}", fontsize=33,
                     fontweight="bold", color=INK_PRIMARY)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(fontsize=24, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


# 전체 feature에 걸쳐 AUC(95%CI)를 clinic4 vs clinic4+AEC, internal vs external로 비교하는 막대그래프
def plot_auc_summary(summary: pd.DataFrame, out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    slugs = [FEATURES[f][0] for f in features]
    x = np.arange(len(features))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(6 * len(features) / 3 + 2, 5.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        c4 = sub[sub["model"] == "clinic4"].set_index("feature").reindex(features)
        c4a = sub[sub["model"] == "clinic4_aec_mean"].set_index("feature").reindex(features)
        c4_err = np.abs(np.vstack([c4["auc"] - c4["auc_ci_lower"], c4["auc_ci_upper"] - c4["auc"]]))
        c4a_err = np.abs(np.vstack([c4a["auc"] - c4a["auc_ci_lower"], c4a["auc_ci_upper"] - c4a["auc"]]))
        ax.bar(x - width / 2, c4["auc"], width, yerr=c4_err, capsize=3, label="clinic4", color="#2a78d6")
        ax.bar(x + width / 2, c4a["auc"], width, yerr=c4a_err, capsize=3, label=f"clinic4 + AEC {N_SEG}구간평균", color="#e2622e")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs)
        ax.set_ylim(0, 1)
        ax.set_title(cohort, fontsize=36, fontweight="bold", color=INK_PRIMARY)
        ax.set_ylabel("AUC")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=24)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


# clinic4(include_sex=False면 clinic3, 4개)만 쓴 모델과 clinic4/3+AEC 8구간평균(8개값)을 쓴 모델로 체성분
# feature 7종의 이상 여부를 logistic regression 예측. cutoff은 internal 성별 mean±1SD로 산출/고정 후 external에
# 그대로 적용, internal OOF로 모델을 학습/평가하고 external에는 고정 모델을 1회만 적용(freeze), AUC는 DeLong test로 비교
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper().to_numpy()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper().to_numpy()

    aec_int_seg = segment_means(meta_int[AEC_COLS].astype(float).to_numpy(), N_SEG)
    aec_ext_seg = segment_means(meta_ext[AEC_COLS].astype(float).to_numpy(), N_SEG)

    x_int_by_model = {}
    x_ext_by_model = {}
    x_int_by_model["clinic4"], scaler = clinical_matrix(meta_int, include_sex=include_sex)
    x_ext_by_model["clinic4"], _ = clinical_matrix(meta_ext, scaler=scaler, include_sex=include_sex)
    x_int_by_model["clinic4_aec_mean"], _ = clinical_matrix(meta_int, aec_int_seg, scaler, include_sex=include_sex)
    x_ext_by_model["clinic4_aec_mean"], _ = clinical_matrix(meta_ext, aec_ext_seg, scaler, include_sex=include_sex)

    cutoff_rows = []
    summary_rows = []
    delong_rows = []
    skipped = []

    for feat, (slug, direction) in FEATURES.items():
        values_int = label_source_values(meta_int, feat)
        values_ext = label_source_values(meta_ext, feat)

        mask_val_int = np.isfinite(values_int)
        cutoffs = sex_specific_cutoffs(values_int[mask_val_int], sex_int[mask_val_int], direction)

        mask_val_ext = np.isfinite(values_ext)
        y_int_all = apply_cutoff_label(values_int, sex_int, cutoffs, direction)
        y_ext_all = apply_cutoff_label(values_ext, sex_ext, cutoffs, direction)

        for s, cutoff_val in cutoffs.items():
            v = values_int[mask_val_int & (sex_int == s)]
            cutoff_rows.append({"feature": feat, "sex": s, "direction": direction,
                                 "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "cutoff": cutoff_val, "n_internal": int(len(v))})

        n_pos_int = int(y_int_all[mask_val_int].sum())
        n_neg_int = int(mask_val_int.sum() - n_pos_int)
        n_pos_ext = int(y_ext_all[mask_val_ext].sum())
        n_neg_ext = int(mask_val_ext.sum() - n_pos_ext)
        cutoff_str = ", ".join(f"{s}={c:.2f}" for s, c in cutoffs.items())
        print(f"[{feat}] cutoff {cutoff_str} | "
              f"internal n_pos={n_pos_int}/{mask_val_int.sum()} external n_pos={n_pos_ext}/{mask_val_ext.sum()}")

        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            msg = (f"[{feat}] SKIP: mean±1SD cutoff이 이 코호트에서 한쪽 클래스를 {MIN_POSITIVES}명 미만으로 "
                   f"만들어 logistic regression/ROC 산출이 불가함 "
                   f"(internal pos={n_pos_int} neg={n_neg_int}, external pos={n_pos_ext} neg={n_neg_ext})")
            print(msg)
            skipped.append({"feature": feat, "reason": msg, "n_pos_internal": n_pos_int,
                             "n_neg_internal": n_neg_int, "n_pos_external": n_pos_ext, "n_neg_external": n_neg_ext})
            continue

        mask_int = mask_val_int
        mask_ext = mask_val_ext
        y_int = y_int_all[mask_int]
        y_ext = y_ext_all[mask_ext]

        n_splits = max(2, min(N_FOLDS, n_pos_int, n_neg_int))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)

        stats_by_model: dict[str, dict[str, dict]] = {"clinic4": {}, "clinic4_aec_mean": {}}
        scores_by_model: dict[str, dict[str, np.ndarray]] = {"clinic4": {}, "clinic4_aec_mean": {}}
        coef_sheets = {}

        for model_name in ("clinic4", "clinic4_aec_mean"):
            x_int = x_int_by_model[model_name][mask_int]
            x_ext = x_ext_by_model[model_name][mask_ext]

            oof_proba = cross_val_predict(LogisticRegression(max_iter=2000), x_int, y_int,
                                           cv=cv, method="predict_proba")[:, 1]
            model = LogisticRegression(max_iter=2000).fit(x_int, y_int)
            ext_proba = model.predict_proba(x_ext)[:, 1]

            threshold = youden_threshold(y_int, oof_proba)

            for cohort, y, score in (("internal", y_int, oof_proba), ("external", y_ext, ext_proba)):
                auc = float(roc_auc_score(y, score))
                ci_lo, ci_hi = bootstrap_auc_ci(y, score)
                cls_stats = classification_stats(y, score, threshold)
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
                     "auc": auc, "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi,
                     "threshold": threshold, **cls_stats}
                _log(feat, model_name, cohort, s)
                summary_rows.append({"feature": feat, "model": model_name, "cohort": cohort, **s})
                stats_by_model[model_name][cohort] = s
                scores_by_model[model_name][cohort] = score

            input_cols = (["sex_M"] if include_sex else []) + ["age", "height", "weight"] + \
                (segment_col_names(N_SEG) if model_name == "clinic4_aec_mean" else [])
            coef_df = pd.DataFrame({
                "term": input_cols + ["intercept"],
                "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
            })
            coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
            coef_sheets[model_name] = coef_df.round(4)

        delong_int = delong_paired_auc_test(y_int, scores_by_model["clinic4"]["internal"],
                                             scores_by_model["clinic4_aec_mean"]["internal"])
        delong_ext = delong_paired_auc_test(y_ext, scores_by_model["clinic4"]["external"],
                                             scores_by_model["clinic4_aec_mean"]["external"])
        for cohort, d in (("internal", delong_int), ("external", delong_ext)):
            print(f"[{feat} / {cohort}] DeLong clinic4 vs clinic4+AEC: "
                  f"AUC diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
            delong_rows.append({"feature": feat, "cohort": cohort, "auc_clinic4": d["auc_a"],
                                 "auc_clinic4_aec_mean": d["auc_b"], "auc_diff": d["diff"],
                                 "z": d["z"], "p_value": d["p_value"]})

        write_sheets(feat_dir / f"{slug}_logistic_coefficients.xlsx", coef_sheets)

        curves = {
            "internal": {"y": y_int, **{m: scores_by_model[m]["internal"] for m in scores_by_model}},
            "external": {"y": y_ext, **{m: scores_by_model[m]["external"] for m in scores_by_model}},
        }
        plot_roc_dual(feat, curves, stats_by_model, {"internal": delong_int, "external": delong_ext},
                      feat_dir / f"{slug}_roc_curve.png")

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_features.csv", index=False)
        print(f"Saved skipped feature log to {output_dir / 'skipped_features.csv'}")

    pd.DataFrame(cutoff_rows).to_csv(output_dir / "label_cutoffs.csv", index=False)
    print(f"Saved label cutoffs to {output_dir / 'label_cutoffs.csv'}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    if not summary.empty:
        plot_auc_summary(summary, output_dir / "logistic_regression_auc_summary.png")


# internal/external 코호트를 로드/전처리 후 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행
def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[clinical_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_clinical_int = valid_rows(meta_int)
    mask_clinical_ext = valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_clinical_int).sum()}/{len(mask_clinical_int)}, "
          f"external {(~mask_clinical_ext).sum()}/{len(mask_clinical_ext)}")
    meta_int = meta_int[mask_clinical_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_clinical_ext].reset_index(drop=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper()

    run(meta_int, meta_ext, OUTPUT_DIR / "total", include_sex=True)
    for sex_label, sub_dir in (("M", OUTPUT_DIR / "male"), ("F", OUTPUT_DIR / "female")):
        print(f"\n=== sex={sex_label} ({sub_dir.name}) ===")
        run(meta_int[sex_int == sex_label].reset_index(drop=True),
            meta_ext[sex_ext == sex_label].reset_index(drop=True), sub_dir, include_sex=False)


if __name__ == "__main__":
    main()
