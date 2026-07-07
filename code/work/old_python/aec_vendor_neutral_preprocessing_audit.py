from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm  # noqa: E402
from aec_midrange_feature_refit import clinical_scores  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_vendor_neutral_preprocessing_audit"
WORK_DATA_DIR = Path(__file__).resolve().parent / "data_cache"
SIGMA = 1.0
SEED = 20260701


def company_from_manufacturer(value: object) -> str:
    """CT 장비 제조사 문자열을 Siemens/Philips/GE/Other 4개 회사 범주로 매핑."""
    s = str(value).upper()
    if any(token in s for token in ["SOMATOM", "SENSATION", "SIEMENS"]):
        return "Siemens"
    if any(token in s for token in ["INGENUITY", "ICT", "PHILIPS"]):
        return "Philips"
    if any(token in s for token in ["REVOLUTION", "LIGHTSPEED", "GE"]):
        return "GE"
    return "Other"


def load_dataset(path: Path, cohort: str) -> dict:
    """엑셀에서 원시 AEC_128 곡선, 라벨, 제조사(회사) 범주를 함께 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    company = meta["Manufacturer"].map(company_from_manufacturer).to_numpy()
    return {"cohort": cohort, "meta": meta, "raw": raw, "y": y, "company": company}


def z_rows(x: np.ndarray) -> np.ndarray:
    """각 행(환자)을 자기 자신의 평균/표준편차로 z-표준화."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x - mu) / sd


def linear_residual(x: np.ndarray) -> np.ndarray:
    """곡선의 양 끝점을 잇는 직선을 빼서, 전체적인 기울기 성분을 제거한 잔차를 계산."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    line = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    return x - line


def rank_rows(x: np.ndarray) -> np.ndarray:
    """각 행의 값을 순위(0~1로 정규화)로 바꿔, 절대값 스케일에 무관한(순서만 남긴) 표현으로 변환."""
    ranked = np.vstack([stats.rankdata(row, method="average") for row in x])
    ranked = (ranked - 1.0) / (x.shape[1] - 1.0)
    return ranked - ranked.mean(axis=1, keepdims=True)


def derivative(x: np.ndarray) -> np.ndarray:
    """1차 도함수를 계산 (길이를 맞추기 위해 첫 값을 복제)."""
    d = np.diff(x, axis=1)
    return np.column_stack([d[:, :1], d])


def transforms(raw: np.ndarray) -> dict[str, np.ndarray]:
    """평활화된 원시곡선으로부터 평균정규화/로그중심화/순위모양/직선잔차/고역통과/도함수 등
    7가지 서로 다른 전처리 표현을 만들어 이름별 딕셔너리로 반환."""
    smooth_raw = ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")
    x = row_norm(smooth_raw)
    log_x = np.log(np.clip(x, 1e-6, None))
    resid = linear_residual(x)
    highpass = x - ndimage.gaussian_filter1d(x, sigma=8.0, axis=1, mode="nearest")
    d1 = derivative(x)
    return {
        "smooth_mean_norm": x,
        "smooth_log_centered": log_x - log_x.mean(axis=1, keepdims=True),
        "smooth_rank_shape": rank_rows(x),
        "smooth_linear_residual_z": z_rows(resid),
        "smooth_highpass_z_sigma8": z_rows(highpass),
        "smooth_derivative_z": z_rows(d1),
        "smooth_resid_plus_deriv_z": np.column_stack([z_rows(resid), z_rows(d1)]),
    }


def one_hot_auc(y: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> float:
    """다중 클래스(제조사) 분류에서 클래스별 원핫 AUC를 계산해 평균(매크로 AUC)을 구함."""
    aucs = []
    for i, label in enumerate(labels):
        yy = (y == label).astype(int)
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(aucs)) if aucs else np.nan


def company_cv_metrics(x: np.ndarray, company: np.ndarray) -> dict:
    """AEC 특징(PCA+로지스틱)만으로 CT 제조사를 얼마나 잘 맞히는지 교차검증으로 측정 — "이 전처리가
    장비 흔적을 얼마나 남기는지"를 재는 지표 (낮을수록 장비 중립적)."""
    keep = company != "Other"
    x = x[keep]
    y = company[keep]
    labels = np.array(sorted(np.unique(y)))
    y_idx = np.array([np.where(labels == v)[0][0] for v in y])
    counts = pd.Series(y).value_counts()
    n_splits = int(min(5, counts.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = np.zeros_like(y_idx)
    proba = np.zeros((len(y_idx), len(labels)), dtype=float)
    for tr, va in skf.split(x, y_idx):
        n_comp = min(12, x.shape[1], len(tr) - 1)
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, random_state=SEED),
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
        )
        model.fit(x[tr], y_idx[tr])
        pred[va] = model.predict(x[va])
        proba[va] = model.predict_proba(x[va])
    acc = float(np.mean(pred == y_idx))
    bal = float(balanced_accuracy_score(y_idx, pred))
    majority = float(counts.max() / counts.sum())
    macro_auc = one_hot_auc(y_idx, proba, np.arange(len(labels)))
    return {
        "company_cv_accuracy": acc,
        "company_cv_balanced_accuracy": bal,
        "company_majority_accuracy": majority,
        "company_macro_auc": macro_auc,
        "company_labels": "|".join(labels.tolist()),
    }


def low_smi_external_auc(xg: np.ndarray, xs: np.ndarray, yg: np.ndarray, ys: np.ndarray) -> dict:
    """train(PCA+로지스틱)으로 학습한 저근감소증 판별 모델의 train 자체 적합 AUC와 외부 AUC를 유의성(Mann-Whitney p)과 함께 계산."""
    n_comp = min(12, xg.shape[1], len(yg) - 1)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp, random_state=SEED),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=0.25),
    )
    model.fit(xg, yg)
    sg = model.decision_function(xg)
    ss = model.decision_function(xs)
    auc_g = float(roc_auc_score(yg, sg))
    auc_s = float(roc_auc_score(ys, ss))
    p_g = float(stats.mannwhitneyu(sg[yg == 1], sg[yg == 0], alternative="two-sided").pvalue)
    p_s = float(stats.mannwhitneyu(ss[ys == 1], ss[ys == 0], alternative="two-sided").pvalue)
    return {
        "aec_low_smi_train_auc_apparent": auc_g,
        "aec_low_smi_train_auc_p": p_g,
        "aec_low_smi_external_auc": auc_s,
        "aec_low_smi_external_auc_p": p_s,
    }


def between_company_ratio(x: np.ndarray, company: np.ndarray) -> float:
    """전체 분산 대비 "회사 간" 분산의 비율(분산분석의 between/total)을 계산 — 회사에 따라 곡선이
    얼마나 체계적으로 달라지는지 재는 지표."""
    keep = company != "Other"
    x = x[keep]
    c = company[keep]
    grand = x.mean(axis=0)
    total = float(np.mean(np.sum((x - grand) ** 2, axis=1)))
    if total <= 0:
        return np.nan
    between = 0.0
    for label in np.unique(c):
        sub = x[c == label]
        between += len(sub) * float(np.sum((sub.mean(axis=0) - grand) ** 2))
    between /= len(x)
    return between / total


def company_template_harmonize_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
    company_train: np.ndarray,
    company_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """train에서 회사별 평균 템플릿과 전체 평균 템플릿을 만들어, 각 환자 곡선에서 "자기 회사 평균"을
    빼고 "전체 평균"을 더하는 방식으로 회사 간 체계적 차이를 조정(harmonize)."""
    keep = company_train != "Other"
    global_template = x_train[keep].mean(axis=0)
    templates = {
        label: x_train[company_train == label].mean(axis=0)
        for label in np.unique(company_train[keep])
    }

    def apply(x: np.ndarray, company: np.ndarray) -> np.ndarray:
        """각 행에서 그 행의 회사 템플릿을 빼고 전체 템플릿을 더해 회사 차이를 보정."""
        out = np.empty_like(x)
        for i, label in enumerate(company):
            template = templates.get(label, global_template)
            out[i] = x[i] - template + global_template
        return out

    return apply(x_train, company_train), apply(x_test, company_test)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: g1090/sdata가 서로 다른 CT 장비(제조사)로 촬영됐는데, AEC
    전처리 방식에 따라 "장비 흔적"이 얼마나 남고, 그 흔적이 저근감소증 판별력에 영향을 주는가?):

    1. g1090/sdata를 로드하며 제조사 문자열을 Siemens/Philips/GE/Other로 분류한다.
    2. transforms로 7가지 전처리 표현(평균정규화/로그중심화/순위모양/직선잔차/고역통과/도함수/
       잔차+도함수 결합)을 만들고, 그 중 3개(평균정규화/로그중심화/순위모양)에는 회사별 템플릿
       보정(company_template_harmonize_train) 버전도 추가로 만든다.
    3. 각 전처리 표현마다: (a) between_company_ratio로 회사 간 분산 비율을, (b) company_cv_metrics로
       AEC 특징만으로 회사를 얼마나 잘 맞히는지(장비 흔적 정도)를, (c) low_smi_external_auc로
       저근감소증 판별 AUC(train 적합/외부)를 모두 계산해 한 표로 모은다.
    4. 모든 전처리 방식의 결과를 회사 판별 정확도·외부 AUC 기준으로 정렬해 CSV로 저장.
    5. 코호트x회사별 표본수를 CSV로 저장하고, 회사 카운트와 전처리 감사 결과를 콘솔에 출력
       (장비 흔적이 적으면서 판별력은 유지되는 전처리를 찾는 것이 목적).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_path = WORK_DATA_DIR / "g1090.xlsx" if (WORK_DATA_DIR / "g1090.xlsx").exists() else DATA_DIR / "g1090.xlsx"
    s_path = WORK_DATA_DIR / "sdata.xlsx" if (WORK_DATA_DIR / "sdata.xlsx").exists() else DATA_DIR / "sdata.xlsx"
    g = load_dataset(g_path, "Gangnam")
    s = load_dataset(s_path, "Sinchon")
    g_trans = transforms(g["raw"])
    s_trans = transforms(s["raw"])
    for base_name in ["smooth_mean_norm", "smooth_log_centered", "smooth_rank_shape"]:
        hg, hs = company_template_harmonize_train(g_trans[base_name], s_trans[base_name], g["company"], s["company"])
        g_trans[f"{base_name}_company_harmonized"] = hg
        s_trans[f"{base_name}_company_harmonized"] = hs
    company_all = np.concatenate([g["company"], s["company"]])

    rows = []
    for name in g_trans:
        xg = g_trans[name]
        xs = s_trans[name]
        x_all = np.vstack([xg, xs])
        row = {
            "preprocessing": name,
            "sigma": SIGMA,
            "between_company_variance_ratio": between_company_ratio(x_all, company_all),
            **company_cv_metrics(x_all, company_all),
            **low_smi_external_auc(xg, xs, g["y"], s["y"]),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["company_cv_balanced_accuracy", "aec_low_smi_external_auc"])
    summary.to_csv(OUT_DIR / "vendor_neutral_preprocessing_summary.csv", index=False)

    company_counts = pd.concat(
        [
            pd.DataFrame({"cohort": "Gangnam", "company": g["company"]}),
            pd.DataFrame({"cohort": "Sinchon", "company": s["company"]}),
        ],
        ignore_index=True,
    )
    company_counts.value_counts(["cohort", "company"]).reset_index(name="n").to_csv(
        OUT_DIR / "company_counts.csv", index=False
    )

    print("\nCOMPANY COUNTS")
    print(company_counts.value_counts(["cohort", "company"]).reset_index(name="n").to_string(index=False))
    print("\nPREPROCESSING AUDIT")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 여러 전처리 방식별로 장비(제조사) 흔적 정도와 저근감소증 판별 AUC를
    # 함께 계산해 비교하는 장비 중립성 감사 파이프라인이 수행된다.
    main()
