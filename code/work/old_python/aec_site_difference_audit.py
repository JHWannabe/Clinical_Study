from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "data_cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_site_difference_audit"
SEED = 20260701
SIGMA = 1.0
N_PERM = 2000


def matrix_from_sheet(df: pd.DataFrame) -> np.ndarray:
    """데이터프레임에서 'aec_숫자' 형태 컬럼만 골라 번호순으로 정렬한 숫자 행렬로 변환."""
    cols = [c for c in df.columns if re.fullmatch(r"aec_\d+", str(c))]
    cols = sorted(cols, key=lambda c: int(str(c).split("_")[1]))
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def company_from_manufacturer(value: object) -> str:
    """CT 장비 제조사 문자열을 Siemens/Philips/GE/Other 4개 범주로 매핑."""
    s = str(value).upper()
    if any(token in s for token in ["SOMATOM", "SENSATION", "SIEMENS"]):
        return "Siemens"
    if any(token in s for token in ["INGENUITY", "ICT", "PHILIPS"]):
        return "Philips"
    if any(token in s for token in ["REVOLUTION", "LIGHTSPEED", "GE"]):
        return "GE"
    return "Other"


def parse_z_len(value: object) -> float:
    """"a-b" 형태 문자열(z축 스캔 범위)에서 두 숫자를 파싱해 그 차이(스캔 길이)를 계산."""
    nums = re.findall(r"-?\d+\.?\d*", str(value))
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        return abs(b - a)
    return np.nan


def row_norm(x: np.ndarray) -> np.ndarray:
    """각 행을 자기 자신의 평균으로 나눠 정규화."""
    m = np.nanmean(x, axis=1, keepdims=True)
    m[~np.isfinite(m) | (m == 0)] = 1.0
    return x / m


def z_rows(x: np.ndarray) -> np.ndarray:
    """각 행을 자기 자신의 평균/표준편차로 z-표준화."""
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return (x - m) / s


def linear_residual(x: np.ndarray) -> np.ndarray:
    """곡선의 양 끝점을 잇는 직선을 빼서 전반적 기울기 성분을 제거한 잔차를 계산."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    line = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    return x - line


def derivative(x: np.ndarray) -> np.ndarray:
    """1차 도함수를 계산 (길이를 맞추기 위해 첫 값을 복제)."""
    d = np.diff(x, axis=1)
    return np.column_stack([d[:, :1], d])


def load_dataset(name: str) -> dict:
    """엑셀에서 원시 AEC_128 곡선, 라벨, 제조사·스캐너 관련 메타데이터(크롭 슬라이스 수, z축 길이 등)를 함께 읽어옴."""
    path = DATA_DIR / f"{name}.xlsx"
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec_df = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(aec_df)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    smi = tama / np.square(height_m)
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    company = meta["Manufacturer"].map(company_from_manufacturer).to_numpy()
    out = meta.copy()
    out["cohort"] = name
    out["company"] = company
    out["low_smi"] = y
    out["smi_calc"] = smi
    out["n_slices_cropped"] = pd.to_numeric(aec_df["n_slices_cropped"], errors="coerce")
    out["z_len"] = aec_df["z_range"].map(parse_z_len)
    return {"name": name, "meta": out, "raw": raw, "y": y, "company": company}


def transforms(raw: np.ndarray) -> dict[str, np.ndarray]:
    """평활화된 원시곡선으로부터 6가지 표현(평활원시/평균정규화/로그중심화/직선잔차/기울기/곡률)을 만듦."""
    raw_s = ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")
    norm = row_norm(raw_s)
    log_centered = np.log(np.clip(norm, 1e-8, None))
    log_centered = log_centered - log_centered.mean(axis=1, keepdims=True)
    resid_z = z_rows(linear_residual(norm))
    slope_z = z_rows(derivative(norm))
    curv_z = z_rows(derivative(derivative(norm)))
    return {
        "raw_smoothed": raw_s,
        "patient_mean_norm": norm,
        "log_centered_norm": log_centered,
        "linear_residual_z": resid_z,
        "slope_z": slope_z,
        "curvature_z": curv_z,
    }


def pooled_company_harmonize(x: np.ndarray, company: np.ndarray) -> np.ndarray:
    """전체(두 코호트 합친) 데이터에서 회사별/전체 평균 템플릿을 만들어, 각 환자 곡선의 회사 평균을 전체 평균으로 치환(harmonize)."""
    keep = company != "Other"
    global_template = x[keep].mean(axis=0)
    templates = {c: x[company == c].mean(axis=0) for c in sorted(np.unique(company[keep]))}
    out = np.empty_like(x)
    for i, c in enumerate(company):
        out[i] = x[i] - templates.get(c, global_template) + global_template
    return out


def company_balanced_means(x: np.ndarray, cohort: np.ndarray, company: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """두 코호트 모두에 존재하는 공통 제조사만 골라, 각 코호트의 "제조사 균형" 평균 곡선(제조사별 평균의 평균)을 계산 — 제조사 구성비 차이로 인한 코호트 간 편향을 제거."""
    common = sorted(
        set(company[(cohort == "g1090") & (company != "Other")]).intersection(
            set(company[(cohort == "sdata") & (company != "Other")])
        )
    )
    means = {}
    for site in ["g1090", "sdata"]:
        parts = []
        for comp in common:
            mask = (cohort == site) & (company == comp)
            if np.any(mask):
                parts.append(x[mask].mean(axis=0))
        means[site] = np.vstack(parts).mean(axis=0)
    return means["g1090"], means["sdata"], common


def mean_gap_metrics(xg: np.ndarray, xs: np.ndarray) -> dict:
    """두 코호트 평균 곡선의 차이(절대평균/RMS/최대/환자내 표준편차로 표준화한 평균)를 계산 — 코호트 간 "격차" 크기를 재는 여러 지표."""
    diff = xs.mean(axis=0) - xg.mean(axis=0)
    pooled_sd = np.sqrt((xg.var(axis=0, ddof=1) + xs.var(axis=0, ddof=1)) / 2.0)
    pooled_sd[~np.isfinite(pooled_sd) | (pooled_sd == 0)] = 1.0
    return {
        "mean_abs_gap": float(np.mean(np.abs(diff))),
        "rms_gap": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs_gap": float(np.max(np.abs(diff))),
        "mean_standardized_abs_gap": float(np.mean(np.abs(diff) / pooled_sd)),
    }


def perm_p_mean_gap(x: np.ndarray, site: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """코호트 라벨을 무작위로 섞어 RMS 격차의 순열분포를 만들고, 관측된 격차가 우연히 나올 확률(p값)을 계산."""
    obs = mean_gap_metrics(x[site == "g1090"], x[site == "sdata"])["rms_gap"]
    n_g = int(np.sum(site == "g1090"))
    idx = np.arange(len(site))
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(idx)
        fake_g = idx[:n_g]
        fake_s = idx[n_g:]
        stat = mean_gap_metrics(x[fake_g], x[fake_s])["rms_gap"]
        ge += stat >= obs - 1e-15
    return obs, float((ge + 1) / (N_PERM + 1))


def auc_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUC를 계산하고(0.5 미만이면 부호를 뒤집어 0.5 이상으로 맞춤) Mann-Whitney U 검정으로 유의성을 함께 반환."""
    auc = float(roc_auc_score(y, score))
    if auc < 0.5:
        score = -score
        auc = 1.0 - auc
    p = float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)
    return auc, p


def site_classifier_auc(x: np.ndarray, site_y: np.ndarray, name: str) -> dict:
    """AEC 곡선 특징(PCA+로지스틱)만으로 어느 기관(g1090 vs sdata) 데이터인지 5-fold 교차검증으로 판별해 AUC를 계산 — 이 표현이 "기관 흔적"을 얼마나 담고 있는지 재는 지표."""
    y = site_y.astype(int)
    n_comp = min(12, x.shape[1], len(y) - 2)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp, random_state=SEED),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=0.5),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(x, y):
        model.fit(x[tr], y[tr])
        scores[va] = model.decision_function(x[va])
    auc, p = auc_p(y, scores)
    return {"feature_set": name, "site_oof_auc": auc, "site_oof_auc_p": p}


def site_classifier_tabular(meta: pd.DataFrame, feature_set: str) -> dict:
    """임상변수/스캐너변수(또는 둘 다)만으로 기관을 판별하는 로지스틱 모델의 5-fold OOF AUC를 계산 — AEC 곡선이 아닌 "다른 변수들"조차 기관을 얼마나 잘 맞히는지 확인하는 비교 기준선."""
    y = (meta["cohort"].to_numpy() == "sdata").astype(int)
    num_cols = []
    cat_cols = []
    if feature_set in {"clinical", "clinical_scanner"}:
        num_cols += ["PatientAge", "Height", "Weight", "BMI", "TAMA", "IMATA"]
        cat_cols += ["PatientSex"]
    if feature_set in {"scanner", "clinical_scanner"}:
        num_cols += ["kVp", "mAs", "n_slices_cropped", "z_len"]
        cat_cols += ["company"]
    x_num = meta[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=meta.index)
    med = x_num.median(numeric_only=True).fillna(0.0)
    x_num = x_num.fillna(med).fillna(0.0)
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x_cat = enc.fit_transform(meta[cat_cols].astype(str))
        x = np.column_stack([x_num.to_numpy(dtype=float), x_cat])
    else:
        x = x_num.to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=0.5))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(x, y):
        model.fit(x[tr], y[tr])
        scores[va] = model.decision_function(x[va])
    auc, p = auc_p(y, scores)
    return {"feature_set": feature_set, "site_oof_auc": auc, "site_oof_auc_p": p}


def cohort_summary(meta: pd.DataFrame) -> pd.DataFrame:
    """연속형 변수는 평균±표준편차와 Mann-Whitney p값, 범주형 변수는 교차표와 카이제곱 p값으로 두 코호트를 비교하는 요약표(Table 1 스타일)를 만듦."""
    rows = []
    cont = ["PatientAge", "Height", "Weight", "BMI", "TAMA", "IMATA", "SMI", "smi_calc", "kVp", "mAs", "n_slices_cropped", "z_len"]
    for col in cont:
        g = pd.to_numeric(meta.loc[meta["cohort"] == "g1090", col], errors="coerce")
        s = pd.to_numeric(meta.loc[meta["cohort"] == "sdata", col], errors="coerce")
        if len(g.dropna()) >= 2 and len(s.dropna()) >= 2:
            p = stats.mannwhitneyu(g.dropna(), s.dropna(), alternative="two-sided").pvalue
        else:
            p = np.nan
        rows.append(
            {
                "variable": col,
                "g1090": f"{g.mean():.3f} ({g.std(ddof=1):.3f})",
                "sdata": f"{s.mean():.3f} ({s.std(ddof=1):.3f})",
                "p_value": float(p),
            }
        )
    for col in ["low_smi", "PatientSex", "company"]:
        tab = pd.crosstab(meta["cohort"], meta[col])
        chi = stats.chi2_contingency(tab)[1]
        rows.append({"variable": col, "g1090": dict(tab.loc["g1090"]), "sdata": dict(tab.loc["sdata"]), "p_value": float(chi)})
    rows.insert(0, {"variable": "n", "g1090": int(np.sum(meta["cohort"] == "g1090")), "sdata": int(np.sum(meta["cohort"] == "sdata")), "p_value": np.nan})
    return pd.DataFrame(rows)


def company_gap_table(x: np.ndarray, cohort: np.ndarray, company: np.ndarray) -> pd.DataFrame:
    """같은 제조사 안에서도 g1090 vs sdata 간 곡선 격차가 남아있는지, 제조사별로 격차 지표를 계산 — "장비가 아니라 기관 자체"의 차이가 있는지 확인."""
    rows = []
    for comp in sorted(set(company) - {"Other"}):
        g = x[(cohort == "g1090") & (company == comp)]
        s = x[(cohort == "sdata") & (company == comp)]
        if len(g) == 0 or len(s) == 0:
            continue
        row = {"company": comp, "g1090_n": len(g), "sdata_n": len(s)}
        row.update(mean_gap_metrics(g, s))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_abs_gap")


def plot_audit(
    g: dict,
    s: dict,
    combined: dict[str, np.ndarray],
    site: np.ndarray,
    company: np.ndarray,
    distance_df: pd.DataFrame,
    site_auc_df: pd.DataFrame,
) -> None:
    """원시/정규화 곡선 비교, 같은 제조사 내 기관 비교, 여러 전처리별 코호트 차이 곡선, 표준화된 격차
    막대그래프, 기관 판별 AUC 막대그래프까지 6패널 종합 진단 그래프를 그려 PNG로 저장."""
    colors = {"g1090": "#2F6B9A", "sdata": "#C54E2C"}
    z = np.arange(1, 129)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    for name, ax, ylabel in [
        ("raw_smoothed", axes[0, 0], "Raw AEC"),
        ("patient_mean_norm", axes[0, 1], "AEC / patient mean"),
    ]:
        x = combined[name]
        for site_name in ["g1090", "sdata"]:
            mask = site == site_name
            m = x[mask].mean(axis=0)
            se = x[mask].std(axis=0, ddof=1) / np.sqrt(mask.sum())
            ax.plot(z, m, color=colors[site_name], lw=2.0, label=site_name)
            ax.fill_between(z, m - 1.96 * se, m + 1.96 * se, color=colors[site_name], alpha=0.12, lw=0)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xlabel("AEC_128 point")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    ax = axes[0, 2]
    x = combined["patient_mean_norm"]
    for comp, ls in [("Siemens", "-"), ("GE", "--"), ("Philips", ":")]:
        for site_name in ["g1090", "sdata"]:
            mask = (site == site_name) & (company == comp)
            if np.any(mask):
                ax.plot(z, x[mask].mean(axis=0), color=colors[site_name], ls=ls, lw=1.8, label=f"{site_name} {comp}")
    ax.set_title("same company, different institution", loc="left", fontweight="bold")
    ax.set_xlabel("AEC_128 point")
    ax.set_ylabel("AEC / patient mean")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 0]
    for name, c in [("patient_mean_norm", "#555555"), ("company_harmonized_norm", "#009E73"), ("company_balanced_norm", "#8B5A2B")]:
        row = distance_df[distance_df["transform"].eq(name)].iloc[0]
        ax.plot(z, combined[name][site == "sdata"].mean(axis=0) - combined[name][site == "g1090"].mean(axis=0), lw=2.0, color=c, label=f"{name}: rms {row['rms_gap']:.3g}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("sdata minus g1090 mean curve", loc="left", fontweight="bold")
    ax.set_xlabel("AEC_128 point")
    ax.set_ylabel("difference")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    top = distance_df[distance_df["transform"].isin(["raw_smoothed", "patient_mean_norm", "log_centered_norm", "company_harmonized_norm", "company_balanced_norm", "linear_residual_z", "slope_z", "curvature_z"])].copy()
    ax.barh(top["transform"], top["mean_standardized_abs_gap"], color="#6C8EBF")
    ax.set_title("site mean gap / within-patient SD", loc="left", fontweight="bold")
    ax.set_xlabel("mean standardized absolute gap")

    ax = axes[1, 2]
    show = site_auc_df.sort_values("site_oof_auc", ascending=True)
    ax.barh(show["feature_set"], show["site_oof_auc"], color="#A75D5D")
    ax.axvline(0.5, color="black", lw=0.8)
    ax.set_xlim(0.45, 1.0)
    ax.set_title("How easily can site be predicted?", loc="left", fontweight="bold")
    ax.set_xlabel("5-fold OOF site AUC")

    fig.suptitle("Why do Gangnam and Sinchon AEC curves differ? Site-effect decomposition", fontsize=15, fontweight="bold")
    fig.savefig(OUT_DIR / "aec_site_difference_decomposition.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: g1090과 sdata 두 코호트의 AEC 곡선은 왜 다른가? — 임상
    구성비 차이 때문인지, 장비(제조사) 차이 때문인지, 아니면 그걸로도 설명 안 되는 순수한
    "기관 효과"가 남아있는지 분해해서 진단):

    1. g1090/sdata를 로드하며 제조사·스캔 관련 메타데이터(kVp, mAs, 크롭 슬라이스 수, z축 길이)도 함께 추출.
    2. cohort_summary로 임상변수·라벨·성별·제조사 분포를 Table 1 스타일로 비교(연속형은 Mann-Whitney,
       범주형은 카이제곱)해 CSV로 저장.
    3. transforms로 6가지 곡선 표현을 만들고, 회사보정(pooled_company_harmonize)과 "제조사 구성비를
       맞춘" 균형 평균(company_balanced_means) 버전도 추가해, 각 표현마다 perm_p_mean_gap으로
       코호트 간 평균곡선 격차와 순열검정 p값을 계산 — 표준화된 격차가 큰 순으로 정렬해 CSV 저장.
    4. company_gap_table로 같은 제조사 안에서도 기관 간 격차가 남는지 확인해 CSV로 저장.
    5. site_classifier_tabular/site_classifier_auc로 (a) 임상변수만, (b) 스캐너변수만, (c) 둘 다,
       (d) 여러 AEC 표현 각각으로 "이 데이터가 g1090인지 sdata인지" 맞히는 판별 모델을 5-fold로
       학습해 AUC를 비교 — 어떤 정보가 기관 흔적을 가장 많이/적게 담고 있는지 확인.
    6. 위 모든 결과를 6패널 종합 진단 그래프로 시각화해 PNG로 저장.
    7. 코호트 요약, 곡선 격차, 제조사 내 격차, 기관 판별 결과를 모두 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    g = load_dataset("g1090")
    s = load_dataset("sdata")
    meta = pd.concat([g["meta"], s["meta"]], axis=0, ignore_index=True)
    site = meta["cohort"].to_numpy()
    site_y = (site == "sdata").astype(int)
    company = meta["company"].to_numpy()
    raw = np.vstack([g["raw"], s["raw"]])
    trans = transforms(raw)
    trans["company_harmonized_norm"] = pooled_company_harmonize(trans["patient_mean_norm"], company)
    cb_g, cb_s, common_companies = company_balanced_means(trans["patient_mean_norm"], site, company)
    cb = trans["patient_mean_norm"].copy()
    cb[site == "g1090"] = cb_g
    cb[site == "sdata"] = cb_s
    trans["company_balanced_norm"] = cb

    summary_df = cohort_summary(meta)
    summary_df.to_csv(OUT_DIR / "cohort_site_summary.csv", index=False)

    distance_rows = []
    for name, x in trans.items():
        if name == "company_balanced_norm":
            diff = cb_s - cb_g
            metrics = {
                "mean_abs_gap": float(np.mean(np.abs(diff))),
                "rms_gap": float(np.sqrt(np.mean(np.square(diff)))),
                "max_abs_gap": float(np.max(np.abs(diff))),
                "mean_standardized_abs_gap": np.nan,
            }
            rms = metrics["rms_gap"]
            p = np.nan
        else:
            rms, p = perm_p_mean_gap(x.copy(), site.copy(), rng)
            metrics = mean_gap_metrics(x[site == "g1090"], x[site == "sdata"])
        row = {"transform": name, "rms_gap_perm_stat": rms, "rms_gap_perm_p": p}
        row.update(metrics)
        distance_rows.append(row)
    distance_df = pd.DataFrame(distance_rows).sort_values("mean_standardized_abs_gap", ascending=False)
    distance_df.to_csv(OUT_DIR / "site_curve_distance_metrics.csv", index=False)

    cg = company_gap_table(trans["patient_mean_norm"], site, company)
    cg.to_csv(OUT_DIR / "site_gap_within_company_patient_mean_norm.csv", index=False)

    site_auc_rows = [
        site_classifier_tabular(meta, "clinical"),
        site_classifier_tabular(meta, "scanner"),
        site_classifier_tabular(meta, "clinical_scanner"),
    ]
    for name in ["raw_smoothed", "patient_mean_norm", "log_centered_norm", "company_harmonized_norm", "linear_residual_z", "slope_z", "curvature_z"]:
        site_auc_rows.append(site_classifier_auc(trans[name], site_y, name))
    site_auc_df = pd.DataFrame(site_auc_rows).sort_values("site_oof_auc", ascending=False)
    site_auc_df.to_csv(OUT_DIR / "site_classifier_auc.csv", index=False)

    plot_audit(g, s, trans, site, company, distance_df, site_auc_df)

    print("cohort summary")
    print(summary_df.to_string(index=False))
    print("\ncurve distance")
    print(distance_df.to_string(index=False))
    print("\nwithin-company norm gap")
    print(cg.to_string(index=False))
    print("\nsite classifier")
    print(site_auc_df.to_string(index=False))
    print("\ncommon_companies", common_companies)
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    # g1090/sdata 두 코호트의 AEC 곡선 차이를 임상 구성비, 장비(제조사) 차이, 순수 기관 효과로
    # 분해해서 진단하는 파이프라인을 실행한다.
    main()
