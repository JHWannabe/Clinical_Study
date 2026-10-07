from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# data/{gangnam,sinchon}_landmark_filtered.xlsx의 slice별 체성분 곡선(VAT/SAT/TAMA, cm2)으로 AUC 향상 연구.
# baseline = clinic4(성별/나이/신장/체중), Model1 = baseline + liver_dome~inferior_pubic_margin 구간 VAT 합,
# Model2~ = Model1 + landmark 체성분 비율 또는 FPCA 점수(한 번에 한 가지 추가 -> 어떤 feature가 AUC를 올리는지 분리).
# 같은 landmark 필터 코호트(라벨/체성분 결측 제외) 안에서 비교하므로 환자 구성 차이는 통제된다.
# 내부 평가는 5-fold CV가 아니라 gangnam train/valid/test = 7/1/2 hold-out(core.clinic4_logistic_regression.run_disease와 같은 프로토콜:
# 1:1 balanced -> stratified split -> valid를 단일 fold로 GridSearchCV -> test AUC), 외부 = trainval로 재학습한 frozen 모델을 sinchon(balanced)에 적용.
# baseline/Model1 대비 DeLong 검정을 덧붙인다.

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

from clinic4_logistic_regression import (CLINICAL_COLS, DATA_DIR, DISEASES, PARAM_GRID, SEED, TEST_RATIO, TRAIN_RATIO, VALID_RATIO, balanced_idx,
                                         f3, format_floats, save_sheet)
from delong_utils import bh_fdr, delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")

OUT_XLSX = "landmark/landmark_vat_auc.xlsx"
TISSUES = ["VAT", "SAT", "TAMA"]  # 2026-10-02 데이터셋 변경: NAMA/LAMA/IMATA가 TAMA(근육 전체)로 통합됨
MUSCLE = ["TAMA"]
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center",
      "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
START, END, N_POINTS, N_PC = "liver_dome", "inferior_pubic_margin", 128, 3
SEGMENTS = [("liver_dome", "L1_center"), ("L1_center", "L3_center"), ("L3_center", "L5_center")]  # 4번째(L5~pubis)는 나머지
QC_EXCLUDE_DISORDERED = True  # 2026-10-06 데이터셋: gangnam 7명/sinchon 16명이 해당. False로 두면 전부 포함
QC_ORDER = ["T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center", "S1_center",
            "femoral_head_center", "inferior_pubic_margin"]


def log_ratio(a, b):
    return np.log((a + 1) / (b + 1))  # 0 방지(비율 feature는 전부 log((a+1)/(b+1)))


# 환자 한 명의 slice 곡선(1-indexed 컬럼 prefix_1..)에서 구간합/landmark 값/liver~pubis 128점 리샘플 곡선을 만든다
def load_cohort(cohort: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    path = DATA_DIR / f"{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    curves = {t: pd.read_excel(path, sheet_name=f"{t}_total").set_index("PatientID").filter(regex=f"^{t}_\\d+$") for t in TISSUES}
    ids = [p for p in meta["PatientID"] if all(p in c.index for c in curves.values())]
    meta = meta[meta["PatientID"].isin(ids)].dropna(subset=CLINICAL_COLS + DISEASES).reset_index(drop=True)
    # QC: T10~S1~femoral_head~pubis slice가 엄격히 증가하지 않는 환자(anchor가 잘못 잡힌 것)는 제외. liver_dome은 T10보다 뒤일 수 있어 순서 검사에서 뺀다
    if QC_EXCLUDE_DISORDERED:
        order = lm.loc[meta["PatientID"], [f"{a}_slice" for a in QC_ORDER]].to_numpy()
        bad = (np.diff(order, axis=1) <= 0).any(axis=1) | (lm.loc[meta["PatientID"], f"{START}_slice"].to_numpy() >= lm.loc[meta["PatientID"], f"{END}_slice"].to_numpy())
        print(f"[{cohort}] QC 제외(landmark 순서 이상): {int(bad.sum())}명")
        meta = meta[~bad].reset_index(drop=True)
    ids = meta["PatientID"].tolist()

    feat = {}
    resampled = {t: [] for t in TISSUES}
    for t in TISSUES:
        c = curves[t].loc[ids].to_numpy(float)
        for i, p in enumerate(ids):
            row, l = c[i], lm.loc[p]
            s, e = int(l[f"{START}_slice"]), int(l[f"{END}_slice"])  # 1-indexed slice
            resampled[t].append(np.interp(np.linspace(s, e, N_POINTS), np.arange(1, len(row) + 1), np.nan_to_num(row)))
            feat.setdefault(f"{t}_sum", []).append(np.nansum(row[s - 1:e]))
            for a, b in SEGMENTS:
                feat.setdefault(f"{t}_{a}~{b}", []).append(np.nansum(row[int(l[f"{a}_slice"]) - 1:int(l[f"{b}_slice"])]))
            for a in LM:
                feat.setdefault(f"{t}@{a}", []).append(np.nan_to_num(row[int(l[f"{a}_slice"]) - 1]))
    df = pd.concat([meta, pd.DataFrame({k: np.array(v) for k, v in feat.items()})], axis=1)
    print(f"[{cohort}] landmark 환자 {len(lm)} -> 곡선/라벨 보유 {len(df)}명 ({', '.join(f'{d}={int(df[d].sum())}' for d in DISEASES)})")
    return df, {t: np.array(v) for t, v in resampled.items()}


# 비율 feature(로그). 전부 liver~pubis 구간합 또는 landmark 단면에서 계산
def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    n = {}
    n["VAT/SAT"] = log_ratio(df["VAT_sum"], df["SAT_sum"])
    n["VAT/TAMA"] = log_ratio(df["VAT_sum"], df["TAMA_sum"])
    n["TAMA/(TAMA+VAT+SAT)"] = log_ratio(df["TAMA_sum"], df["VAT_sum"] + df["SAT_sum"])
    for a, b in SEGMENTS:  # VAT가 어느 구간에 몰려 있는지(구간 VAT / 전체 VAT)
        n[f"VATfrac_{a}~{b}"] = df[f"VAT_{a}~{b}"] / df["VAT_sum"].clip(lower=1e-6)
    for a in ("L1_center", "L3_center", "L5_center"):
        n[f"VAT/SAT@{a}"] = log_ratio(df[f"VAT@{a}"], df[f"SAT@{a}"])
    for a in ("L1_center", "L5_center", "femoral_head_center"):  # landmark 간 비율(L3 대비)
        for t in ("VAT", "SAT", "TAMA"):
            n[f"{t}@{a}/L3"] = log_ratio(df[f"{t}@{a}"], df[f"{t}@L3_center"])
    return pd.concat([df, pd.DataFrame(n)], axis=1)


# 모델명 -> Model1 위에 더할 feature 컬럼(비율은 add_ratios 컬럼, FPCA는 fpca_{조직}_pc{i})
def model_sets() -> dict[str, list[str]]:
    fp = lambda *ts: [f"fpca_{t}_pc{i}" for t in ts for i in range(1, N_PC + 1)]
    seg_cols = [f"VATfrac_{a}~{b}" for a, b in SEGMENTS]
    vsr_lm = [f"VAT/SAT@{a}" for a in ("L1_center", "L3_center", "L5_center")]
    lm_ratio = [f"{t}@{a}/L3" for a in ("L1_center", "L5_center", "femoral_head_center") for t in ("VAT", "SAT", "TAMA")]
    # all_ratio에는 vsr_lm을 넣지 않는다: VAT/SAT@L1 - VAT/SAT@L3 = VAT@L1/L3 - SAT@L1/L3 라서 lm_ratio와 완전 공선
    all_ratio = ["VAT/SAT", "VAT/TAMA", "TAMA/(TAMA+VAT+SAT)"] + seg_cols + lm_ratio
    return {
        "baseline": [], "Model1": [],
        "Model2 +VAT/SAT": ["VAT/SAT"],
        "Model3 +VAT/TAMA": ["VAT/TAMA"],
        "Model4 +TAMA/(TAMA+fat)": ["TAMA/(TAMA+VAT+SAT)"],
        "Model5 +VAT segment fraction": seg_cols,
        "Model6 +VAT/SAT@L1,L3,L5": vsr_lm,
        "Model7 +VAT,SAT,TAMA landmark/L3": lm_ratio,
        "Model8 +all ratios": all_ratio,
        "Model9 +FPCA(VAT)": fp("VAT"),
        "Model10 +FPCA(SAT)": fp("SAT"),
        "Model11 +FPCA(TAMA)": fp("TAMA"),
        "Model12 +FPCA(VAT,SAT)": fp("VAT", "SAT"),
        "Model13 +FPCA(all3)": fp(*TISSUES),
    }


# PCA는 라벨을 쓰지 않는 비지도 변환이라 gangnam 전체로 fit, sinchon엔 frozen transform(기존 add_curve_features와 동일 방침)
def add_fpca(df, curves, df_ext, curves_ext) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, PCA]]:
    pcas = {}
    for t in TISSUES:
        pcas[t] = PCA(N_PC, random_state=SEED).fit(curves[t])
        for d, c in ((df, curves[t]), (df_ext, curves_ext[t])):
            sc = pcas[t].transform(c)
            for i in range(N_PC):
                d[f"fpca_{t}_pc{i + 1}"] = sc[:, i]
    return df, df_ext, pcas


# balanced 부분집합 y에 대한 train/valid/test = 7/1/2 stratified split 인덱스(run_disease와 동일)
def holdout_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_trainval, idx_test = train_test_split(np.arange(len(y)), test_size=TEST_RATIO, random_state=SEED, stratify=y)
    idx_train, idx_valid = train_test_split(idx_trainval, test_size=VALID_RATIO / (TRAIN_RATIO + VALID_RATIO), random_state=SEED, stratify=y[idx_trainval])
    return idx_train, idx_valid, idx_test


# 컬럼 cols만 써서: valid를 단일 fold로 GridSearchCV -> train 모델의 test/valid 점수, trainval 재학습 frozen 모델의 외부 점수
def holdout_fit(x: np.ndarray, y: np.ndarray, split, cols, x_ext: np.ndarray) -> dict:
    idx_train, idx_valid, idx_test = split
    idx_tv = np.concatenate([idx_train, idx_valid])
    fold = np.concatenate([np.full(len(idx_train), -1), np.zeros(len(idx_valid), dtype=int)])
    grid = GridSearchCV(LogisticRegression(max_iter=2000), PARAM_GRID, scoring="roc_auc", cv=PredefinedSplit(fold)).fit(x[idx_tv][:, cols], y[idx_tv])
    best = grid.best_params_
    model = LogisticRegression(max_iter=2000, **best).fit(x[idx_train][:, cols], y[idx_train])
    final = LogisticRegression(max_iter=2000, **best).fit(x[idx_tv][:, cols], y[idx_tv])
    return {"valid_score": model.predict_proba(x[idx_valid][:, cols])[:, 1], "test_score": model.predict_proba(x[idx_test][:, cols])[:, 1],
            "ext_score": final.predict_proba(x_ext[:, cols])[:, 1], "best_params": best}


# disease 하나 + feature set(x/df: gangnam 전체, x_ext/df_ext: sinchon 전체)의 hold-out 평가. 반환 = (summary 행, 예측 df)
# rng 호출 순서(gangnam balanced_idx -> sinchon balanced_idx)는 run_disease와 같게 유지
def run_disease_holdout(disease: str, model_name: str, x_full: np.ndarray, df: pd.DataFrame, x_ext_full: np.ndarray, df_ext: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    y_full, pid_full = df[disease].to_numpy(int), df["PatientID"].to_numpy()
    idx = balanced_idx(y_full, rng)
    x, y, pid = x_full[idx], y_full[idx], pid_full[idx]
    y_ext_full, pid_ext_full = df_ext[disease].to_numpy(int), df_ext["PatientID"].to_numpy()
    idx_ext = balanced_idx(y_ext_full, rng)
    x_ext, y_ext, pid_ext = x_ext_full[idx_ext], y_ext_full[idx_ext], pid_ext_full[idx_ext]
    split = holdout_split(y)
    out = holdout_fit(x, y, split, list(range(x.shape[1])), x_ext)
    idx_train, idx_valid, idx_test = split
    row = {"disease": disease, "model": model_name, "n_train": len(idx_train), "n_valid": len(idx_valid), "n_test": len(idx_test), "n_external": len(y_ext),
           "valid_auc": roc_auc_score(y[idx_valid], out["valid_score"]), "internal_auc": roc_auc_score(y[idx_test], out["test_score"]),
           "external_auc": roc_auc_score(y_ext, out["ext_score"]), "best_C": out["best_params"]["C"], "best_penalty": out["best_params"]["penalty"],
           "best_class_weight": out["best_params"]["class_weight"]}
    pred = pd.concat([pd.DataFrame({"disease": disease, "model": model_name, "cohort": "gangnam", "patient_id": pid[idx_test], "y": y[idx_test],
                                    "score": out["test_score"], "fold": -1}),
                      pd.DataFrame({"disease": disease, "model": model_name, "cohort": "sinchon", "patient_id": pid_ext, "y": y_ext,
                                    "score": out["ext_score"], "fold": -1})], ignore_index=True)
    print(f"[{disease}/{model_name}] n_train/valid/test={len(idx_train)}/{len(idx_valid)}/{len(idx_test)} valid={f3(row['valid_auc'])} "
          f"test={f3(row['internal_auc'])} external={f3(row['external_auc'])} (n_ext={len(y_ext)})")
    return row, pred


def matrix(df: pd.DataFrame, cols: list[str], scaler: StandardScaler | None = None):
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_COLS + cols].to_numpy(float)
    scaler = scaler or StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


# 질병별 OOF/외부 예측에 대해 reference 모델 대비 paired DeLong(BH-FDR은 모델 전체에서 질병·cohort 별로 보정)
def delong_vs(preds: pd.DataFrame, ref: str) -> pd.DataFrame:
    rows = []
    for (disease, cohort), g in preds.groupby(["disease", "cohort"]):
        piv = g.pivot(index="patient_id", columns="model", values="score")
        y = g.drop_duplicates("patient_id").set_index("patient_id")["y"].loc[piv.index].to_numpy(int)
        for m in piv.columns.drop(ref):
            r = delong_paired_auc_test(y, piv[ref].to_numpy(), piv[m].to_numpy())
            rows.append({"ref": ref, "disease": disease, "cohort": cohort, "model": m, **r})
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    if "p_value" in out:  # 예측이 동일한 모델 쌍은 분산 0 -> p=NaN. NaN이 BH 보정 전체로 번지지 않게 유효한 p만 보정
        ok = out["p_value"].notna()
        out.loc[ok, "p_fdr"] = bh_fdr(out.loc[ok, "p_value"].to_numpy())
    return out


def main() -> None:
    (df, curves), (df_ext, curves_ext) = load_cohort("gangnam"), load_cohort("sinchon")
    df, df_ext = add_ratios(df), add_ratios(df_ext)
    df, df_ext, _ = add_fpca(df, curves, df_ext, curves_ext)

    rows, preds = [], []
    for name, extra in model_sets().items():
        cols = [] if name == "baseline" else ["VAT_sum"] + extra
        x, scaler = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, scaler)
        for disease in DISEASES:
            row, pred = run_disease_holdout(disease, name, x, df, x_ext, df_ext)
            rows.append(row)
            preds.append(pred)
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)

    stats = pd.concat([delong_vs(preds, "baseline"), delong_vs(preds, "Model1")], ignore_index=True)
    save_sheet(preds, OUT_XLSX, "predictions")
    save_sheet(format_floats(summary), OUT_XLSX, "summary")
    save_sheet(format_floats(stats), OUT_XLSX, "delong")

    cols = ["internal_auc", "external_auc"]
    piv = summary.pivot(index="model", columns="disease", values=cols)
    order = [(m, d) for m in cols for d in DISEASES]
    print("\nAUC (internal=5-fold hold-out test | external=sinchon)")
    print(piv.loc[list(model_sets())][order].map(f3).to_string())
    for ref in ("baseline", "Model1"):
        delta = piv.sub(piv.loc[ref], axis=1).loc[list(model_sets())][order]
        print(f"\ndelta vs {ref}\n" + delta.map(f3).to_string())


if __name__ == "__main__":
    main()
