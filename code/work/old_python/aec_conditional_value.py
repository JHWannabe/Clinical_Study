from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATA_DIR = Path(__file__).resolve().parent / "data_cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec_conditional_value"
SEED = 20260629
RNG = np.random.default_rng(SEED)


def aec_columns(df: pd.DataFrame) -> list[str]:
    """데이터프레임에서 'aec_'로 시작하는 컬럼명을 뽑아 번호 순서로 정렬."""
    return sorted([c for c in df.columns if str(c).startswith("aec_")], key=lambda c: int(str(c).split("_")[1]))


def matrix_from_sheet(df: pd.DataFrame) -> np.ndarray:
    """AEC 컬럼들을 숫자 행렬로 변환하고, 결측/비정상 값을 열별(불가능하면 전체) 중앙값으로 대체."""
    x = df[aec_columns(df)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    global_med = float(np.nanmedian(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    col_med = np.nanmedian(x, axis=0)
    col_med[~np.isfinite(col_med)] = global_med
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(col_med, np.where(bad)[1])
    x[~np.isfinite(x)] = global_med
    return x


def resample_rows(x: np.ndarray, n: int = 128) -> np.ndarray:
    """행렬의 각 행(신호)을 선형보간으로 길이 n(기본 128)으로 리샘플링."""
    old = np.linspace(0.0, 1.0, x.shape[1])
    new = np.linspace(0.0, 1.0, n)
    return np.vstack([np.interp(new, old, row) for row in x])


def row_norm(x: np.ndarray) -> np.ndarray:
    """각 행을 자기 자신의 평균으로 나눠 스케일을 정규화 (평균이 0/비정상이면 1로 대체)."""
    m = np.mean(x, axis=1)
    m[~np.isfinite(m) | (m == 0)] = 1.0
    return x / m[:, None]


def load_dataset(path: Path) -> dict:
    """엑셀 파일(metadata/aec_128/aec_cropped)을 읽어 라벨(저근감소증 여부)과 정규화된 AEC 곡선(128구간+크롭 128구간 결합)을 구성."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    a128 = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl")
    crop = pd.read_excel(path, sheet_name="aec_cropped", engine="openpyxl")
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    a128_mat = resample_rows(matrix_from_sheet(a128), 128)
    crop_mat = resample_rows(matrix_from_sheet(crop), 128)
    direct_curve = np.column_stack([row_norm(a128_mat) - 1.0, row_norm(crop_mat) - 1.0])
    return {"meta": meta, "y": y, "aec": direct_curve, "sex": sex, "smi": smi}


def clinical_matrix(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """나이/키/몸무게/성별로 임상 변수 행렬을 만들고, train 기준 결측 대체 및 표준화를 test에도 동일 적용."""
    names = ["PatientAge", "Height", "Weight", "sex_M"]
    def raw(meta: pd.DataFrame) -> np.ndarray:
        x = np.column_stack(
            [
                pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float),
                (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float),
            ]
        )
        return x

    tr = raw(train_meta)
    te = raw(test_meta)
    med = np.nanmedian(tr, axis=0)
    tr = np.where(np.isfinite(tr), tr, med)
    te = np.where(np.isfinite(te), te, med)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (tr - mu) / sd, (te - mu) / sd, names


def make_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """클래스 비율을 유지하며 데이터를 k개의 교차검증 폴드 인덱스로 분할."""
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        RNG.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=int) for f in folds]


def clinical_estimator() -> LogisticRegression:
    """임상 변수 전용 로지스틱 회귀 모델(정규화 거의 없음)을 생성."""
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)


def aec_estimator(n_features: int, seed: int) -> Pipeline:
    """결측대체→표준화→상위 특징 선택→선형 SVM으로 이어지는 AEC 전용 분류 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(128, n_features))),
            ("svm", LinearSVC(C=0.2, class_weight="balanced", max_iter=10000, random_state=seed)),
        ]
    )


def score_model(model, x: np.ndarray) -> np.ndarray:
    """모델 종류에 따라 decision_function/predict_proba/predict 중 있는 것으로 점수를 산출."""
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def oof_and_external(model_factory, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """폴드별로 학습해 train의 out-of-fold 예측을 만들고, 전체 train으로 재학습한 모델로 외부 데이터 예측도 함께 반환."""
    oof = np.zeros(len(ytr), dtype=float)
    all_idx = np.arange(len(ytr))
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = model_factory(SEED + fold_id)
        model.fit(xtr[tr_idx], ytr[tr_idx])
        oof[val_idx] = score_model(model, xtr[val_idx])
    final = model_factory(SEED + 99)
    final.fit(xtr, ytr)
    return oof, score_model(final, xte)


def zfit_apply(train_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """train 점수의 평균/표준편차로 train·test 점수를 함께 z-표준화."""
    mu = float(np.mean(train_score))
    sd = float(np.std(train_score))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (train_score - mu) / sd, (test_score - mu) / sd, mu, sd


def threshold_youden(y: np.ndarray, score: np.ndarray) -> float:
    """Youden index(민감도+특이도-1)를 최대화하는 임계값을 후보 점수들 중에서 탐색."""
    order = np.argsort(score)
    candidates = np.unique(score[order])
    best_th = float(candidates[0])
    best_j = -np.inf
    for th in candidates:
        pred = score >= th
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_th = float(th)
    return best_th


def binary_metrics(y: np.ndarray, score: np.ndarray, th: float) -> dict:
    """주어진 임계값에서 TP/FP/FN/TN과 민감도·특이도·PPV·NPV를 계산."""
    pred = score >= th
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "threshold": float(th),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def continuous_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """점수를 로지스틱 회귀로 확률에 재보정한 뒤 AUC/AP/로그손실/Brier를 계산."""
    calibrated = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    # This is used only to map a decision score to probabilities for proper scoring.
    calibrated.fit(score.reshape(-1, 1), y)
    p = calibrated.predict_proba(score.reshape(-1, 1))[:, 1]
    return {
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss_recalibrated_in_sample": float(log_loss(y, p)),
        "brier_recalibrated_in_sample": float(brier_score_loss(y, p)),
    }


def logit_lrt(y: np.ndarray, base_cols: list[np.ndarray], add_cols: list[np.ndarray], labels: list[str]) -> dict:
    """base 변수만 넣은 모델과 add 변수까지 넣은 모델을 각각 적합해 우도비검정(LRT) 통계량과 p-value, 계수를 반환."""
    x0 = sm.add_constant(np.column_stack(base_cols), has_constant="add")
    x1 = sm.add_constant(np.column_stack(base_cols + add_cols), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    stat = 2 * (m1.llf - m0.llf)
    df = x1.shape[1] - x0.shape[1]
    params = dict(zip(["const"] + labels, m1.params))
    pvals = dict(zip(["const"] + labels, m1.pvalues))
    return {
        "lrt_chi2": float(stat),
        "df": int(df),
        "lrt_p": float(stats.chi2.sf(stat, df)),
        "ll_base": float(m0.llf),
        "ll_full": float(m1.llf),
        "params": {k: float(v) for k, v in params.items()},
        "pvalues": {k: float(v) for k, v in pvals.items()},
    }


def bootstrap_metric_deltas(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, th_a: float, th_b: float, n_boot: int = 3000) -> dict:
    """두 점수(a vs b)의 AUC/민감도/특이도/PPV/FP/TP 차이를 부트스트랩으로 반복 추정해 신뢰구간과 단측 p값을 계산."""
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        ma = binary_metrics(yi, score_a[idx], th_a)
        mb = binary_metrics(yi, score_b[idx], th_b)
        rows.append(
            [
                roc_auc_score(yi, score_b[idx]) - roc_auc_score(yi, score_a[idx]),
                mb["sensitivity"] - ma["sensitivity"],
                mb["specificity"] - ma["specificity"],
                mb["ppv"] - ma["ppv"],
                mb["fp"] - ma["fp"],
                mb["tp"] - ma["tp"],
            ]
        )
    arr = np.asarray(rows)
    names = ["delta_auc", "delta_sensitivity", "delta_specificity", "delta_ppv", "delta_fp", "delta_tp"]
    out = {}
    for i, name in enumerate(names):
        vals = arr[:, i]
        out[name] = {
            "mean": float(np.mean(vals)),
            "ci2.5": float(np.quantile(vals, 0.025)),
            "ci97.5": float(np.quantile(vals, 0.975)),
            "p_le_0": float(np.mean(vals <= 0)),
            "p_ge_0": float(np.mean(vals >= 0)),
        }
    return out


def permutation_lrt_p(y: np.ndarray, clinical_z: np.ndarray, gated_aec: np.ndarray, observed_lrt: float, n_perm: int = 5000) -> float:
    """게이트된 AEC 값을 무작위로 섞어 LRT 통계량의 순열분포를 만들고, 관측 통계량 대비 순열 p-value를 계산."""
    stats_perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = RNG.permutation(gated_aec)
        res = logit_lrt(y, [clinical_z], [perm], ["clinical_z", "permuted_gated_aec"])
        stats_perm[i] = res["lrt_chi2"]
    return float((np.sum(stats_perm >= observed_lrt) + 1) / (n_perm + 1))


def group_summary(y: np.ndarray, mask: np.ndarray) -> dict:
    """mask로 선택된 부분집합의 표본수, 이벤트수, 유병률을 계산."""
    n = int(mask.sum())
    e = int(y[mask].sum())
    return {"n": n, "events": e, "prevalence": e / n if n else np.nan}


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상변수 외에 AEC가 추가 정보를 주는가?):

    1. g1090.xlsx(학습)과 sdata.xlsx(외부검증)를 load_dataset으로 읽어 라벨(y)과 AEC 곡선을 만든다.
    2. clinical_matrix로 임상변수(나이/키/몸무게/성별) 행렬을 만들고, make_folds로 5-fold를 나눈다.
    3. 임상변수 단독 모델(clinical_estimator)과 AEC 단독 모델(aec_estimator, SVM)을 각각
       oof_and_external로 학습해, train의 out-of-fold 예측값과 외부(sdata) 예측값을 동시에 얻는다.
       -> out-of-fold를 쓰는 이유: train 데이터로 만든 임계값/게이트가 그 데이터 자체에 과적합되지
          않도록, "그 샘플이 학습에 쓰이지 않았을 때의 예측"으로 임계값을 정하기 위함.
    4. 두 점수를 zfit_apply로 표준화(z-score)하고, threshold_youden으로 임상점수의 "양성 경계"를 찾는다.
    5. 경계 근처의 여성 환자에게만 AEC 점수를 가중치로 얹는 두 가지 게이트를 만든다:
       - female_boundary_gaussian: 경계로부터 거리에 가우시안 가중치를 줘서 부드럽게 AEC를 반영
       - hard_zone_female: 경계 ±0.50 구간 안의 여성에서만 AEC를 딱 잘라 반영
       (단순 스택 모델도 비교군으로 함께 학습)
    6. logit_lrt로 "임상점수만 넣은 모델" vs "임상점수+AEC(또는 게이트) 점수를 넣은 모델"의
       우도비검정(LRT)을 train(oof)과 외부 데이터 양쪽에서 수행해, AEC가 임상변수를 통제한 뒤에도
       독립적으로 결과와 연관되는지를 확인한다. permutation_lrt_p로 순열검정 p-value도 추가 검증.
    7. 외부 데이터에서 "임상 양성/음성 x 게이트 양성/음성" 2x2 재분류표를 만들고, Fisher 정확검정으로
       임상양성군 내에서 게이트가 실제로 고위험군을 더 잘 골라내는지 확인한다.
    8. McNemar류의 대응 이항검정으로, 게이트 적용 시 새로 생기는 위양성(FP)/놓치는 진양성(TP) 개수가
       통계적으로 유의한 변화인지 확인한다.
    9. bootstrap_metric_deltas로 (게이트 있음 vs 임상 단독)의 AUC/민감도/특이도/PPV 차이를 3000회
       재표본추출로 신뢰구간과 함께 추정한다.
    10. 위 모든 결과(모델별 지표, 재분류표, 통계검정, 부트스트랩)를 하나의 딕셔너리로 모아
        OUT_DIR 아래 CSV 3개 + JSON 2개 파일로 저장하고, 콘솔에도 JSON으로 출력한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]
    xclin_tr, xclin_te, clin_names = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(ytr, 5)

    clinical_oof, clinical_test = oof_and_external(
        lambda seed: clinical_estimator(),
        xclin_tr,
        ytr,
        xclin_te,
        folds,
    )
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

    gated_aec_tr = female_boundary_tr * a_z
    gated_aec_te = female_boundary_te * a_te_z
    gate_score_tr = c_z + 0.25 * gated_aec_tr
    gate_score_te = c_te_z + 0.25 * gated_aec_te
    gate_th = threshold_youden(ytr, gate_score_tr)
    gate_pred = gate_score_te >= gate_th

    hard_zone_tr = (np.abs(c_z - clinical_z_th) <= 0.50).astype(float)
    hard_zone_te = (np.abs(c_te_z - clinical_z_th) <= 0.50).astype(float)
    hard_gated_aec_tr = hard_zone_tr * female_tr * a_z
    hard_gated_aec_te = hard_zone_te * female_te * a_te_z
    hard_score_tr = c_z + 0.25 * hard_gated_aec_tr
    hard_score_te = c_te_z + 0.25 * hard_gated_aec_te
    hard_th = threshold_youden(ytr, hard_score_tr)
    hard_pred = hard_score_te >= hard_th

    stack = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000)
    stack.fit(np.column_stack([c_z, a_z]), ytr)
    stack_tr = stack.decision_function(np.column_stack([c_z, a_z]))
    stack_te = stack.decision_function(np.column_stack([c_te_z, a_te_z]))
    stack_th = threshold_youden(ytr, stack_tr)

    lrt_aec_score_train = logit_lrt(ytr, [c_z], [a_z], ["clinical_z", "aec_z"])
    lrt_aec_score_external = logit_lrt(yte, [c_te_z], [a_te_z], ["clinical_z", "aec_z"])
    lrt_gated_train = logit_lrt(ytr, [c_z], [gated_aec_tr], ["clinical_z", "female_boundary_aec_z"])
    lrt_gated_external = logit_lrt(yte, [c_te_z], [gated_aec_te], ["clinical_z", "female_boundary_aec_z"])
    lrt_hard_external = logit_lrt(yte, [c_te_z], [hard_gated_aec_te], ["clinical_z", "female_hard_zone_aec_z"])
    perm_p_gated_external = permutation_lrt_p(yte, c_te_z, gated_aec_te, lrt_gated_external["lrt_chi2"], n_perm=2000)

    clinical_pos = clinical_pred
    gate_pos = gate_pred
    reclass_rows = []
    for label, mask in [
        ("clinical+ / aec_gate+", clinical_pos & gate_pos),
        ("clinical+ / aec_gate-", clinical_pos & ~gate_pos),
        ("clinical- / aec_gate+", ~clinical_pos & gate_pos),
        ("clinical- / aec_gate-", ~clinical_pos & ~gate_pos),
    ]:
        row = {"group": label, **group_summary(yte, mask)}
        reclass_rows.append(row)
    reclass = pd.DataFrame(reclass_rows)

    # Conditional enrichment inside clinical-positive patients.
    cp_aec_pos = clinical_pos & gate_pos
    cp_aec_neg = clinical_pos & ~gate_pos
    table = np.array(
        [
            [int(yte[cp_aec_pos].sum()), int(cp_aec_pos.sum() - yte[cp_aec_pos].sum())],
            [int(yte[cp_aec_neg].sum()), int(cp_aec_neg.sum() - yte[cp_aec_neg].sum())],
        ]
    )
    fisher_or, fisher_p = stats.fisher_exact(table)

    # Paired false-positive test among non-events: clinical FP removed vs newly introduced FP.
    non_event = yte == 0
    clinical_fp = clinical_pred & non_event
    gate_fp = gate_pred & non_event
    fp_removed = int(np.sum(clinical_fp & ~gate_fp))
    fp_added = int(np.sum(~clinical_fp & gate_fp))
    fp_mcnemar = stats.binomtest(min(fp_removed, fp_added), fp_removed + fp_added, 0.5).pvalue if fp_removed + fp_added else np.nan

    event = yte == 1
    clinical_tp = clinical_pred & event
    gate_tp = gate_pred & event
    tp_lost = int(np.sum(clinical_tp & ~gate_tp))
    tp_gained = int(np.sum(~clinical_tp & gate_tp))
    tp_mcnemar = stats.binomtest(min(tp_lost, tp_gained), tp_lost + tp_gained, 0.5).pvalue if tp_lost + tp_gained else np.nan

    models = {
        "clinical_only": (clinical_oof, clinical_test, clinical_th),
        "aec_only_svm_expert": (aec_oof, aec_test, threshold_youden(ytr, aec_oof)),
        "simple_logistic_stack_clinical_plus_aec": (stack_tr, stack_te, stack_th),
        "hard_zone_female_width0.50_lambda0.25": (hard_score_tr, hard_score_te, hard_th),
        "female_boundary_gaussian_lambda0.25": (gate_score_tr, gate_score_te, gate_th),
    }
    model_rows = []
    for name, (score_tr, score_te, th) in models.items():
        row = {"model": name}
        row.update({f"train_oof_{k}": v for k, v in continuous_metrics(ytr, score_tr).items()})
        row.update({f"external_{k}": v for k, v in continuous_metrics(yte, score_te).items()})
        row.update({f"external_{k}": v for k, v in binary_metrics(yte, score_te, th).items()})
        model_rows.append(row)
    model_df = pd.DataFrame(model_rows)

    boot_gate = bootstrap_metric_deltas(yte, clinical_test, gate_score_te, clinical_th, gate_th)
    boot_hard = bootstrap_metric_deltas(yte, clinical_test, hard_score_te, clinical_th, hard_th)

    result = {
        "cohorts": {
            "g1090_train_n": int(len(ytr)),
            "g1090_train_events": int(ytr.sum()),
            "g1090_train_prevalence": float(ytr.mean()),
            "sdata_external_n": int(len(yte)),
            "sdata_external_events": int(yte.sum()),
            "sdata_external_prevalence": float(yte.mean()),
        },
        "main_question": "Does AEC contain incremental information beyond age, sex, height, weight?",
        "interpretation_rule": "Evidence is strongest if a train-derived AEC score or gate is associated with outcome after conditioning on the clinical score and improves locked external reclassification/utility.",
        "conditional_association_tests": {
            "oof_train_y_on_clinical_plus_aec_score": lrt_aec_score_train,
            "external_y_on_clinical_plus_aec_score": lrt_aec_score_external,
            "oof_train_y_on_clinical_plus_female_boundary_aec": lrt_gated_train,
            "external_y_on_clinical_plus_female_boundary_aec": {
                **lrt_gated_external,
                "permutation_p": perm_p_gated_external,
            },
            "external_y_on_clinical_plus_female_hard_zone_aec": lrt_hard_external,
        },
        "external_model_metrics": model_df.to_dict(orient="records"),
        "external_reclassification_2x2": reclass.to_dict(orient="records"),
        "clinical_positive_aec_gate_enrichment": {
            "table_rows": ["clinical+_aec_gate+", "clinical+_aec_gate-"],
            "table_cols": ["low_smi_event", "non_event"],
            "table": table.tolist(),
            "fisher_exact_or": float(fisher_or),
            "fisher_exact_p": float(fisher_p),
            "aec_gate_positive": group_summary(yte, cp_aec_pos),
            "aec_gate_negative": group_summary(yte, cp_aec_neg),
        },
        "paired_external_error_changes_gate_vs_clinical": {
            "false_positives_removed": fp_removed,
            "false_positives_added": fp_added,
            "false_positive_paired_binomial_p": float(fp_mcnemar),
            "true_positives_lost": tp_lost,
            "true_positives_gained": tp_gained,
            "true_positive_paired_binomial_p": float(tp_mcnemar),
        },
        "bootstrap_external_delta_gate_vs_clinical": boot_gate,
        "bootstrap_external_delta_hard_zone_vs_clinical": boot_hard,
    }

    model_df.to_csv(OUT_DIR / "external_model_metrics.csv", index=False)
    reclass.to_csv(OUT_DIR / "external_reclassification_2x2.csv", index=False)
    pd.DataFrame(result["conditional_association_tests"]).to_json(OUT_DIR / "conditional_association_tests.json", force_ascii=False, indent=2)
    with open(OUT_DIR / "aec_conditional_value_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
