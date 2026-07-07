from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import clinical_matrix  # noqa: E402
from aec_direct_vote_auc_boost import build_base_features  # noqa: E402
from aec_lock_smoothed_deesc_gate import DATA_DIR, build_candidate_bank, clinical_scores, load_dataset  # noqa: E402
from aec_region_constrained_cnn_gate import d1, d2, row_z  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_oof_auc_max_search"
PROB_PATH = Path(
    r"C:\Users\user\OneDrive\Dokumen\radiation\outputs\aec_region_cnn_pattern_gate\direct_vote_probabilities.npz"
)
SEED = 20260701


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model_key: str
    k: int | None = None
    c: float | None = None
    depth: int | None = None
    leaf: int | None = None


def auc_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """정답 라벨 y와 점수 score로 ROC AUC와 Mann-Whitney U 검정 p-value를 계산해 함께 반환한다."""
    auc = float(roc_auc_score(y, score))
    p = float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)
    return auc, p


def score_model(model, x: np.ndarray) -> np.ndarray:
    """모델이 지원하는 방식(decision_function > predict_proba > predict 순)으로 입력 x에 대한 점수를 산출한다."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def clinical_score_features(score: np.ndarray, z: np.ndarray) -> np.ndarray:
    """임상 점수와 그 z값의 원값/제곱/세제곱/절대값/tanh 변환을 묶어 비선형 특징 집합을 만든다."""
    return np.column_stack([score, z, z**2, z**3, np.abs(z), np.tanh(z)])


def curve_features(d: dict) -> np.ndarray:
    """AEC 곡선(norm)의 원값, z-정규화값, 1차/2차 미분의 z값, 스무딩 원본(raw)의 z값을 결합한 특징 행렬을 만든다."""
    norm = d["norm"]
    return np.column_stack(
        [
            norm,
            row_z(norm),
            row_z(d1(norm)),
            row_z(d2(norm)),
            row_z(d["smooth_raw"]),
        ]
    ).astype(float)


def load_direct_vote_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """PROB_PATH의 npz 파일에서 설정별(g/s) 확률 예측값을 불러와 direct-vote 기반 특징 행렬을 구성한다."""
    data = np.load(PROB_PATH, allow_pickle=True)
    configs = [str(x) for x in data["configs"]]
    prob_g = {name: np.asarray(data[f"{name}_prob_g"], dtype=float) for name in configs}
    prob_s = {name: np.asarray(data[f"{name}_prob_s"], dtype=float) for name in configs}
    xg, names = build_base_features(prob_g)
    xs, _ = build_base_features(prob_s)
    return xg, xs, names


def matrix_dict(g: dict, s: dict, clinical_oof: np.ndarray, clinical_ext: np.ndarray, c_g: np.ndarray, c_s: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """임상 공변량, 임상 점수, AEC 곡선, 후보뱅크, direct-vote 특징들을 개별 및 조합한 여러 (g, s) 특징 행렬 쌍을 이름별 딕셔너리로 구성한다."""
    xclin_g, xclin_s, _ = clinical_matrix(g["meta"], s["meta"])
    cscore_g = clinical_score_features(clinical_oof, c_g)
    cscore_s = clinical_score_features(clinical_ext, c_s)
    curve_g = curve_features(g)
    curve_s = curve_features(s)
    vote_g, vote_s, _ = load_direct_vote_features()

    print("building handcrafted candidate bank", flush=True)
    bank_g = build_candidate_bank(g["norm"]).to_numpy(dtype=float)
    bank_s = build_candidate_bank(s["norm"]).to_numpy(dtype=float)

    clinical_all_g = np.column_stack([xclin_g, cscore_g])
    clinical_all_s = np.column_stack([xclin_s, cscore_s])
    aec_all_g = np.column_stack([curve_g, bank_g, vote_g])
    aec_all_s = np.column_stack([curve_s, bank_s, vote_s])
    return {
        "clinical_cov": (xclin_g, xclin_s),
        "clinical_score": (cscore_g, cscore_s),
        "clinical_all": (clinical_all_g, clinical_all_s),
        "aec_curve": (curve_g, curve_s),
        "aec_bank": (bank_g, bank_s),
        "aec_vote": (vote_g, vote_s),
        "aec_curve_vote": (np.column_stack([curve_g, vote_g]), np.column_stack([curve_s, vote_s])),
        "aec_all": (aec_all_g, aec_all_s),
        "clinical_plus_curve": (np.column_stack([clinical_all_g, curve_g]), np.column_stack([clinical_all_s, curve_s])),
        "clinical_plus_bank": (np.column_stack([clinical_all_g, bank_g]), np.column_stack([clinical_all_s, bank_s])),
        "clinical_plus_vote": (np.column_stack([clinical_all_g, vote_g]), np.column_stack([clinical_all_s, vote_s])),
        "clinical_plus_curve_vote": (
            np.column_stack([clinical_all_g, curve_g, vote_g]),
            np.column_stack([clinical_all_s, curve_s, vote_s]),
        ),
        "clinical_plus_all": (np.column_stack([clinical_all_g, aec_all_g]), np.column_stack([clinical_all_s, aec_all_s])),
    }


def selector(k: int | None, n_features: int):
    """결측치 중앙값 대체 단계와, k가 지정되고 전체 특징 수보다 작으면 f_classif 기반 SelectKBest 단계를 담은 파이프라인 스텝 목록을 만든다."""
    steps = [SimpleImputer(strategy="median")]
    if k is not None and k < n_features:
        steps.append(SelectKBest(f_classif, k=max(2, min(k, n_features))))
    return steps


def model_factory(cand: Candidate, n_features: int, seed: int):
    """Candidate에 지정된 model_key(logit_l2/l1, linear_svm, rbf_svm, knn, histgb, extratrees, rf)에 맞는 sklearn 파이프라인을 생성한다."""
    ksteps = selector(cand.k, n_features)
    c = 1.0 if cand.c is None else cand.c
    if cand.model_key == "logit_l2":
        return make_pipeline(
            *ksteps,
            StandardScaler(),
            LogisticRegression(C=c, solver="lbfgs", class_weight="balanced", max_iter=5000, random_state=seed),
        )
    if cand.model_key == "logit_l1":
        return make_pipeline(
            *ksteps,
            StandardScaler(),
            LogisticRegression(C=c, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed),
        )
    if cand.model_key == "linear_svm":
        return make_pipeline(
            *ksteps,
            StandardScaler(),
            LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=20000, random_state=seed),
        )
    if cand.model_key == "rbf_svm":
        return make_pipeline(
            *ksteps,
            StandardScaler(),
            SVC(C=c, gamma="scale", kernel="rbf", class_weight="balanced", random_state=seed),
        )
    if cand.model_key == "knn":
        return make_pipeline(*ksteps, StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
    if cand.model_key == "histgb":
        return make_pipeline(
            *ksteps,
            HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=240,
                max_leaf_nodes=15 if cand.depth is None else cand.depth,
                min_samples_leaf=20 if cand.leaf is None else cand.leaf,
                l2_regularization=0.08,
                random_state=seed,
            ),
        )
    if cand.model_key == "extratrees":
        return make_pipeline(
            *ksteps,
            ExtraTreesClassifier(
                n_estimators=800,
                max_depth=cand.depth,
                min_samples_leaf=8 if cand.leaf is None else cand.leaf,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if cand.model_key == "rf":
        return make_pipeline(
            *ksteps,
            RandomForestClassifier(
                n_estimators=700,
                max_depth=cand.depth,
                min_samples_leaf=10 if cand.leaf is None else cand.leaf,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(cand.model_key)


def candidates() -> list[Candidate]:
    """특징 집합 x 모델 종류 x 하이퍼파라미터(k, C, depth, leaf) 조합으로 탐색할 전체 Candidate 목록을 생성한다."""
    out: list[Candidate] = []
    feature_sets = [
        "clinical_all",
        "clinical_plus_curve",
        "clinical_plus_bank",
        "clinical_plus_vote",
        "clinical_plus_curve_vote",
        "clinical_plus_all",
        "aec_all",
    ]
    for fs in feature_sets:
        for k in [40, 100, 250, 600]:
            for c in [0.03, 0.1, 0.3]:
                out.append(Candidate(f"{fs}__l2_k{k}_C{c}", fs, "logit_l2", k=k, c=c))
        for k in [100, 250]:
            for c in [0.03, 0.1]:
                out.append(Candidate(f"{fs}__l1_k{k}_C{c}", fs, "logit_l1", k=k, c=c))
        for k in [100, 250]:
            for c in [0.03, 0.1, 0.3]:
                out.append(Candidate(f"{fs}__linear_svm_k{k}_C{c}", fs, "linear_svm", k=k, c=c))
        for k in [40, 100]:
            for c in [0.3, 1.0]:
                out.append(Candidate(f"{fs}__rbf_svm_k{k}_C{c}", fs, "rbf_svm", k=k, c=c))
        for k in [100, 250]:
            out.append(Candidate(f"{fs}__histgb_k{k}", fs, "histgb", k=k, depth=15, leaf=20))
            out.append(Candidate(f"{fs}__extratrees_k{k}", fs, "extratrees", k=k, depth=5, leaf=10))
            out.append(Candidate(f"{fs}__rf_k{k}", fs, "rf", k=k, depth=5, leaf=12))
        out.append(Candidate(f"{fs}__knn_k100", fs, "knn", k=100))
    return out


def crossfit_candidate(cand: Candidate, xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """후보 모델을 5-fold로 학습해 내부 OOF 점수와 외부 데이터 fold별 평균 점수를 구하고, 전체 데이터로 재학습한 모델의 외부 재적합 점수도 함께 반환한다."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(yg), dtype=float)
    ext_scores = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(yg)), yg)):
        model = model_factory(cand, xg.shape[1], SEED + fold)
        model.fit(xg[tr], yg[tr])
        oof[va] = score_model(model, xg[va])
        ext_scores.append(score_model(model, xs))
    final = model_factory(cand, xg.shape[1], SEED + 999)
    final.fit(xg, yg)
    ext_refit = score_model(final, xs)
    return oof, np.mean(ext_scores, axis=0), ext_refit


def plot_top(summary: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """내부 OOF AUC 기준 상위 top_n개 후보에 대해 내부 OOF AUC와 외부 재적합 AUC를 나란히 비교하는 가로 막대 그래프를 그려 저장한다."""
    top = summary.sort_values("internal_oof_auc", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, max(6, 0.35 * len(top))), constrained_layout=True)
    y = np.arange(len(top))
    ax.barh(y - 0.18, top["internal_oof_auc"], height=0.34, color="#4c78a8", label="Internal OOF")
    ax.barh(y + 0.18, top["external_refit_auc"], height=0.34, color="#f58518", label="External refit")
    ax.axvline(0.90, color="#d62728", ls=":", lw=1.5, label="0.90 target")
    ax.set_yticks(y)
    ax.set_yticklabels(top["name"].tolist(), fontsize=7)
    ax.set_xlim(0.65, 0.93)
    ax.set_xlabel("AUC")
    ax.set_title("OOF-max internal leaderboard", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """임상 점수 베이스라인과 대량의 후보 조합(candidates)에 대해 내부 OOF AUC를 계산하여 순위를 매기고,
    요약/오류 CSV, 선택된 점수 CSV, 상위 30개 그래프, 설명 JSON을 저장한 뒤 상위 결과를 콘솔에 출력한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    matrices = matrix_dict(g, s, clinical_oof, clinical_ext, c_g, c_s)
    yg = g["y"].astype(int)
    ys = s["y"].astype(int)

    rows = []
    score_cols: dict[str, np.ndarray | list] = {
        "dataset": ["g1090_internal"] * len(yg) + ["sdata_external"] * len(ys),
        "row_index": list(range(len(yg))) + list(range(len(ys))),
        "y_low_smi": np.r_[yg, ys],
        "clinical_score": np.r_[clinical_oof, clinical_ext],
    }
    cg_auc, cg_p = auc_p(yg, clinical_oof)
    cs_auc, cs_p = auc_p(ys, clinical_ext)
    rows.append(
        {
            "name": "clinical_only_current_baseline",
            "feature_set": "clinical_score",
            "model_key": "baseline",
            "k": np.nan,
            "internal_oof_auc": cg_auc,
            "internal_oof_p": cg_p,
            "external_fold_auc": cs_auc,
            "external_fold_p": cs_p,
            "external_refit_auc": cs_auc,
            "external_refit_p": cs_p,
        }
    )

    cand_list = candidates()
    for i, cand in enumerate(cand_list, start=1):
        xg, xs = matrices[cand.feature_set]
        try:
            oof, ext_fold, ext_refit = crossfit_candidate(cand, xg, yg, xs)
            iauc, ip = auc_p(yg, oof)
            efauc, efp = auc_p(ys, ext_fold)
            erauc, erp = auc_p(ys, ext_refit)
            rows.append(
                {
                    "name": cand.name,
                    "feature_set": cand.feature_set,
                    "model_key": cand.model_key,
                    "k": cand.k if cand.k is not None else np.nan,
                    "c": cand.c if cand.c is not None else np.nan,
                    "depth": cand.depth if cand.depth is not None else np.nan,
                    "leaf": cand.leaf if cand.leaf is not None else np.nan,
                    "n_features": xg.shape[1],
                    "internal_oof_auc": iauc,
                    "internal_oof_p": ip,
                    "external_fold_auc": efauc,
                    "external_fold_p": efp,
                    "external_refit_auc": erauc,
                    "external_refit_p": erp,
                }
            )
            if iauc >= 0.84 or i % 100 == 0:
                print(f"[{i}/{len(cand_list)}] {cand.name} OOF={iauc:.4f} EXT={erauc:.4f}", flush=True)
            if len(score_cols) < 60 and iauc >= 0.84:
                score_cols[f"{cand.name}__oof_or_external_refit"] = np.r_[oof, ext_refit]
        except Exception as exc:  # keep the broad search moving.
            rows.append(
                {
                    "name": cand.name,
                    "feature_set": cand.feature_set,
                    "model_key": cand.model_key,
                    "k": cand.k if cand.k is not None else np.nan,
                    "error": repr(exc),
                }
            )
            print(f"[{i}/{len(cand_list)}] FAILED {cand.name}: {exc}", flush=True)

    summary = pd.DataFrame(rows)
    good = summary[summary["internal_oof_auc"].notna()].copy()
    good = good.sort_values(["internal_oof_auc", "external_refit_auc"], ascending=False).reset_index(drop=True)
    good.insert(0, "selected_by_internal_oof_rank", np.arange(1, len(good) + 1))
    errors = summary[summary["internal_oof_auc"].isna()].copy()
    good.to_csv(OUT_DIR / "oof_auc_max_search_summary.csv", index=False)
    errors.to_csv(OUT_DIR / "oof_auc_max_search_errors.csv", index=False)
    pd.DataFrame(score_cols).to_csv(OUT_DIR / "oof_auc_max_search_selected_scores.csv", index=False)
    plot_top(good, OUT_DIR / "oof_auc_max_search_top30.png")
    with (OUT_DIR / "oof_auc_max_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "warning": "Exploratory OOF leaderboard. Selecting the best row by OOF makes the OOF optimistic; external AUC remains the held-out check.",
                "n_candidates_attempted": len(cand_list),
                "n_candidates_successful": int(len(good) - 1),
                "feature_sets": list(matrices.keys()),
                "validation": "g1090 internal 5-fold OOF; sdata external held-out",
            },
            f,
            indent=2,
        )
    show_cols = [
        "selected_by_internal_oof_rank",
        "name",
        "feature_set",
        "model_key",
        "k",
        "internal_oof_auc",
        "internal_oof_p",
        "external_refit_auc",
        "external_refit_p",
        "external_fold_auc",
    ]
    print("\nTOP 30 BY INTERNAL OOF AUC")
    print(good[show_cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# 다양한 특징 조합/모델/하이퍼파라미터 후보를 내부 OOF AUC 기준으로 대규모 탐색하여 최적 조합을 찾고 결과를 저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
