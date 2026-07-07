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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, OPS, clinical_scores, load_dataset  # noqa: E402
from aec_region_cnn_direct_vote_gate import soft_atleast2_np  # noqa: E402


PROB_PATH = Path(
    r"C:\Users\user\OneDrive\Dokumen\radiation\outputs\aec_region_cnn_pattern_gate\direct_vote_probabilities.npz"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_direct_vote_auc_boost"
SEED = 20260701
BOOT_N = 2000


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model_key: str


CANDIDATES = [
    Candidate("vote_only_logit_l2", "vote", "logit_l2"),
    Candidate("vote_only_logit_l1", "vote", "logit_l1"),
    Candidate("vote_only_svm_rbf", "vote", "svm_rbf"),
    Candidate("vote_only_histgb", "vote", "histgb"),
    Candidate("vote_only_extratrees", "vote", "extratrees"),
    Candidate("vote_poly_logit_l2", "vote_poly", "logit_l2"),
    Candidate("clinical_plus_vote_logit_l2", "clinical_vote", "logit_l2"),
    Candidate("clinical_plus_vote_logit_l1", "clinical_vote", "logit_l1"),
    Candidate("clinical_plus_vote_poly_logit_l2", "clinical_vote_poly", "logit_l2"),
    Candidate("clinical_plus_vote_svm_rbf", "clinical_vote", "svm_rbf"),
    Candidate("clinical_plus_vote_histgb", "clinical_vote", "histgb"),
    Candidate("clinical_plus_vote_randomforest", "clinical_vote", "rf"),
    Candidate("clinical_plus_vote_extratrees", "clinical_vote", "extratrees"),
]


def auc_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """점수로 AUC를 계산하고, Mann-Whitney U 검정으로 두 그룹(양성/음성) 점수 분포 차이의 p값도 함께 반환."""
    auc = float(roc_auc_score(y, score))
    p = float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)
    return auc, p


def paired_delta_bootstrap(
    y: np.ndarray, score_new: np.ndarray, score_ref: np.ndarray, seed: int, n_boot: int = BOOT_N
) -> tuple[float, float, float, float]:
    """새 점수와 기준(임상) 점수의 AUC 차이를 관측하고, 환자를 복원추출로 재표본화한 부트스트랩 분포로 그 차이의 양측 p값과 95% 신뢰구간을 계산."""
    obs = float(roc_auc_score(y, score_new) - roc_auc_score(y, score_ref))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], score_new[idx]) - roc_auc_score(y[idx], score_ref[idx])))
    arr = np.asarray(deltas)
    if arr.size == 0:
        return obs, np.nan, np.nan, np.nan
    p = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return obs, float(min(1.0, p)), float(lo), float(hi)


def score_estimator(model, x: np.ndarray) -> np.ndarray:
    """학습된 sklearn 모델에서 decision_function, 없으면 predict_proba(양성 클래스), 그마저 없으면 predict를 사용해 순위용 위험점수를 뽑아냄."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def model_factory(key: str, seed: int):
    """키 이름(logit_l2/logit_l1/svm_rbf/histgb/rf/extratrees/linear_svm)에 따라 표준화·특징선택 등을 포함한 sklearn 파이프라인 또는 분류기를 생성해 반환."""
    if key == "logit_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if key == "logit_l1":
        return make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=40),
            LogisticRegression(
                C=0.08,
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if key == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=60),
            SVC(C=0.6, gamma="scale", kernel="rbf", class_weight="balanced", probability=False, random_state=seed),
        )
    if key == "histgb":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.035,
            max_leaf_nodes=7,
            max_iter=220,
            l2_regularization=0.12,
            random_state=seed,
        )
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=18,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if key == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_depth=4,
            min_samples_leaf=16,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if key == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.05, class_weight="balanced", dual="auto", max_iter=10000, random_state=seed),
        )
    raise ValueError(key)


def direct_vote_features(prob: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    """(환자, 운영점, 구간) 3차원의 direct-vote CNN 확률로부터, 원본 확률/2표이상 근사확률/구간별·운영점별 평균과 표준편차/여러 임계값에서의 투표·득표수·합의여부 등 다양한 파생 특징들을 만들어 하나의 특징행렬과 이름 목록으로 반환."""
    mats = []
    names = []
    flat = prob.reshape(len(prob), -1)
    mats.append(flat)
    names += [f"{prefix}_op{o+1}_r{r+1}" for o in range(prob.shape[1]) for r in range(prob.shape[2])]
    soft2 = soft_atleast2_np(prob)
    mats.append(soft2)
    names += [f"{prefix}_soft2_op{o+1}" for o in range(prob.shape[1])]
    mats.append(np.column_stack([soft2.mean(axis=1), soft2.min(axis=1), soft2.max(axis=1), soft2.std(axis=1)]))
    names += [f"{prefix}_soft2_mean", f"{prefix}_soft2_min", f"{prefix}_soft2_max", f"{prefix}_soft2_sd"]
    branch_mean = prob.mean(axis=1)
    branch_sd = prob.std(axis=1)
    op_mean = prob.mean(axis=2)
    op_sd = prob.std(axis=2)
    mats.extend([branch_mean, branch_sd, op_mean, op_sd])
    names += [f"{prefix}_branch{j+1}_mean" for j in range(prob.shape[2])]
    names += [f"{prefix}_branch{j+1}_sd" for j in range(prob.shape[2])]
    names += [f"{prefix}_op{o+1}_mean" for o in range(prob.shape[1])]
    names += [f"{prefix}_op{o+1}_sd" for o in range(prob.shape[1])]
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        v = (prob >= th).astype(float)
        mats.append(v.reshape(len(prob), -1))
        names += [f"{prefix}_vote_t{th:.2f}_op{o+1}_r{r+1}" for o in range(prob.shape[1]) for r in range(prob.shape[2])]
        counts = v.sum(axis=2)
        mats.append(counts)
        names += [f"{prefix}_count_t{th:.2f}_op{o+1}" for o in range(prob.shape[1])]
        mats.append((counts >= 2).astype(float))
        names += [f"{prefix}_consensus_t{th:.2f}_op{o+1}" for o in range(prob.shape[1])]
    return np.column_stack(mats), names


def build_base_features(prob_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """여러 CNN 설정(prob_dict의 각 키)마다 direct_vote_features로 만든 특징들을 모두 이어붙여 하나의 기반 특징행렬을 구성."""
    mats = []
    names = []
    for key, prob in prob_dict.items():
        x, n = direct_vote_features(prob, key)
        mats.append(x)
        names.extend(n)
    return np.column_stack(mats), names


def add_clinical_features(
    x: np.ndarray,
    names: list[str],
    clinical_score: np.ndarray,
    clinical_z: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """임상점수/z값과 그 거듭제곱, 운영점별 임계값과의 거리(delta), 경계 근접도(boundary), 임상양성 여부를 특징으로 만들어 기존 특징행렬 x에 이어붙임."""
    th_values = np.array([thresholds[op] for op, _ in OPS], dtype=float)
    delta = clinical_z[:, None] - th_values[None, :]
    boundary = np.exp(-0.5 * (delta / 0.5) ** 2)
    cfeat = np.column_stack(
        [
            clinical_score,
            clinical_z,
            clinical_z**2,
            clinical_z**3,
            delta,
            boundary,
            (delta >= 0).astype(float),
        ]
    )
    cnames = (
        ["clinical_score", "clinical_z", "clinical_z2", "clinical_z3"]
        + [f"clinical_delta_{op}" for op, _ in OPS]
        + [f"clinical_boundary_{op}" for op, _ in OPS]
        + [f"clinical_positive_{op}" for op, _ in OPS]
    )
    return np.column_stack([x, cfeat]), names + cnames


def feature_set_matrix(
    feature_set: str,
    base_x: np.ndarray,
    base_names: list[str],
    clinical_score: np.ndarray,
    clinical_z: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """feature_set 이름(vote/clinical_vote/vote_poly/clinical_vote_poly)에 따라 기반 특징을 그대로 쓰거나, 임상 특징을 더하거나, 2차 상호작용항(PolynomialFeatures)을 추가한 최종 특징행렬을 구성."""
    if feature_set == "vote":
        return base_x, base_names
    if feature_set == "clinical_vote":
        return add_clinical_features(base_x, base_names, clinical_score, clinical_z, thresholds)
    if feature_set == "vote_poly":
        x = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit_transform(base_x[:, :80])
        return x, [f"vote_poly_{i}" for i in range(x.shape[1])]
    if feature_set == "clinical_vote_poly":
        x, names = add_clinical_features(base_x, base_names, clinical_score, clinical_z, thresholds)
        # Keep the interaction search controlled: direct-vote summary features + clinical features.
        summary_idx = [i for i, n in enumerate(names) if ("soft2_" in n or "branch" in n or n.startswith("clinical_"))]
        summary_idx = summary_idx[:120]
        xp = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit_transform(x[:, summary_idx])
        return xp, [f"clinical_vote_poly_{i}" for i in range(xp.shape[1])]
    raise ValueError(feature_set)


def crossfit_candidate(candidate: Candidate, xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """지정된 후보 모델을 5-fold로 교차학습해 내부 OOF 점수를 얻고, 전체 데이터로 재학습한 최종모델과 fold별 모델들의 외부 예측을 절반씩 블렌딩해 분산을 줄인 외부 점수를 반환."""
    oof = np.zeros(len(yg), dtype=float)
    ext_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(yg)), yg)):
        model = model_factory(candidate.model_key, SEED + fold)
        model.fit(xg[tr], yg[tr])
        oof[va] = score_estimator(model, xg[va])
        ext_scores.append(score_estimator(model, xs))
    final = model_factory(candidate.model_key, SEED + 999)
    final.fit(xg, yg)
    ext_final = score_estimator(final, xs)
    # Blend final and fold ensemble to reduce variance.
    ext = 0.5 * ext_final + 0.5 * np.mean(ext_scores, axis=0)
    return oof, ext


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    """임상단독을 제외한 모든 후보 모델의 내부/외부 AUC를 외부 AUC 오름차순 가로 막대그래프로 그리고, 임상단독 외부 AUC와 목표선(0.90)을 함께 표시해 PNG로 저장."""
    rows = summary[summary["model"].ne("clinical_only")].sort_values("external_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.38 * len(rows))), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y - 0.18, rows["internal_auc"], height=0.34, color="#4c78a8", label="Internal/Gangnam OOF")
    ax.barh(y + 0.18, rows["external_auc"], height=0.34, color="#f58518", label="External/Sinchon")
    clinical = float(summary.loc[summary["model"].eq("clinical_only"), "external_auc"].iloc[0])
    ax.axvline(clinical, color="black", ls="--", lw=1.2, label=f"Clinical external {clinical:.3f}")
    ax.axvline(0.90, color="#d62728", ls=":", lw=1.6, label="AUC 0.90 target")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["model"].tolist(), fontsize=8)
    ax.set_xlim(0.50, 0.93)
    ax.set_xlabel("AUC")
    ax.set_title("Direct-vote CNN score boosting", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 이미 학습된 direct-vote CNN 구간 확률을 원료 삼아, 다양한 파생
    특징 + 다양한 분류기(로지스틱/SVM/트리 앙상블 등) 조합으로 임상단독보다 AUC를 최대한 끌어올릴 수
    있는가? — de-escalation 게이트가 아니라 순수 AUC 극대화를 노리는 탐색적 실험):

    1. g1090/sdata와 임상점수를 로드하고, 미리 저장된 direct-vote CNN 확률(PROB_PATH)을 불러와
       다양한 파생 특징(direct_vote_features/build_base_features)을 구성.
    2. 임상단독과, CNN의 "2표 이상" 근사확률 평균만 사용한 원시 점수의 AUC를 기준으로 계산.
    3. CANDIDATES에 정의된 여러 (특징집합, 분류기) 조합마다 특징행렬을 만들고(feature_set_matrix)
       5-fold 교차학습(crossfit_candidate)으로 내부 OOF/외부 점수를 얻어 AUC와 임상단독 대비
       부트스트랩 AUC 증분(p값, 신뢰구간)을 계산.
    4. 모든 모델을 외부 AUC 기준으로 정렬한 요약표를 그래프로 시각화.
    5. 요약표/원점수를 CSV로, 실행 설정을 JSON으로 저장한 뒤 콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    data = np.load(PROB_PATH, allow_pickle=True)
    configs = [str(x) for x in data["configs"]]
    prob_g = {name: np.asarray(data[f"{name}_prob_g"], dtype=float) for name in configs}
    prob_s = {name: np.asarray(data[f"{name}_prob_s"], dtype=float) for name in configs}
    base_g, base_names = build_base_features(prob_g)
    base_s, _ = build_base_features(prob_s)

    rows = []
    score_df = pd.DataFrame(
        {
            "dataset": ["g1090_internal"] * len(g["y"]) + ["sdata_external"] * len(s["y"]),
            "row_index": list(range(len(g["y"]))) + list(range(len(s["y"]))),
            "y_low_smi": np.r_[g["y"], s["y"]],
            "clinical_score": np.r_[clinical_oof, clinical_ext],
            "clinical_z": np.r_[c_g, c_s],
        }
    )
    cg_auc, cg_p = auc_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_p(s["y"], clinical_ext)
    rows.append(
        {
            "model": "clinical_only",
            "feature_set": "clinical",
            "internal_auc": cg_auc,
            "internal_auc_p": cg_p,
            "external_auc": cs_auc,
            "external_auc_p": cs_p,
            "internal_delta_vs_clinical": 0.0,
            "internal_delta_p_bootstrap": np.nan,
            "external_delta_vs_clinical": 0.0,
            "external_delta_p_bootstrap": np.nan,
        }
    )

    for name in configs:
        lowrisk_g = soft_atleast2_np(prob_g[name]).mean(axis=1)
        lowrisk_s = soft_atleast2_np(prob_s[name]).mean(axis=1)
        score_g = -lowrisk_g
        score_s = -lowrisk_s
        ig_auc, ig_p = auc_p(g["y"], score_g)
        es_auc, es_p = auc_p(s["y"], score_s)
        idel, idelp, _, _ = paired_delta_bootstrap(g["y"], score_g, clinical_oof, SEED + len(rows))
        edel, edelp, _, _ = paired_delta_bootstrap(s["y"], score_s, clinical_ext, SEED + 100 + len(rows))
        rows.append(
            {
                "model": f"{name}_raw_low_smi_risk",
                "feature_set": "raw_direct_vote_score",
                "internal_auc": ig_auc,
                "internal_auc_p": ig_p,
                "external_auc": es_auc,
                "external_auc_p": es_p,
                "internal_delta_vs_clinical": idel,
                "internal_delta_p_bootstrap": idelp,
                "external_delta_vs_clinical": edel,
                "external_delta_p_bootstrap": edelp,
            }
        )
        score_df.loc[score_df["dataset"].eq("g1090_internal"), f"{name}_raw_low_smi_risk"] = score_g
        score_df.loc[score_df["dataset"].eq("sdata_external"), f"{name}_raw_low_smi_risk"] = score_s

    for i, cand in enumerate(CANDIDATES):
        print(f"[{i + 1}/{len(CANDIDATES)}] {cand.name}", flush=True)
        xg, _ = feature_set_matrix(cand.feature_set, base_g, base_names, clinical_oof, c_g, thresholds)
        xs, _ = feature_set_matrix(cand.feature_set, base_s, base_names, clinical_ext, c_s, thresholds)
        score_g, score_s = crossfit_candidate(cand, xg, g["y"].astype(int), xs)
        ig_auc, ig_p = auc_p(g["y"], score_g)
        es_auc, es_p = auc_p(s["y"], score_s)
        idel, idelp, idlo, idhi = paired_delta_bootstrap(g["y"], score_g, clinical_oof, SEED + 200 + i)
        edel, edelp, edlo, edhi = paired_delta_bootstrap(s["y"], score_s, clinical_ext, SEED + 400 + i)
        rows.append(
            {
                "model": cand.name,
                "feature_set": cand.feature_set,
                "internal_auc": ig_auc,
                "internal_auc_p": ig_p,
                "external_auc": es_auc,
                "external_auc_p": es_p,
                "internal_delta_vs_clinical": idel,
                "internal_delta_p_bootstrap": idelp,
                "internal_delta_ci_low": idlo,
                "internal_delta_ci_high": idhi,
                "external_delta_vs_clinical": edel,
                "external_delta_p_bootstrap": edelp,
                "external_delta_ci_low": edlo,
                "external_delta_ci_high": edhi,
            }
        )
        score_df.loc[score_df["dataset"].eq("g1090_internal"), cand.name] = score_g
        score_df.loc[score_df["dataset"].eq("sdata_external"), cand.name] = score_s

    summary = pd.DataFrame(rows).sort_values(["external_auc", "internal_auc"], ascending=False).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "direct_vote_auc_boost_summary.csv", index=False)
    score_df.to_csv(OUT_DIR / "direct_vote_auc_boost_scores.csv", index=False)
    plot_summary(summary, OUT_DIR / "direct_vote_auc_boost_plot.png")
    with (OUT_DIR / "direct_vote_auc_boost_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_probabilities": str(PROB_PATH),
                "input": "direct-vote CNN branch probabilities plus optional clinical score context",
                "validation": "internal g1090 cross-fitted OOF; external sdata held-out",
                "target": "low SMI",
                "note": "Exploratory AUC-max calibration. External AUC is the only relevant target for portability.",
            },
            f,
            indent=2,
        )
    show = summary[
        [
            "model",
            "feature_set",
            "internal_auc",
            "internal_auc_p",
            "internal_delta_vs_clinical",
            "internal_delta_p_bootstrap",
            "external_auc",
            "external_auc_p",
            "external_delta_vs_clinical",
            "external_delta_p_bootstrap",
        ]
    ]
    print("\nDIRECT-VOTE AUC BOOST SUMMARY")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    # 저장된 direct-vote CNN 확률과 임상점수 로드 -> 파생 특징 구성 -> 여러 (특징집합, 분류기) 후보를
    # 5-fold 교차학습해 내부/외부 AUC와 임상단독 대비 증분 계산 -> 결과 정렬/그래프화 및 저장 순으로
    # 실행된다.
    main()
