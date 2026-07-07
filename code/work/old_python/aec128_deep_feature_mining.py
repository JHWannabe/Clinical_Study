from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import savgol_filter
from scipy import stats

from aec128_common_shape_feature import FILES, feature_stats, load_aec128, summarize_feature
from aec128_cylindrical_features import extract_cylindrical_features, smooth_log_profile
from aec128_visual_shape_features import extract_visual_features


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_deep_feature_mining"
SEED = 20260629
RNG = np.random.default_rng(SEED)


def stratified_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """클래스 비율을 유지하며 데이터를 k개의 교차검증 폴드 인덱스로 분할."""
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in [False, True]:
        idx = np.flatnonzero(y == cls)
        RNG.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=int) for f in folds]


def unit_template(template: np.ndarray) -> np.ndarray:
    """템플릿을 평균 0, 노름(길이) 1이 되도록 정규화."""
    centered = template - np.mean(template)
    norm = np.linalg.norm(centered)
    if not np.isfinite(norm) or norm == 0:
        return centered
    return centered / norm


def template_from_xy(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """양성-음성 평균 곡선 차이로 매칭 필터 템플릿을 만들고(옵션으로 특정 구간만 mask), 단위 정규화."""
    t = np.nanmean(x[y], axis=0) - np.nanmean(x[~y], axis=0)
    if mask is not None:
        t = t * mask
    return unit_template(t)


def matched_filter_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """폴드별로 템플릿을 다시 만들어 train의 out-of-fold 매칭필터 점수를 구하고, 전체 train으로 만든 최종 템플릿으로 외부 점수까지 계산."""
    oof = np.zeros(train_x.shape[0], dtype=float)
    folds = stratified_folds(train_y, k=k)
    all_idx = np.arange(train_x.shape[0])
    for val_idx in folds:
        tr_idx = np.setdiff1d(all_idx, val_idx)
        template = template_from_xy(train_x[tr_idx], train_y[tr_idx], mask=mask)
        oof[val_idx] = train_x[val_idx] @ template
    final_template = template_from_xy(train_x, train_y, mask=mask)
    test_score = test_x @ final_template
    return oof, test_score, final_template


def haar_features(p: np.ndarray, levels: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """1~5단계 Haar 웨이블릿 스타일로 곡선을 점점 세분화된 블록으로 나눠, 각 블록 쌍의 우측-좌측 평균 차이를 특징으로 계산."""
    rows = {}
    n = p.shape[1]
    for level in levels:
        n_blocks = 2**level
        block = n // n_blocks
        for b in range(n_blocks // 2):
            left = slice((2 * b) * block, (2 * b + 1) * block)
            right = slice((2 * b + 1) * block, (2 * b + 2) * block)
            rows[f"haar_l{level}_b{b + 1:02d}_right_minus_left"] = p[:, right].mean(axis=1) - p[:, left].mean(axis=1)
    return pd.DataFrame(rows)


def spectral_features(p: np.ndarray) -> pd.DataFrame:
    """DCT 계수 16개와 저/중/고주파 에너지, 중/저주파 에너지 비율을 특징으로 계산."""
    coeff = dct(p, type=2, norm="ortho", axis=1)
    rows = {}
    for i in range(1, 17):
        rows[f"dct_{i:02d}"] = coeff[:, i]
    rows["dct_low_freq_energy_1_4"] = np.mean(coeff[:, 1:5] ** 2, axis=1)
    rows["dct_mid_freq_energy_5_12"] = np.mean(coeff[:, 5:13] ** 2, axis=1)
    rows["dct_high_freq_energy_13_32"] = np.mean(coeff[:, 13:33] ** 2, axis=1)
    rows["dct_mid_to_low_energy_ratio"] = rows["dct_mid_freq_energy_5_12"] / (rows["dct_low_freq_energy_1_4"] + 1e-8)
    return pd.DataFrame(rows)


def run_length_features(p: np.ndarray) -> pd.DataFrame:
    """곡선의 부호(양/음) 패턴에서 가장 긴 양/음 연속 구간의 길이·위치, 부호 전환 횟수, 양/음 비율을 위상학적 특징으로 계산."""
    rows = []
    z = np.arange(1, p.shape[1] + 1)
    for row in p:
        pos = row > 0
        neg = row < 0

        def longest_run(mask: np.ndarray) -> tuple[int, float, float]:
            best_len = 0
            best_start = np.nan
            best_end = np.nan
            start = None
            for i, val in enumerate(mask):
                if val and start is None:
                    start = i
                if start is not None and (not val or i == len(mask) - 1):
                    end = i if val and i == len(mask) - 1 else i - 1
                    length = end - start + 1
                    if length > best_len:
                        best_len = length
                        best_start = z[start]
                        best_end = z[end]
                    start = None
            return best_len, best_start, best_end

        pos_len, pos_start, pos_end = longest_run(pos)
        neg_len, neg_start, neg_end = longest_run(neg)
        rows.append(
            {
                "longest_positive_run_len": pos_len,
                "longest_positive_run_start": pos_start,
                "longest_positive_run_end": pos_end,
                "longest_negative_run_len": neg_len,
                "longest_negative_run_start": neg_start,
                "longest_negative_run_end": neg_end,
                "n_zero_crossings": int(np.sum(np.diff(np.signbit(row)) != 0)),
                "fraction_positive": float(np.mean(pos)),
                "fraction_negative": float(np.mean(neg)),
            }
        )
    return pd.DataFrame(rows)


def moment_features(p: np.ndarray) -> pd.DataFrame:
    """곡선의 절댓값을 확률질량처럼 취급해 무게중심·퍼짐·왜도·첨도(통계적 모멘트)를 계산하고, 부호를 살린 왜도/꼬리 두께 근사치도 추가."""
    z = np.linspace(-1.0, 1.0, p.shape[1])
    absdev = np.abs(p)
    denom = absdev.sum(axis=1) + 1e-8
    centroid = (absdev @ z) / denom
    centered = z[None, :] - centroid[:, None]
    spread = np.sqrt(np.sum(absdev * (centered**2), axis=1) / denom)
    skew = np.sum(absdev * (centered**3), axis=1) / (denom * (spread**3 + 1e-8))
    kurt = np.sum(absdev * (centered**4), axis=1) / (denom * (spread**4 + 1e-8))
    return pd.DataFrame(
        {
            "abs_shape_centroid_z": centroid,
            "abs_shape_spread_z": spread,
            "abs_shape_skewness_z": skew,
            "abs_shape_kurtosis_z": kurt,
            "signed_shape_skew_proxy": np.mean(p * (z[None, :] ** 3), axis=1),
            "signed_shape_tail_heaviness_proxy": np.mean(p * (z[None, :] ** 4), axis=1),
        }
    )


def fixed_deep_features(x_norm: np.ndarray) -> pd.DataFrame:
    """visual/cylindrical/spectral/Haar/topology/moment 특징에 도함수 기반 변동성·곡률 특징까지 모두 합쳐, 라벨을 쓰지 않는 대규모 고정 특징 테이블을 구성."""
    p = smooth_log_profile(x_norm)
    d1 = np.gradient(p, axis=1)
    d2 = np.gradient(d1, axis=1)
    dfs = [
        extract_visual_features(x_norm).add_prefix("visual_"),
        extract_cylindrical_features(x_norm).add_prefix("cyl_"),
        spectral_features(p).add_prefix("spectral_"),
        haar_features(p).add_prefix("haar_"),
        run_length_features(p).add_prefix("topology_"),
        moment_features(p).add_prefix("moment_"),
        pd.DataFrame(
            {
                "deriv_total_variation": np.sum(np.abs(d1), axis=1),
                "deriv_positive_variation_78_110": np.sum(np.maximum(d1[:, 77:110], 0.0), axis=1),
                "deriv_negative_variation_42_78": np.sum(np.maximum(-d1[:, 41:78], 0.0), axis=1),
                "deriv_max_upstroke_78_110": np.max(d1[:, 77:110], axis=1),
                "curv_total_energy": np.mean(d2**2, axis=1),
                "curv_late_energy_100_128": np.mean(d2[:, 99:128] ** 2, axis=1),
            }
        ),
    ]
    return pd.concat(dfs, axis=1)


def summarize_all_features(
    feature_tables: dict[str, pd.DataFrame],
    datasets: dict[str, dict],
    extra_scores: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """모든 고정 특징과 매칭필터 점수에 대해 코호트별 통계를 계산하고, 두 코호트 방향 일치·최소 AUC
    거리·pooled p값 기준으로 정렬해 "가장 일관되게 유망한" 특징 순서를 매김."""
    rows = []
    feature_names = list(feature_tables["g1090"].columns)
    for name in feature_names:
        vals = {cohort: feature_tables[cohort][name].to_numpy(dtype=float) for cohort in feature_tables}
        rows.extend(summarize_feature(name, vals, datasets))
    for name, vals in extra_scores.items():
        rows.extend(summarize_feature(name, vals, datasets))
    out = pd.DataFrame(rows)

    wide = out.pivot_table(index="feature", columns="cohort", values="auc_if_higher_predicts_low_smi", aggfunc="first")
    delta = out.pivot_table(index="feature", columns="cohort", values="delta_low_minus_nonlow", aggfunc="first")
    consistency = (
        np.sign(delta["g1090"]) == np.sign(delta["sdata"])
    ) & np.isfinite(delta["g1090"]) & np.isfinite(delta["sdata"])
    min_sep = pd.concat([(wide["g1090"] - 0.5).abs(), (wide["sdata"] - 0.5).abs()], axis=1).min(axis=1)
    pooled_p = out[out["cohort"].eq("pooled")].set_index("feature")["mannwhitney_p"]
    rank = pd.DataFrame({"consistent": consistency, "min_auc_distance": min_sep, "pooled_p": pooled_p}).sort_values(
        ["consistent", "min_auc_distance", "pooled_p"], ascending=[False, False, True]
    )
    order = rank.index.tolist()
    out["direction_consistent_g1090_sdata"] = out["feature"].map(consistency.to_dict())
    out["min_abs_auc_distance_g1090_sdata"] = out["feature"].map(min_sep.to_dict())
    out["feature_order"] = out["feature"].map({f: i for i, f in enumerate(order)})
    return out.sort_values(["feature_order", "cohort"]).drop(columns=["feature_order"])


def add_matched_filter_scores(datasets: dict[str, dict]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """로그 프로파일 자체/중간+후반 구간/후반 전용 구간 마스크와, 1차/2차 도함수에 대해 각각 매칭필터
    점수를 계산해, "atlas" 접두사가 붙은 5개 매칭필터 특징과 그 템플릿을 반환."""
    train = datasets["g1090"]
    test = datasets["sdata"]
    p_tr = smooth_log_profile(train["x"])
    p_te = smooth_log_profile(test["x"])
    d1_tr = np.gradient(p_tr, axis=1)
    d1_te = np.gradient(p_te, axis=1)
    d2_tr = np.gradient(d1_tr, axis=1)
    d2_te = np.gradient(d1_te, axis=1)

    masks = {}
    mid_late = np.zeros(128, dtype=float)
    mid_late[41:78] = 1.0
    mid_late[99:128] = 1.0
    late = np.zeros(128, dtype=float)
    late[99:128] = 1.0
    masks["atlas_log_profile_score"] = None
    masks["atlas_log_mid_late_score"] = mid_late
    masks["atlas_log_late_only_score"] = late

    scores: dict[str, dict[str, np.ndarray]] = {}
    templates: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        oof, ext, templ = matched_filter_scores(p_tr, train["y"], p_te, mask=mask)
        scores[name] = {"g1090": oof, "sdata": ext}
        templates[name] = templ

    oof, ext, templ = matched_filter_scores(d1_tr, train["y"], d1_te, mask=mid_late)
    scores["atlas_derivative_mid_late_score"] = {"g1090": oof, "sdata": ext}
    templates["atlas_derivative_mid_late_score"] = templ

    oof, ext, templ = matched_filter_scores(d2_tr, train["y"], d2_te, mask=mid_late)
    scores["atlas_curvature_mid_late_score"] = {"g1090": oof, "sdata": ext}
    templates["atlas_curvature_mid_late_score"] = templ
    return scores, templates


def plot_deep_feature_outputs(
    datasets: dict[str, dict],
    feature_tables: dict[str, pd.DataFrame],
    extra_scores: dict[str, dict[str, np.ndarray]],
    templates: dict[str, np.ndarray],
) -> None:
    """매칭필터 템플릿들의 모양을 겹쳐 그린 그래프와, 대표 특징 4개의 코호트별 분포 히스토그램들을 PNG로 저장."""
    z = np.arange(1, 129)
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    for name, color in [
        ("atlas_log_profile_score", "#333333"),
        ("atlas_log_mid_late_score", "#C84630"),
        ("atlas_derivative_mid_late_score", "#4C78A8"),
    ]:
        ax.plot(z, templates[name], lw=2.0, label=name, color=color)
    ax.axhline(0, color="#555555", ls="--", lw=0.9)
    ax.axvspan(42, 78, color="#2F6F73", alpha=0.10)
    ax.axvspan(100, 128, color="#C84630", alpha=0.10)
    ax.set_xlabel("AEC_128 point index")
    ax.set_ylabel("Unit template weight")
    ax.set_title("Train-derived matched-filter templates", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "deep_matched_filter_templates.png", dpi=200)
    plt.close(fig)

    selected = [
        "atlas_log_mid_late_score",
        "atlas_log_profile_score",
        "cyl_cyl_late_positive_plus_mid_negative",
        "visual_aec128_rebound_height_peak_minus_valley",
    ]
    fig, axes = plt.subplots(len(selected), 2, figsize=(11.8, 10.0), sharey=False)
    for row, feature in enumerate(selected):
        for col, cohort in enumerate(["g1090", "sdata"]):
            ax = axes[row, col]
            y = datasets[cohort]["y"]
            if feature.startswith("atlas_"):
                vals = extra_scores[feature][cohort]
            else:
                vals = feature_tables[cohort][feature].to_numpy(dtype=float)
            ax.hist(vals[~y], bins=36, density=True, color="#2F6F73", alpha=0.55, label="Non-low SMI")
            ax.hist(vals[y], bins=24, density=True, color="#C84630", alpha=0.58, label="Low SMI")
            ax.set_title(f"{cohort}: {feature}", loc="left", fontsize=9.5, fontweight="bold")
            ax.grid(alpha=0.22)
            if row == len(selected) - 1:
                ax.set_xlabel("Feature value")
            if col == 0:
                ax.set_ylabel("Density")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "deep_feature_distributions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 지금까지 만든 시각/원기둥 특징에, 스펙트럼/웨이블릿/
    위상학/모멘트/매칭필터 특징까지 대량으로 더 캐보면 g1090·sdata 양쪽에서 일관된 새로운
    특징이 더 나오는가? — 이름 그대로 "특징 대량 채굴"):

    1. g1090/sdata를 로드하고, fixed_deep_features로 visual+cylindrical+spectral+Haar+
       topology+moment+도함수 특징을 모두 합친 대규모 특징 테이블(라벨 미사용)을 만든다.
    2. add_matched_filter_scores로 g1090에서 만든 "저근감소증-비저근감소증 평균차" 템플릿을
       (로그 프로파일 자체/중간+후반 마스크/후반만/1차 도함수/2차 도함수 5가지 버전으로) 만들어
       train은 OOF로, 외부는 고정 템플릿으로 매칭필터 점수를 계산한다.
    3. summarize_all_features로 모든 특징+매칭필터 점수를 코호트별 통계와 함께 정리하고,
       두 코호트에서 방향이 일치하며 AUC가 0.5에서 가장 멀리 떨어진 "가장 일관된" 특징들을
       상위 30개 뽑아 텍스트 파일로 저장.
    4. 매칭필터 템플릿 모양과 대표 특징들의 분포를 그래프로 저장.
    5. 방법론 설명(고정 특징은 라벨 미사용, 매칭필터는 g1090 템플릿 기반)과 "최종 후보 선택 기준"을
       JSON으로 저장하고, 상위 12개 특징의 통계를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}
    feature_tables = {cohort: fixed_deep_features(d["x"]) for cohort, d in datasets.items()}
    extra_scores, templates = add_matched_filter_scores(datasets)

    for cohort, df in feature_tables.items():
        out = df.copy()
        for name, vals in extra_scores.items():
            out[name] = vals[cohort]
        out.insert(0, "low_smi", datasets[cohort]["y"].astype(int))
        out.to_csv(OUT_DIR / f"{cohort}_aec128_deep_features_patient_level.csv", index=False)

    stats_df = summarize_all_features(feature_tables, datasets, extra_scores)
    stats_df.to_csv(OUT_DIR / "aec128_deep_feature_stats.csv", index=False)

    consistent_mask = stats_df["direction_consistent_g1090_sdata"].fillna(False).astype(bool)
    top = (
        stats_df[consistent_mask & stats_df["min_abs_auc_distance_g1090_sdata"].notna()]
        .sort_values(["min_abs_auc_distance_g1090_sdata", "mannwhitney_p"], ascending=[False, True])
        .groupby("feature", as_index=False)
        .first()
        .sort_values("min_abs_auc_distance_g1090_sdata", ascending=False)
        .head(30)
    )
    top["feature"].to_csv(OUT_DIR / "aec128_top30_consistent_deep_features.txt", index=False, header=False)
    plot_deep_feature_outputs(datasets, feature_tables, extra_scores, templates)

    definitions = {
        "unsupervised_fixed_features": "visual/cylindrical/spectral/Haar/topology/moment features are computed from normalized smoothed log AEC_128 without using outcome labels.",
        "matched_filter_features": "atlas scores use g1090 low-minus-nonlow templates. g1090 score is 5-fold OOF; sdata score uses the template fitted on all g1090.",
        "main_deep_feature_candidate": "Choose the feature with consistent direction in g1090 and sdata and largest minimum AUC distance from 0.5 across the two cohorts.",
    }
    with open(OUT_DIR / "aec128_deep_feature_definitions.json", "w", encoding="utf-8") as f:
        json.dump(definitions, f, ensure_ascii=False, indent=2)

    selected = stats_df[stats_df["feature"].isin(top["feature"].head(12))]
    print(selected.to_string(index=False))
    print(OUT_DIR / "aec128_deep_feature_stats.csv")
    print(OUT_DIR / "deep_matched_filter_templates.png")
    print(OUT_DIR / "deep_feature_distributions.png")


if __name__ == "__main__":
    main()
