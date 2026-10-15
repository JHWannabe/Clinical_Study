from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# Grad-CAM 설명가능성 심화(학습 없음, {질환}_best.pt 사용). clinic4_landmark_image_cam_analysis의 단순 비중 비교를 보완한다.
# (1) 조직 분류 검증: overlay 색으로 센 조직 픽셀 수 vs 엑셀 landmark 단면 면적(VAT/SAT/TAMA)의 Spearman 상관
# (2) enrichment = (조직 위 CAM 비중) / (몸 안 그 조직의 면적 비중). 1이면 면적만큼만 봄, >1이면 그 조직을 선호. 양성 vs 음성 Mann-Whitney + landmark x 조직 히트맵(BH-FDR)
# (3) 보정 회귀: CAM 조직 비중 ~ 클래스 + 조직 면적 비중 + 성별/나이/신장/체중 -> 클래스 계수 p (면적/임상 교란 제거 후에도 남는지)
# (4) occlusion(faithfulness): 조직 픽셀 / CAM 상위 15% 픽셀을 검게 가렸을 때 점수 하락 vs 같은 개수 무작위 픽셀, paired Wilcoxon
# 주의: 조직은 overlay 색 기반 근사, occlusion은 overlay+해부 이미지를 함께 가리므로 해부학적 인과가 아닌 '모델 의존성'을 본다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

import clinic4_landmark_image_cam_analysis as A
import clinic4_landmark_image_cnn as M
from clinic4_landmark_vat_auc import holdout_split, load_cohort, matrix
from clinic4_logistic_regression import DISEASES, SEED, balanced_idx, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
OUT = M.FIG_DIR / "cam_xai"
TS = ["VAT", "SAT", "muscle"]  # tissue_map 인덱스 0/1/2 (3=기타)
XLSX_TISSUE = {"VAT": "VAT", "SAT": "SAT", "muscle": "TAMA"}
LM_FULL = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center",
           "S1_center", "femoral_head_center", "inferior_pubic_margin"]
TOP, CHUNK = 0.15, 64
CONDS = [f"{t}{s}" for s in ("", "_rand") for t in TS] + ["CAMtop", "CAMtop_rand"]


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def mwu(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    return stats.mannwhitneyu(a, b).pvalue if min(len(a), len(b)) >= 10 else np.nan


# 환자 1명: enrichment(조직4, landmark x 조직3), CAM 조직 비중, 면적 비중, 조직 픽셀 수(검증용), occlusion 마스크
def per_patient(net, img1, x1, rng):
    cam = M.gradcam(net, img1, x1, normalize=False)
    cam = cam / max(cam.sum(), 1e-12)
    mk, tm = M.body_mask(img1), A.tissue_map(img1)
    tis = [(tm == k) & mk for k in range(4)]
    share = np.array([cam[t].sum() for t in tis])
    area = np.array([t.sum() for t in tis]) / max(mk.sum(), 1)
    enr = np.where(area > 0.01, share / np.maximum(area, 1e-9), np.nan)
    mass_j = cam.sum((1, 2))
    area_jk = np.stack([[tis[k][j].sum() / max(mk[j].sum(), 1) for k in range(3)] for j in range(12)])
    mass_jk = np.stack([[cam[j][tis[k][j]].sum() for k in range(3)] for j in range(12)])
    enr_jk = np.where((area_jk > 0.02) & (mass_j[:, None] > 0), mass_jk / (mass_j[:, None] * np.maximum(area_jk, 1e-9)), np.nan)
    counts = np.stack([[tis[k][j].sum() for k in range(3)] for j in range(12)])
    pool = np.flatnonzero(mk.ravel())

    def rand_like(n):  # 몸 안에서 같은 개수의 무작위 픽셀
        m = np.zeros(mk.size, bool)
        m[rng.choice(pool, min(n, len(pool)), replace=False)] = True
        return m.reshape(mk.shape)

    top = (cam >= np.quantile(cam[mk], 1 - TOP)) & mk
    masks = {**{TS[k]: tis[k] for k in range(3)}, **{f"{TS[k]}_rand": rand_like(tis[k].sum()) for k in range(3)}, "CAMtop": top, "CAMtop_rand": rand_like(top.sum())}
    return share, area, enr, enr_jk, counts, masks


def ols_class_p(y, cls, covs):  # y ~ 1 + class + covs, class 계수의 p(양측 t)
    X = np.column_stack([np.ones(len(y)), cls, (covs - covs.mean(0)) / (covs.std(0) + 1e-9)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    dof = len(y) - X.shape[1]
    se = np.sqrt((r @ r / dof) * np.linalg.inv(X.T @ X)[1, 1])
    return b[1], 2 * stats.t.sf(abs(b[1] / se), dof)


def analyse(disease, cohort, net, img, x, y, clin):
    rng = np.random.default_rng(SEED)
    sh, ar, en, ejk, cnt, s0, occ = [], [], [], [], [], [], {c: [] for c in CONDS}
    for st in range(0, len(y), CHUNK):
        ids = range(st, min(st + CHUNK, len(y)))
        res = [per_patient(net, img[i], x[i], rng) for i in ids]
        for r in res:
            sh.append(r[0]), ar.append(r[1]), en.append(r[2]), ejk.append(r[3]), cnt.append(r[4])
        base = M.predict(net, img[list(ids)], x[list(ids)])
        s0.append(base)
        for c in CONDS:
            im = img[list(ids)].copy()
            im[np.stack([r[5][c] for r in res])] = 0
            occ[c].append(base - M.predict(net, im, x[list(ids)]))  # 점수 하락량(양수 = 가리면 점수가 떨어짐)
    sh, ar, en, ejk, cnt = map(np.array, (sh, ar, en, ejk, cnt))
    occ = {c: np.concatenate(v) for c, v in occ.items()}
    pos, neg = y == 1, y == 0
    rows = {"enrich": [], "adj": [], "occ": [], "valid": []}

    for k, t in enumerate(TS + ["other"]):  # (2) enrichment + (3) 보정 회귀
        p, n = en[pos, k], en[neg, k]
        rows["enrich"].append({"disease": disease, "cohort": cohort, "tissue": t, "pos_mean": np.nanmean(p), "neg_mean": np.nanmean(n),
                               "pos_ci": A.boot_ci(p[~np.isnan(p)]), "neg_ci": A.boot_ci(n[~np.isnan(n)]), "p_mwu": mwu(p, n)})
        b, pv = ols_class_p(sh[:, k], y.astype(float), np.column_stack([ar[:, k], clin]))
        rows["adj"].append({"disease": disease, "cohort": cohort, "tissue": t, "class_coef_adj": b, "p_adj": pv,
                            "p_unadj": mwu(sh[pos, k], sh[neg, k])})
    for r, q in zip(rows["enrich"], bh_fdr(np.array([r["p_mwu"] for r in rows["enrich"]]))): r["p_fdr"] = q
    for r, q in zip(rows["adj"], bh_fdr(np.array([r["p_adj"] for r in rows["adj"]]))): r["p_adj_fdr"] = q

    hm_d = np.array([[np.nanmean(ejk[pos, j, k]) - np.nanmean(ejk[neg, j, k]) for k in range(3)] for j in range(12)])
    hm_p = np.array([[mwu(ejk[pos, j, k], ejk[neg, j, k]) for k in range(3)] for j in range(12)])
    ok = ~np.isnan(hm_p)
    hm_q = np.full_like(hm_p, np.nan)
    hm_q[ok] = bh_fdr(hm_p[ok])

    for t in TS + ["CAMtop"]:  # (4) occlusion
        for g, m in (("all", np.ones(len(y), bool)), ("pos", pos), ("neg", neg)):
            d = occ[t][m] - occ[f"{t}_rand"][m]
            rows["occ"].append({"disease": disease, "cohort": cohort, "target": t, "group": g, "drop_target": occ[t][m].mean(),
                                "drop_random": occ[f"{t}_rand"][m].mean(), "p_wilcoxon": stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan})

    return rows, (hm_d, hm_q), cnt, occ


def validate(cnt, xl, disease, cohort):  # xl: DataFrame(landmark 단면 면적 컬럼 포함)
    out = []
    for k, t in enumerate(TS):
        rho = [stats.spearmanr(cnt[:, j, k], xl[f"{XLSX_TISSUE[t]}@{LM_FULL[j]}"]).statistic for j in range(12)]
        out.append({"disease": disease, "cohort": cohort, "tissue": t, "rho_median": np.nanmedian(rho), "rho_min": np.nanmin(rho), "rho_max": np.nanmax(rho)})
    return out


def plot(disease, cohort, rows, hm, occ_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    e = pd.DataFrame(rows["enrich"])
    xs = np.arange(len(e))
    for off, k, lab in ((-0.2, "pos", "positive"), (0.2, "neg", "negative")):
        ci = np.array(e[f"{k}_ci"].tolist())
        ax[0, 0].bar(xs + off, e[f"{k}_mean"], 0.4, yerr=[e[f"{k}_mean"] - ci[:, 0], ci[:, 1] - e[f"{k}_mean"]], capsize=2, label=lab)
    ax[0, 0].axhline(1, c="k", ls=":"), ax[0, 0].set_xticks(xs), ax[0, 0].set_xticklabels(e["tissue"])
    for j, q in enumerate(e["p_fdr"]):
        ax[0, 0].text(j, max(np.array(e["pos_ci"].tolist())[j, 1], np.array(e["neg_ci"].tolist())[j, 1]) * 1.02, star(q), ha="center")
    ax[0, 0].set(title="CAM enrichment by tissue (1 = proportional to area)", ylabel="CAM share / area share"), ax[0, 0].legend()

    hd, hq = hm
    v = np.nanmax(np.abs(hd))
    im = ax[0, 1].imshow(hd, cmap="bwr", vmin=-v, vmax=v, aspect="auto")
    for j in range(12):
        for k in range(3):
            ax[0, 1].text(k, j, star(hq[j, k]) if not np.isnan(hq[j, k]) else "", ha="center", va="center", fontsize=11)
    ax[0, 1].set_xticks(range(3)), ax[0, 1].set_xticklabels(TS), ax[0, 1].set_yticks(range(12)), ax[0, 1].set_yticklabels(M.LM_NAMES)
    ax[0, 1].set_title("enrichment diff (pos - neg), red = pos higher; * BH-FDR<0.05"), fig.colorbar(im, ax=ax[0, 1])

    o = pd.DataFrame(rows["occ"])
    o = o[o.group != "all"]
    targets = TS + ["CAMtop"]
    for gi, (g, lab) in enumerate((("pos", "positive"), ("neg", "negative"))):
        d = o[o.group == g].set_index("target")
        ax[1, 0].bar(np.arange(4) * 3 + gi * 1.2 - 0.3 + 0, d.loc[targets, "drop_target"], 0.5, label=f"{lab}: occlude target")
        ax[1, 0].bar(np.arange(4) * 3 + gi * 1.2 + 0.3 + 0.0, d.loc[targets, "drop_random"], 0.5, color="lightgray" if gi == 0 else "silver", label=f"{lab}: random same-size" if True else None)
        for ti, t in enumerate(targets):
            ax[1, 0].text(ti * 3 + gi * 1.2, d.loc[t, "drop_target"], star(d.loc[t, "p_wilcoxon"]), ha="center", va="bottom")
    ax[1, 0].set_xticks(np.arange(4) * 3 + 0.6), ax[1, 0].set_xticklabels(["VAT", "SAT", "muscle", f"CAM top {int(TOP * 100)}%"]), ax[1, 0].axhline(0, c="k", lw=0.5)
    ax[1, 0].set(title="score drop when occluded (target vs random same-size; * paired Wilcoxon)", ylabel="mean drop in predicted probability"), ax[1, 0].legend(fontsize=7)

    ax[1, 1].axis("off")
    a = pd.DataFrame(rows["adj"])
    txt = "Adjusted class effect on CAM tissue share\n(share ~ class + tissue area + sex/age/height/weight)\n\n" + a.assign(
        unadj=a.p_unadj.map("{:.1e}".format), adj=a.p_adj.map("{:.1e}".format), fdr=a.p_adj_fdr.map("{:.1e}".format), coef=a.class_coef_adj.map("{:+.3f}".format)
    )[["tissue", "coef", "unadj", "adj", "fdr"]].to_string(index=False)
    v = pd.DataFrame(rows["valid"])
    if len(v):
        txt += "\n\nTissue-colour validation (Spearman, pixel count vs xlsx area; median [min,max] over 12 landmarks)\n" + "\n".join(
            f"{r.tissue}: {r.rho_median:.2f} [{r.rho_min:.2f}, {r.rho_max:.2f}]" for r in v.itertuples())
    ax[1, 1].text(0, 1, txt, va="top", family="monospace", fontsize=8.5)
    fig.suptitle(f"{disease} {cohort}: Grad-CAM explainability"), fig.tight_layout()
    fig.savefig(OUT / f"{disease}_{cohort}_xai.png", dpi=140), plt.close(fig)


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    df = df[[(M.IMG_DIR["gangnam"] / f"{p}.png").exists() for p in df["PatientID"]]].reset_index(drop=True)
    df_ext = df_ext[[(M.IMG_DIR["sinchon"] / f"{p}.png").exists() for p in df_ext["PatientID"]]].reset_index(drop=True)
    img_full, img_ext_full = M.load_panels("gangnam", df["PatientID"].tolist()), M.load_panels("sinchon", df_ext["PatientID"].tolist())
    x_full, sc = matrix(df, [])
    x_ext_full, _ = matrix(df_ext, [], sc)
    clin = lambda d: np.column_stack([(d["PatientSex"].astype(str).str.upper() == "M").astype(float), d[["PatientAge", "Height", "Weight"]].to_numpy(float)])
    tabs = {k: [] for k in ("enrich", "adj", "occ", "valid")}
    for d in DISEASES:
        rng = np.random.default_rng(SEED)  # CNN 학습과 같은 balanced/split 재현
        y_full = df[d].to_numpy(int)
        idx = balanced_idx(y_full, rng)
        y_ext_full = df_ext[d].to_numpy(int)
        idx_ext = balanced_idx(y_ext_full, rng)
        _, _, te = holdout_split(y_full[idx])
        net = M.Net(x_full.shape[1]).to(M.DEV)
        net.load_state_dict(torch.load(M.FIG_DIR / f"{d}_best.pt"))
        net.eval()
        sub = df.iloc[idx].iloc[te].reset_index(drop=True)
        sub_ext = df_ext.iloc[idx_ext].reset_index(drop=True)
        for cohort, im, xx, yy, meta in (("gangnam_test", img_full[idx][te], x_full[idx][te], y_full[idx][te], sub),
                                         ("sinchon_external", img_ext_full[idx_ext], x_ext_full[idx_ext], y_ext_full[idx_ext], sub_ext)):
            rows, hm, cnt, _ = analyse(d, cohort, net, im, xx, yy, clin(meta))
            rows["valid"] = validate(cnt, meta, d, cohort)
            plot(d, cohort, rows, hm, rows["occ"])
            for k in tabs:
                tabs[k] += rows[k]
            e, a, o = pd.DataFrame(rows["enrich"]), pd.DataFrame(rows["adj"]), pd.DataFrame(rows["occ"])
            print(f"[{d}/{cohort}] n={len(yy)}")
            print(e[["tissue", "pos_mean", "neg_mean", "p_fdr"]].round(3).to_string(index=False))
            print(a[["tissue", "class_coef_adj", "p_adj", "p_adj_fdr"]].round(4).to_string(index=False))
            print(o[o.group == "all"][["target", "drop_target", "drop_random", "p_wilcoxon"]].round(4).to_string(index=False))
            print(pd.DataFrame(rows["valid"])[["tissue", "rho_median", "rho_min", "rho_max"]].round(2).to_string(index=False))
    flat = lambda rs: pd.DataFrame(rs).drop(columns=[c for c in ("pos_ci", "neg_ci") if c in pd.DataFrame(rs).columns])
    save_sheet(flat(tabs["enrich"]), M.OUT_XLSX, "xai_enrichment")
    save_sheet(flat(tabs["adj"]), M.OUT_XLSX, "xai_adjusted")
    save_sheet(flat(tabs["occ"]), M.OUT_XLSX, "xai_occlusion")
    save_sheet(flat(tabs["valid"]), M.OUT_XLSX, "xai_tissue_validation")


if __name__ == "__main__":
    main()
