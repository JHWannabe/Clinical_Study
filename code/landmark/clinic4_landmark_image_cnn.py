from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark_report 이미지로 ResNet18을 end-to-end fine-tuning하는 모델(질병별 1개).
# 12패널을 같은 backbone에 통과 -> 패널 특징 평균 -> [성별,나이,신장,체중]과 concat -> 선형 head -> logit.
# 평가 프로토콜은 clinic4_landmark_vat_auc와 동일: gangnam balanced 7/1/2 split, valid AUC 최고 epoch을 test로,
# 같은 모델(trainval 재학습 없음)을 sinchon balanced에 적용. 패널은 PX로 줄여 uint8 캐시.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_auc_score

from clinic4_landmark_image_auc import IMG_DIR, S, XS, YS
from clinic4_landmark_vat_auc import load_cohort, matrix, run_disease_holdout
from delong_utils import delong_paired_auc_test
from clinic4_landmark_vat_auc import holdout_split
from clinic4_logistic_regression import CLINIC4_DIR, DATA_DIR, sens_spec, youden_threshold, DISEASES, SEED, balanced_idx, f3, save_sheet

sys.stdout.reconfigure(encoding="utf-8")

PX, EPOCHS, BATCH = 160, 200, 16
LR_FACTOR, LR_PATIENCE = 0.5, 5  # valid loss 5 epoch 개선 없으면 lr 절반
MIN_EPOCHS = 100  # 이 epoch까지는 early stopping 금지, 이후 PATIENCE epoch 연속 valid loss 미개선이면 중단
LR_CNN, LR_HEAD, WD, PATIENCE, SMOOTH = 5e-6, 2e-4, 5e-2, 10, 0.1  # 과적합 방지: backbone은 작은 lr, weight decay/label smoothing, valid loss early stopping
OUT_XLSX = "landmark/landmark_image_cnn.xlsx"
FIG_DIR = CLINIC4_DIR / "landmark/image_cnn"
GRADCAM_ONLY = bool(os.environ.get("GRADCAM_ONLY"))  # 1이면 학습 없이 저장된 {질환}_best.pt로 Grad-CAM만 생성
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


# 코호트의 (N,12,PX,PX,3) uint8 패널 배열. npy 캐시
def load_panels(cohort: str, ids: list) -> np.ndarray:
    path = DATA_DIR / f"landmark_report_panels{PX}_{cohort}.npy"
    key = DATA_DIR / f"landmark_report_panels{PX}_{cohort}_ids.npy"
    if path.exists() and key.exists() and (np.load(key).astype(str) == np.array(ids, dtype=str)).all():
        return np.load(path)
    arr = np.stack([np.stack([np.asarray(Image.open(IMG_DIR[cohort] / f"{p}.png").convert("RGB").crop((x, y, x + S, y + S)).resize((PX, PX)))
                              for y in YS for x in XS]) for p in ids])
    np.save(path, arr), np.save(key, np.array(ids, dtype=str))
    return arr


class Net(nn.Module):
    def __init__(self, n_clin: int):
        super().__init__()
        self.cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(512 + n_clin, 1))

    def forward(self, img, clin):  # img (B,12,PX,PX,3) uint8
        b = img.shape[0]
        t = (img.flatten(0, 1).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
        if self.training:  # 패널마다 작은 회전/이동/확대 + 밝기 변동(해부학 방향이 의미 있어 좌우반전은 쓰지 않음)
            n = len(t)
            ang, sc = (torch.rand(n, device=t.device) - .5) * 2 * 0.09, 1 + (torch.rand(n, device=t.device) - .5) * 0.2
            theta = torch.zeros(n, 2, 3, device=t.device)
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = sc * ang.cos(), -sc * ang.sin(), sc * ang.sin(), sc * ang.cos()
            theta[:, :, 2] = (torch.rand(n, 2, device=t.device) - .5) * 0.12
            t = nn.functional.grid_sample(t, nn.functional.affine_grid(theta, t.shape, align_corners=False), padding_mode="zeros", align_corners=False)
            t = t * (1 + (torch.rand(n, 1, 1, 1, device=t.device) - .5) * 0.2) + (torch.rand(n, 1, 1, 1, device=t.device) - .5) * 0.2
        f = self.cnn(t).view(b, 12, -1).mean(1)
        return self.head(torch.cat([f, clin], 1)).squeeze(1)


@torch.no_grad()
def predict(net, img, clin) -> np.ndarray:
    net.eval()
    out = [torch.sigmoid(net(torch.from_numpy(img[i:i + BATCH]).to(DEV), torch.from_numpy(clin[i:i + BATCH]).float().to(DEV))).cpu().numpy()
           for i in range(0, len(img), BATCH)]
    return np.concatenate(out)


def run(disease, img_full, x_full, df, img_ext_full, x_ext_full, df_ext):
    rng = np.random.default_rng(SEED)
    y_full = df[disease].to_numpy(int)
    idx = balanced_idx(y_full, rng)
    y = y_full[idx]
    y_ext_full = df_ext[disease].to_numpy(int)
    idx_ext = balanced_idx(y_ext_full, rng)
    y_ext = y_ext_full[idx_ext]
    img, x, img_ext, x_ext = img_full[idx], x_full[idx], img_ext_full[idx_ext], x_ext_full[idx_ext]
    tr, va, te = holdout_split(y)

    torch.manual_seed(SEED)
    net = Net(x.shape[1]).to(DEV)
    pid = df["PatientID"].to_numpy()[idx][te]
    if GRADCAM_ONLY:
        net.load_state_dict(torch.load(FIG_DIR / f"{disease}_best.pt"))
        plot_gradcam(disease, net, img[te], x[te], y[te], predict(net, img[te], x[te]), pid)
        return None
    opt = torch.optim.AdamW([{"params": net.cnn.parameters(), "lr": LR_CNN}, {"params": net.head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    warm = int(np.ceil(len(tr) / BATCH))  # 첫 epoch은 lr warmup(배치 단위), 이후 valid loss가 LR_PATIENCE epoch 개선 없으면 lr x LR_FACTOR
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=1e-8)
    step = 0
    best, best_state, hist, bad = np.inf, None, [], 0  # best = valid loss(작을수록 좋음). n이 작은 valid에서 AUC보다 안정적
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.default_rng(SEED + ep).permutation(tr)
        losses = []
        bar = tqdm(range(0, len(perm), BATCH), desc=f"{disease} epoch {ep + 1}/{EPOCHS}", leave=False)
        for i in bar:
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(torch.from_numpy(img[b]).to(DEV), torch.from_numpy(x[b]).float().to(DEV)),
                torch.from_numpy(y[b]).float().to(DEV) * (1 - SMOOTH) + SMOOTH / 2)  # label smoothing
            opt.zero_grad(), loss.backward(), opt.step()
            if step < warm:
                warmup.step()
            step += 1
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses):.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}")  # backbone lr
        v = roc_auc_score(y[va], predict(net, img[va], x[va]))
        vl = nn.functional.binary_cross_entropy(torch.from_numpy(predict(net, img[va], x[va])).float().clamp(1e-6, 1 - 1e-6), torch.from_numpy(y[va]).float()).item()
        plateau.step(vl)  # valid loss를 학습 lr에 반영
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "valid_loss": vl, "valid_auc": v,
                     "train_auc": roc_auc_score(y[tr], predict(net, img[tr], x[tr])), "lr_cnn": opt.param_groups[0]["lr"]})
        if vl < best:
            best, best_state, best_ep, bad = vl, {k: t.clone() for k, t in net.state_dict().items()}, ep + 1, 0
        else:
            bad += 1
        print(f"  [{disease}] epoch {ep + 1}/{EPOCHS} train_loss={np.mean(losses):.3f} valid_loss={vl:.3f} valid_auc={v:.3f}" + (" *" if bad == 0 else ""))
        if ep + 1 >= MIN_EPOCHS and bad >= PATIENCE:
            print(f"  [{disease}] early stop (best epoch {best_ep})")
            break
    net.load_state_dict(best_state)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, FIG_DIR / f"{disease}_best.pt")  # 이미지 재생성용 가중치
    s_te, s_ext = predict(net, img[te], x[te]), predict(net, img_ext, x_ext)
    row = {"disease": disease, "model": "ResNet18 fine-tune (12 panels + clinic4)", "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
           "n_external": len(y_ext), "best_epoch": best_ep, "valid_loss": best, "valid_auc": hist[best_ep - 1]["valid_auc"],
           "internal_auc": roc_auc_score(y[te], s_te), "external_auc": roc_auc_score(y_ext, s_ext)}
    # baseline(clinic4 로지스틱, 같은 환자/split) 성능과 DeLong 비교. run_disease_holdout은 같은 rng 순서로 같은 부분집합을 만든다
    brow, bpred = run_disease_holdout(disease, "baseline", x_full, df, x_ext_full, df_ext)
    b_te, b_ext = bpred[bpred.cohort == "gangnam"]["score"].to_numpy(), bpred[bpred.cohort == "sinchon"]["score"].to_numpy()
    row.update({"base_valid_auc": brow["valid_auc"], "base_internal_auc": brow["internal_auc"], "base_external_auc": brow["external_auc"]})
    for name, yy, ss, bb in (("internal", y[te], s_te, b_te), ("external", y_ext, s_ext, b_ext)):
        row[f"{name}_p_vs_baseline"] = delong_paired_auc_test(yy, bb, ss)["p_value"]
    thr = youden_threshold(y[va], predict(net, img[va], x[va]))  # cutoff는 valid에서 정하고 test/external에 고정
    row["cutoff"] = thr
    for name, yy, ss in (("internal", y[te], s_te), ("external", y_ext, s_ext)):
        row[f"{name}_sens"], row[f"{name}_spec"] = sens_spec(yy, ss, thr)
    print(f"[{disease}] valid={f3(row['valid_auc'])} test={f3(row['internal_auc'])} external={f3(row['external_auc'])}")
    plot_disease(disease, pd.DataFrame(hist), row, (y[te], s_te, b_te), (y_ext, s_ext, b_ext), thr)
    plot_gradcam(disease, net, img[te], x[te], y[te], s_te, pid)
    pred = pd.concat([pd.DataFrame({"disease": disease, "cohort": "gangnam", "patient_id": df["PatientID"].to_numpy()[idx][te], "y": y[te], "score": s_te}),
                      pd.DataFrame({"disease": disease, "cohort": "sinchon", "patient_id": df_ext["PatientID"].to_numpy()[idx_ext], "y": y_ext, "score": s_ext})])
    return row, pd.DataFrame(hist).assign(disease=disease), pred


LM_NAMES = ["liver_dome", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "femoral_head", "pubic_margin"]


# 환자 1명(12패널)의 Grad-CAM: 마지막 conv(layer4)의 활성 x 정답 logit gradient -> ReLU -> PX로 upsample. 반환 (12,PX,PX) in [0,1](환자 내 최대값 기준)
CAM_LAYER = "layer3"  # layer4는 160px에서 5x5라 너무 뭉툭 -> layer3(10x10)


# 패널별 몸 영역 mask(밝기>임계 -> 가장 큰 연결 성분 -> 내부 구멍(폐 등) 채움). CT 테이블/배경에 CAM이 번지는 것을 제거
def body_mask(img1) -> np.ndarray:
    out = []
    for p in img1:
        lab, n = ndimage.label(p.max(-1) > 25)
        out.append(ndimage.binary_fill_holes(lab == (1 + np.argmax(ndimage.sum(lab > 0, lab, range(1, n + 1))))) if n else np.ones(p.shape[:2], bool))
    return np.stack(out)


def gradcam(net, img1, x1, normalize: bool = True) -> np.ndarray:
    net.eval()
    store = {}
    def hook(m, i, o):  # 출력은 바꾸지 않고(None 반환) 활성/gradient만 저장
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))
    h = getattr(net.cnn, CAM_LAYER).register_forward_hook(hook)
    try:
        net.zero_grad()
        net(torch.from_numpy(img1[None]).to(DEV), torch.from_numpy(x1[None]).float().to(DEV)).sum().backward()
    finally:
        h.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=(PX, PX), mode="bilinear", align_corners=False)[:, 0].detach().cpu().numpy()
    cam = cam * body_mask(img1)
    return cam / max(cam.max(), 1e-8) if normalize else cam


# 고정된 test 환자(점수와 무관, seed 기준 양성/음성 각 k명)의 12패널 Grad-CAM 몬타주 + landmark별 평균 CAM 강도(양성 vs 음성).
# split이 seed로 고정이라 재학습해도 같은 환자가 선택되어 모델 간 CAM을 같은 환자로 비교할 수 있다
def plot_gradcam(disease, net, img, x, y, score, pid, k=3):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    picks = [("pos", i) for i in np.sort(rng.choice(pos, k, replace=False))] + [("neg", i) for i in np.sort(rng.choice(neg, k, replace=False))]
    for tag, i in picks:
        cam = gradcam(net, img[i], x[i])
        fig, ax = plt.subplots(3, 4, figsize=(12, 9.4))
        for j, a in enumerate(ax.ravel()):
            a.imshow(img[i, j]), a.imshow(cam[j], cmap="jet", alpha=0.45, vmin=0, vmax=1), a.set_title(LM_NAMES[j], fontsize=9), a.axis("off")
        fig.suptitle(f"{disease} Grad-CAM  PatientID={pid[i]}  true={tag}  score={score[i]:.2f}")
        fig.tight_layout(), fig.savefig(FIG_DIR / f"{disease}_gradcam_{tag}_{pid[i]}.png", dpi=110), plt.close(fig)
    def inside(i):  # 몸 영역 안 픽셀만 평균(몸 크기 차이로 인한 희석 방지)
        mk = body_mask(img[i])
        return (gradcam(net, img[i], x[i]) * mk).sum((1, 2)) / mk.sum((1, 2)).clip(1)
    m = {t: np.mean([inside(i) for i in idx[:40]], 0) for t, idx in (("pos", pos), ("neg", neg))}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(np.arange(12) - 0.2, m["pos"], 0.4, label="positive"), ax.bar(np.arange(12) + 0.2, m["neg"], 0.4, label="negative")
    ax.set_xticks(range(12)), ax.set_xticklabels(LM_NAMES, rotation=45, ha="right"), ax.set(title=f"{disease} mean Grad-CAM intensity inside body by landmark (test, up to 40/class)", ylabel="mean CAM"), ax.legend()
    fig.tight_layout(), fig.savefig(FIG_DIR / f"{disease}_gradcam_by_landmark.png", dpi=150), plt.close(fig)


# loss/AUC 학습곡선, ROC(내부/외부), confusion matrix(valid Youden cutoff)를 질환별 png로 저장
def plot_disease(disease, hist, row, internal, external, thr):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train (epoch mean)"), ax[0].plot(hist["epoch"], hist["valid_loss"], label="valid")
    ax[0].set(title=f"{disease} loss", xlabel="epoch", ylabel="BCE loss"), ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_auc"], label="train"), ax[1].plot(hist["epoch"], hist["valid_auc"], label="valid")
    ax[1].axhline(row["base_valid_auc"], ls=":", c="k", label=f"baseline valid {row['base_valid_auc']:.3f}")
    ax[1].axvline(row["best_epoch"], ls="--", c="gray", label=f"best epoch {row['best_epoch']}")
    ax[1].set(title=f"{disease} AUC", xlabel="epoch", ylabel="AUC"), ax[1].legend()
    fig.tight_layout(), fig.savefig(FIG_DIR / f"{disease}_training_curve.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k, (name, (yy, ss, bb)) in enumerate((("internal (gangnam test)", internal), ("external (sinchon)", external))):
        c = f"C{k}"
        RocCurveDisplay.from_predictions(yy, ss, ax=ax[0], color=c, name=f"{name} CNN AUC={roc_auc_score(yy, ss):.3f}", plot_chance_level=False)
        RocCurveDisplay.from_predictions(yy, bb, ax=ax[0], color=c, ls="--", name=f"{name} baseline AUC={roc_auc_score(yy, bb):.3f}", plot_chance_level=False)
        ConfusionMatrixDisplay(confusion_matrix(yy, (ss >= thr).astype(int), labels=[0, 1]), display_labels=["neg", "pos"]).plot(ax=ax[k + 1], colorbar=False)
        ax[k + 1].set_title(f"{name}\nsens={row[['internal_sens', 'external_sens'][k]]:.2f} spec={row[['internal_spec', 'external_spec'][k]]:.2f}\nAUC CNN {roc_auc_score(yy, ss):.3f} vs baseline {roc_auc_score(yy, bb):.3f} (p={row[['internal_p_vs_baseline', 'external_p_vs_baseline'][k]]:.3f})", fontsize=9)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8), ax[0].set_title(f"{disease} ROC"), ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(), fig.savefig(FIG_DIR / f"{disease}_roc_cm.png", dpi=150), plt.close(fig)


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    df = df[[(IMG_DIR["gangnam"] / f"{p}.png").exists() for p in df["PatientID"]]].reset_index(drop=True)
    df_ext = df_ext[[(IMG_DIR["sinchon"] / f"{p}.png").exists() for p in df_ext["PatientID"]]].reset_index(drop=True)
    img, img_ext = load_panels("gangnam", df["PatientID"].tolist()), load_panels("sinchon", df_ext["PatientID"].tolist())
    x, sc = matrix(df, [])
    x_ext, _ = matrix(df_ext, [], sc)
    res = [run(d, img, x, df, img_ext, x_ext, df_ext) for d in DISEASES]
    if GRADCAM_ONLY:
        return
    save_sheet(pd.DataFrame([r[0] for r in res]), OUT_XLSX, "summary")
    save_sheet(pd.concat([r[1] for r in res]), OUT_XLSX, "history")
    save_sheet(pd.concat([r[2] for r in res]), OUT_XLSX, "predictions")  # DeLong 등 후속 비교용


if __name__ == "__main__":
    main()
