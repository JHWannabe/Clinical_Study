from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark_report 몬타주 이미지(12 landmark 단면 + 분할 overlay)를 쓰는 모델.
# 이미지 -> 12패널 crop -> ImageNet pretrained ResNet18(frozen) 512-d 특징 -> 패널 평균(또는 L3 패널) -> gangnam fit PCA(N_PC) ->
# baseline(clinic4) 뒤에 붙여 LogisticRegression. 평가 프로토콜은 clinic4_landmark_vat_auc와 동일(gangnam 7/1/2 hold-out + sinchon 외부).
# fine-tuning은 하지 않는다(환자 수백 명이라 과적합 위험; 필요하면 backbone을 학습 가능으로 바꿔 확장).
# 주의: overlay가 분할 결과(VAT/SAT/근육 색)를 포함하므로 이미지 특징에는 체성분 정보가 이미 들어 있다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch/sklearn OpenMP 중복 로드 오류 회피
import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA

from clinic4_landmark_vat_auc import delong_vs, load_cohort, matrix, run_disease_holdout
from clinic4_logistic_regression import DATA_DIR, DISEASES, SEED, f3, save_sheet

sys.stdout.reconfigure(encoding="utf-8")

IMG_ROOT = _p.Path(r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data")
IMG_DIR = {"gangnam": IMG_ROOT / "강남/landmark/landmark_report", "sinchon": IMG_ROOT / "신촌/landmark/landmark_report"}
CACHE = DATA_DIR / "landmark_report_resnet18_{cohort}.npz"
OUT_XLSX = "landmark/landmark_image_auc.xlsx"
XS, YS, S = [31, 470, 910, 1349], [147, 681, 1215], 369  # 1760x1716 몬타주의 4x3 패널 좌상단/한 변(제목·수치 텍스트 제외)
L3_PANEL = 6  # L3_center (liver_dome, T10, T11, T12, L1, L2, L3, ...)
N_PC = 16


# 몬타주 한 장 -> (12, 512) 패널별 ResNet18 특징. 같은 파일 재실행은 npz 캐시 사용
def extract(cohort: str, ids: list) -> np.ndarray:
    path = _p.Path(str(CACHE).format(cohort=cohort))
    cache = {}
    if path.exists():
        z = np.load(path)
        cache = dict(zip(z["ids"].astype(str), z["feat"]))
    todo = [p for p in ids if str(p) not in cache]
    if todo:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval().to(dev)
        mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        for n, p in enumerate(todo):
            f = IMG_DIR[cohort] / f"{p}.png"
            im = Image.open(f).convert("RGB")
            panels = [im.crop((x, y, x + S, y + S)).resize((224, 224)) for y in YS for x in XS]
            t = (torch.from_numpy(np.stack(panels)).permute(0, 3, 1, 2).float() / 255 - mean) / std
            with torch.no_grad():
                cache[str(p)] = net(t.to(dev)).cpu().numpy()
            if n % 200 == 0:
                print(f"[{cohort}] 특징 추출 {n}/{len(todo)}")
        np.savez(path, ids=np.array(list(cache)), feat=np.stack(list(cache.values())))
    return np.stack([cache[str(p)] for p in ids])


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    # 이미지 없는 환자 제외
    df = df[[(IMG_DIR["gangnam"] / f"{p}.png").exists() for p in df["PatientID"]]].reset_index(drop=True)
    df_ext = df_ext[[(IMG_DIR["sinchon"] / f"{p}.png").exists() for p in df_ext["PatientID"]]].reset_index(drop=True)
    f, f_ext = extract("gangnam", df["PatientID"].tolist()), extract("sinchon", df_ext["PatientID"].tolist())  # (N,12,512)

    variants = {"img mean(12 panels)": lambda a: a.mean(1), "img L3 panel": lambda a: a[:, L3_PANEL], "img concat(12 panels)": lambda a: a.reshape(len(a), -1)}
    sets = {"baseline": []}
    for name, fn in variants.items():  # PCA는 비지도라 gangnam 전체로 fit, sinchon은 frozen transform
        pca = PCA(N_PC, random_state=SEED).fit(fn(f))
        for d, a in ((df, f), (df_ext, f_ext)):
            sc = pca.transform(fn(a))
            for i in range(N_PC):
                d[f"{name}_pc{i + 1}"] = sc[:, i]
        sets[name] = [f"{name}_pc{i + 1}" for i in range(N_PC)]

    rows, preds = [], []
    for name, cols in sets.items():
        x, sc = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, sc)
        for d in DISEASES:
            r, p = run_disease_holdout(d, name, x, df, x_ext, df_ext)
            rows.append(r), preds.append(p)
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    dl = delong_vs(preds, "baseline")
    for cohort, tag in (("gangnam", "internal"), ("sinchon", "external")):  # baseline 대비 DeLong p(raw/FDR)를 summary에도 붙임
        g = dl[dl["cohort"] == cohort].set_index(["disease", "model"])[["p_value", "p_fdr"]].add_prefix(f"{tag}_")
        summary = summary.join(g, on=["disease", "model"])
    save_sheet(summary, OUT_XLSX, "summary")  # 순차 저장(병렬 실행 금지: clinic4_xlsx_write_race)
    save_sheet(dl, OUT_XLSX, "delong_vs_baseline")
    fmt = lambda a, p: "-" if pd.isna(p) else f"{a:.3f} (p={p:.3f})"  # baseline은 p 없음
    for tag in ("internal", "external"):
        s = summary.assign(v=[fmt(a, p) for a, p in zip(summary[f"{tag}_auc"], summary[f"{tag}_p_value"])])
        s["v"] = np.where(s[f"{tag}_p_value"].isna(), s[f"{tag}_auc"].map(f3), s["v"])
        print(f"\n[{tag}] AUC (DeLong p vs baseline)\n", s.pivot(index="model", columns="disease", values="v"))


if __name__ == "__main__":
    main()
