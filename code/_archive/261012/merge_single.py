from __future__ import annotations

# 질환별로 따로 돌린 정상군 vs 단독 질환군 결과(outputs/261012_single/{HTN,DM,CKD})를 한 파일로 합친다 (단독 실행)
import shutil
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
O = Path(__file__).resolve().parents[2] / "outputs"
import os
S, DIS = O / os.environ.get("OUT_ROOT", "261012_single"), ["HTN", "DM", "CKD"]
names = sorted({p.name for d in DIS for p in (S / d).glob("*.xlsx")})
for n in names:
    parts = [pd.read_excel(S / d / n, sheet_name=None) for d in DIS if (S / d / n).exists()]
    with pd.ExcelWriter(S / n) as w:
        for sh in parts[0]:
            pd.concat([p[sh] for p in parts if sh in p], ignore_index=True).to_excel(w, sheet_name=sh, index=False)
    print("merged", n)
for f in ["prognosis_auc.xlsx", "prognosis_auc.png"]:  # 5년 예후 결과는 정의가 달라 그대로 복사
    shutil.copy(O / "261012" / f, S / f)
