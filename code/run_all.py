from __future__ import annotations

# 덱(docs/261002_*.pptx)에 쓰이는 분석·그림을 한 번에 다시 만드는 진입점. 스크립트를 순서대로 실행한다.
# 병렬 실행 금지 - save_sheet가 read-modify-write라 두 스크립트가 같은 xlsx를 동시에 쓰면 시트가 통째로 날아간다.
#
#   python code/run_all.py            # 전체 실행
#   python code/run_all.py aec        # 이름에 'aec'가 들어간 스크립트만

import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CODE_DIR = Path(__file__).resolve().parent

# 실행 순서(앞 단계 산출물을 뒤 단계가 읽는다). 이전 hold-out 방식 스크립트는 _archive/에 보관
MODEL_SCRIPTS = [
    "clinic4_data_distribution.py",       # Table 1
    "clinic4_5fold_unbalanced.py",        # Table 3 성능·DeLong, 예측값 저장(predictions_5fold_unbalanced.xlsx)
    "clinic4_calibration_plot.py",        # Calibration plot
    "clinic4_delta_auc_plot.py",          # Figure 1 (ΔAUC)
    "clinic4_fpca_recovered_5fold.py",    # Figure 2 (AEC 위치별 기여도)
    "seminar_figures.py",                 # AEC 실제 데이터 그림, FPCA 개념
    "seminar_figures2.py",                # 연구 설계도, elbow, FPCA 체성분 역변환, Odds Ratio forest
]


# 스크립트 하나를 실행하고 소요 시간을 출력. 실패하면 그 자리에서 멈춘다(뒤 결과가 옛 시트와 섞이지 않게)
def run(script: str) -> None:
    print(f"\n{'=' * 70}\n>>> {script}\n{'=' * 70}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(CODE_DIR / script)], cwd=CODE_DIR)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(f"[FAIL] {script} (exit {result.returncode}, {elapsed:.0f}s)")
    print(f"[OK] {script} ({elapsed:.0f}s)", flush=True)


def main(argv: list[str]) -> None:
    scripts = [s for s in MODEL_SCRIPTS if not argv or any(k in s for k in argv)]

    started = time.time()
    for script in scripts:
        run(script)
    print(f"\nAll done: {len(scripts)} scripts in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
