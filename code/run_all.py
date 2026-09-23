from __future__ import annotations

# clinic4 파이프라인 전체를 한 번에 돌리는 진입점. 모델 스크립트를 순서대로 실행한 뒤 보고용 pptx를 다시 만든다.
# 병렬 실행 금지 - save_sheet가 read-modify-write라 두 스크립트가 같은 xlsx를 동시에 쓰면 시트가 통째로 날아가고,
# 빌더가 반쯤 쓰인 xlsx를 읽으면 BadZipFile로 죽는다.
#
#   python code/run_all.py            # 전체 실행
#   python code/run_all.py aec        # 이름에 'aec'가 들어간 스크립트만
#   python code/run_all.py --pptx     # 모델 재실행 없이 pptx만 다시 생성
#   python code/run_all.py --no-pptx  # 모델만 실행하고 pptx는 건너뜀

import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CODE_DIR = Path(__file__).resolve().parent

# 실행 순서(baseline이 먼저여야 나머지가 비교 대상으로 삼는 시트가 생김)
MODEL_SCRIPTS = [
    "clinic4_data_distribution.py",
    "clinic4_logistic_regression.py",
    "clinic4_aec_logistic_regression.py",
    "clinic4_aec_vat_logistic.py",
    "clinic4_aec_lama_logistic.py",
    "clinic4_aec_nama_logistic.py",
    "clinic4_aec_bodycomp_logistic.py",
    "clinic4_bodycomp_logistic.py",
    "save_roc_individual.py",
]
REPORT_SCRIPT = "build_pptx_report.py"


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
    if "--pptx" in argv:
        scripts = [REPORT_SCRIPT]
    else:
        keywords = [a for a in argv if not a.startswith("--")]
        scripts = [s for s in MODEL_SCRIPTS if not keywords or any(k in s for k in keywords)]
        if "--no-pptx" not in argv:
            scripts.append(REPORT_SCRIPT)

    started = time.time()
    for script in scripts:
        run(script)
    print(f"\nAll done: {len(scripts)} scripts in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
