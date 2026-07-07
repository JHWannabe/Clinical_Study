from __future__ import annotations

"""
Run the final AEC low-SMI analysis from raw Excel files only.

This is the student-facing entry point.

The student should have only two raw files:

    data/g1090.xlsx
    data/sdata.xlsx

Then run:

    python code/run_from_raw.py

What this script does
---------------------
This file intentionally does not hide the intermediate CSV files.
Instead, it creates them in order, so the student can inspect each stage.

Stage 1. Search and lock interpretable AEC features
    input:
        g1090.xlsx, sdata.xlsx
    output:
        outputs/aec_lock_smoothed_deesc_gate/locked_gate_features.csv

    Meaning:
        The code smooths each patient's 128-point AEC curve, normalizes each
        patient by their own mean AEC level, creates many curve-shape features,
        and selects a small locked feature set using internal data only.

Stage 2. Train region-guided CNN branch probabilities
    input:
        locked AEC features from Stage 1
    output:
        outputs/aec_region_cnn_pattern_gate/direct_vote_probabilities.npz

    Meaning:
        The CNN learns region-specific curve morphology. It does not receive
        the final external result while deciding the model.

Stage 3. Convert CNN branch probabilities into patient-level AEC scores
    input:
        direct_vote_probabilities.npz
    output:
        outputs/aec_direct_vote_auc_boost/direct_vote_auc_boost_scores.csv

    Meaning:
        This is where the final score column vote_only_logit_l1 is created.
        That CSV is not a mysterious hand-made file; it is generated here.

Stage 4. Final phenotype enrichment analysis
    input:
        direct_vote_auc_boost_scores.csv
    output:
        outputs/aec_final_global_quintile_phenotype/

    Meaning:
        The final paper-style result:
        Among clinical high-risk patients, compare AEC-high vs AEC-low.

Important design rule
---------------------
All thresholds and model choices are locked from the internal cohort.
The external cohort is used only after the internal procedure is fixed.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Small utility functions
# ---------------------------------------------------------------------------

def timestamp() -> str:
    """Return a short clock string for progress messages."""
    return time.strftime("%H:%M:%S")


def say(message: str) -> None:
    """Print progress immediately, even during long CNN training."""
    print(f"[{timestamp()}] {message}", flush=True)


def require_file(path: Path, explanation: str) -> None:
    """
    Stop early if an expected input file is absent.

    This is friendlier than letting pandas fail later with a long traceback.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {explanation}: {path}")


def check_python_dependencies() -> None:
    """
    Verify that required Python packages are importable.

    The final method uses PyTorch and scikit-learn. If these are missing,
    the student should install them or use the provided analysis environment.
    """
    required = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "torch",
        "matplotlib",
        "openpyxl",
    ]
    missing: list[str] = []
    versions: list[str] = []
    for name in required:
        try:
            mod = importlib.import_module(name)
            versions.append(f"{name}={getattr(mod, '__version__', 'ok')}")
        except Exception:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(missing)
            + "\nInstall these before running the full pipeline."
        )

    say("Dependency check passed: " + "; ".join(versions))


def run_step(step_name: str, marker: Path, force: bool, action) -> None:
    """
    Run one pipeline stage.

    marker:
        A file that proves the stage already succeeded.

    force:
        If True, rerun even if the marker exists.
    """
    if marker.exists() and not force:
        say(f"SKIP {step_name}: found {marker.name}")
        return

    if marker.exists() and force:
        marker.unlink()

    say(f"START {step_name}")
    t0 = time.time()
    action()
    elapsed = time.time() - t0

    if not marker.exists():
        raise RuntimeError(f"{step_name} finished but expected output is missing: {marker}")

    say(f"DONE  {step_name} ({elapsed / 60.0:.1f} min)")


def reset_shared_random_state(mods: dict) -> None:
    """
    Reset the older helper module's fold generator before each stage.

    Why this matters:
        In the original research workflow, each script was run as a separate
        Python process. That automatically reset module-level random states.

        In this student runner, all stages are called inside one Python
        process. Without this reset, the clinical cross-validation folds can
        change from stage to stage simply because the random generator has
        already been used. That would be a subtle but real reproducibility bug.
    """
    conditional = mods["conditional"]
    if hasattr(conditional, "SEED"):
        conditional.RNG = np.random.default_rng(int(conditional.SEED))


def stage_action(mods: dict, action):
    """
    Wrap a stage so it starts from the same seeded helper state every time.
    """
    def _wrapped():
        reset_shared_random_state(mods)
        return action()

    return _wrapped


# ---------------------------------------------------------------------------
# 2. Path patching
# ---------------------------------------------------------------------------

def import_pipeline_modules(code_dir: Path):
    """
    Import the analysis modules that were copied into this folder.

    We insert code_dir at the front of sys.path so Python uses the student's
    bundled files, not some older copy elsewhere on the computer.
    """
    sys.path.insert(0, str(code_dir))

    import aec_conditional_value as conditional
    import aec_universal_boundary_gate as universal
    import aec128_common_shape_feature as common_shape
    import aec128_mass_feature_combinations as mass_features
    import aec_offset_score as offset_score
    import aec_lock_smoothed_deesc_gate as locked
    import aec_region_constrained_cnn_gate as constrained_cnn
    import aec_region_cnn_teacher_mimic_gate as teacher_mimic
    import aec_region_cnn_direct_vote_gate as direct_vote
    import aec_region_cnn_pattern_gate as pattern_gate
    import aec_direct_vote_auc_boost as auc_boost
    import aec_final_global_quintile_phenotype_pipeline as final_analysis

    return {
        "conditional": conditional,
        "universal": universal,
        "common_shape": common_shape,
        "mass_features": mass_features,
        "offset_score": offset_score,
        "locked": locked,
        "constrained_cnn": constrained_cnn,
        "teacher_mimic": teacher_mimic,
        "direct_vote": direct_vote,
        "pattern_gate": pattern_gate,
        "auc_boost": auc_boost,
        "final": final_analysis,
    }


def patch_module_paths(mods: dict, project_root: Path, data_dir: Path, output_root: Path) -> dict[str, Path]:
    """
    Replace the original research-computer paths with student-local paths.

    Many research scripts started life with hard-coded paths. Rather than
    making the student edit every file manually, this function redirects all
    important module constants at runtime.
    """
    paths = {
        "lock_dir": output_root / "aec_lock_smoothed_deesc_gate",
        "pattern_dir": output_root / "aec_region_cnn_pattern_gate",
        "boost_dir": output_root / "aec_direct_vote_auc_boost",
        "final_dir": output_root / "aec_final_global_quintile_phenotype",
        "conditional_dir": output_root / "aec_conditional_value",
        "universal_dir": output_root / "aec_universal_boundary_gate",
        "mass_dir": output_root / "aec128_mass_feature_combinations",
        "offset_dir": output_root / "aec_offset_score",
        "constrained_dir": output_root / "aec_region_constrained_cnn_gate",
        "teacher_dir": output_root / "aec_region_cnn_teacher_mimic_gate",
        "direct_vote_dir": output_root / "aec_region_cnn_direct_vote_gate",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Modules used by clinical score and older helper functions.
    mods["conditional"].DATA_DIR = data_dir
    mods["conditional"].OUT_DIR = paths["conditional_dir"]
    mods["universal"].DATA_DIR = data_dir
    mods["universal"].OUT_DIR = paths["universal_dir"]

    # Feature-bank helper modules. These are mostly used for functions, but
    # their path constants are redirected for safety.
    mods["mass_features"].BASE_DIR = output_root / "aec128_signal_audit"
    mods["mass_features"].OUT_DIR = paths["mass_dir"]
    mods["offset_score"].DATA_DIR = data_dir
    mods["offset_score"].OUT_DIR = paths["offset_dir"]
    if hasattr(mods["common_shape"], "FILES"):
        mods["common_shape"].FILES = {
            "g1090": data_dir / "g1090.xlsx",
            "sdata": data_dir / "sdata.xlsx",
        }

    # Stage 1: locked feature search.
    mods["locked"].DATA_DIR = data_dir
    mods["locked"].OUT_DIR = paths["lock_dir"]

    # Stage 2 helper CNN modules.
    mods["constrained_cnn"].OUT_DIR = paths["constrained_dir"]
    mods["teacher_mimic"].OUT_DIR = paths["teacher_dir"]
    mods["teacher_mimic"].LOCK_DIR = paths["lock_dir"]
    mods["direct_vote"].OUT_DIR = paths["direct_vote_dir"]

    # Stage 2 main CNN probability script.
    mods["pattern_gate"].DATA_DIR = data_dir
    mods["pattern_gate"].OUT_DIR = paths["pattern_dir"]
    mods["pattern_gate"].PROB_CACHE = paths["pattern_dir"] / "direct_vote_probabilities.npz"

    # Stage 3 score-generation script.
    mods["auc_boost"].DATA_DIR = data_dir
    mods["auc_boost"].OUT_DIR = paths["boost_dir"]
    mods["auc_boost"].PROB_PATH = mods["pattern_gate"].PROB_CACHE

    # Stage 4 final analysis.
    mods["final"].PROJECT_ROOT = project_root
    mods["final"].DATA_DIR = data_dir
    mods["final"].INTERNAL_XLSX = data_dir / "g1090.xlsx"
    mods["final"].EXTERNAL_XLSX = data_dir / "sdata.xlsx"
    mods["final"].OUT_DIR = paths["final_dir"]
    mods["final"].OUT_DIR.mkdir(parents=True, exist_ok=True)
    mods["final"].DIRECT_VOTE_SCORE_CSV = paths["boost_dir"] / "direct_vote_auc_boost_scores.csv"

    return paths


# ---------------------------------------------------------------------------
# 3. Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run AEC low-SMI analysis from raw g1090/sdata Excel files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Folder containing g1090.xlsx and sdata.xlsx. Default: ../data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Folder where all generated CSV/PNG/JSON files will be written. Default: ../outputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun every stage even if cached outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check files, dependencies, and planned output paths. Do not train models.",
    )
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent
    data_dir = args.data_dir.resolve()
    output_root = args.output_dir.resolve()

    say(f"Project root: {project_root}")
    say(f"Data folder : {data_dir}")
    say(f"Output folder: {output_root}")

    require_file(data_dir / "g1090.xlsx", "internal/Gangnam raw Excel file")
    require_file(data_dir / "sdata.xlsx", "external/Sinchon raw Excel file")
    check_python_dependencies()

    mods = import_pipeline_modules(code_dir)
    paths = patch_module_paths(mods, project_root, data_dir, output_root)

    say("Planned stage outputs:")
    say(f"  Stage 1 locked features : {paths['lock_dir'] / 'locked_gate_features.csv'}")
    say(f"  Stage 2 CNN probabilities: {paths['pattern_dir'] / 'direct_vote_probabilities.npz'}")
    say(f"  Stage 3 AEC scores      : {paths['boost_dir'] / 'direct_vote_auc_boost_scores.csv'}")
    say(f"  Stage 4 final table     : {paths['final_dir'] / '01_quintile_vs_quartile_enrichment.csv'}")

    if args.dry_run:
        say("Dry run complete. No model training was executed.")
        return

    run_step(
        "Stage 1/4: internal locked feature search",
        paths["lock_dir"] / "locked_gate_features.csv",
        args.force,
        stage_action(mods, mods["locked"].main),
    )

    run_step(
        "Stage 2/4: region-guided CNN branch probabilities",
        paths["pattern_dir"] / "direct_vote_probabilities.npz",
        args.force,
        stage_action(mods, mods["pattern_gate"].main),
    )

    run_step(
        "Stage 3/4: direct-vote AEC score generation",
        paths["boost_dir"] / "direct_vote_auc_boost_scores.csv",
        args.force,
        stage_action(mods, mods["auc_boost"].main),
    )

    run_step(
        "Stage 4/4: final global quintile phenotype analysis",
        paths["final_dir"] / "01_quintile_vs_quartile_enrichment.csv",
        args.force,
        stage_action(mods, mods["final"].main),
    )

    say("All stages complete.")
    say(f"Main final result: {paths['final_dir'] / '01_quintile_vs_quartile_enrichment.csv'}")
    say(f"Main final figure: {paths['final_dir'] / 'figure_quintile_enrichment.png'}")


if __name__ == "__main__":
    main()
