"""Central input/output path configuration.

The original notebooks saved figures inconsistently: some wrote to
``../results/...``, others to bare filenames in whatever the notebook's
current working directory happened to be. Every script in this package
saves through :data:`RESULTS_DIR` / :data:`INFERENCE_DIR` instead.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
SHORT_TERM_DIR = DATA_DIR / "short_term"
LONG_TERM_DIR = DATA_DIR / "long_term"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
INFERENCE_DIR = RESULTS_DIR / "inference"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
