from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
PARAMS_DIR = DATA_DIR / "params"

DIFFUSION_MODEL_RESULTS = PROJ_ROOT / "diffusion_model_results"
