from pathlib import Path


ROOT = Path(__file__).resolve().parent

IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "default_image.jpg"

MODEL_DIR = ROOT / "weights"
CUSTOM_MODEL = MODEL_DIR / "insects.pt"
