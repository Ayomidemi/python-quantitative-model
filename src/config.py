from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

BASE_CURRENCY = "NGN"
FX_PAIRS = {
    ("USD", "NGN"): "USDNGN=X",
    ("NGN", "USD"): "NGNUSD=X",
}
