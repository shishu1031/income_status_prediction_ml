from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
RANDOM_STATE = 42
TARGET = "income_status"
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
