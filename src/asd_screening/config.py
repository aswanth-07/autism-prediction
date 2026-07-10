"""Project-wide constants and canonical data schema."""

from pathlib import Path

PROJECT_VERSION = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_AGE = 18
MAX_AGE = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "Data" / "Raw Data" / "Raw Data.csv"
MODEL_PATH = PROJECT_ROOT / "Models" / "production_pipeline.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "class_asd"
AQ_FEATURES = tuple(f"A{i}_Score" for i in range(1, 11))
NUMERIC_FEATURES = (*AQ_FEATURES, "age")
CATEGORICAL_FEATURES = (
    "gender",
    "ethnicity",
    "jaundice",
    "family_asd_history",
    "country_of_residence",
    "used_app_before",
    "relation",
)
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

RAW_TO_CANONICAL = {
    "austim": "family_asd_history",
    "contry_of_res": "country_of_residence",
    "Class/ASD": TARGET,
}
