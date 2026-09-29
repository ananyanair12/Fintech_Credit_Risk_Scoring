"""Central configuration: paths, column names, seeds and business assumptions."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
CANDIDATES = ARTIFACTS / "candidates"
DOCS_IMAGES = ROOT / "docs" / "images"

MODEL_PATH = ARTIFACTS / "model.joblib"
METADATA_PATH = ARTIFACTS / "metadata.json"
COMPARISON_PATH = ARTIFACTS / "model_comparison.csv"
SHAP_GLOBAL_PATH = ARTIFACTS / "shap_global.csv"

SEED = 42
TARGET = "default"

# Original Kaggle column names -> short snake_case names used everywhere else
COLUMN_MAP = {
    "SeriousDlqin2yrs": "default",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}

# Business assumption: a missed default (false negative) costs this many times
# more than wrongly declining a good applicant (false positive).
# Override with:  COST_RATIO=15 python -m src.models.evaluate
COST_RATIO = float(os.getenv("COST_RATIO", "10"))

# MLflow (local SQLite backend; view with:  mlflow ui --backend-store-uri sqlite:///mlflow.db)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT = "credit-risk"