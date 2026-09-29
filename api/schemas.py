"""Pydantic request/response models (input validation for /predict)."""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApplicantFeatures(BaseModel):
    revolving_utilization: float = Field(
        ..., ge=0, description="Balance on cards/credit lines divided by total credit limits"
    )
    age: int = Field(..., ge=18, le=120, description="Applicant age in years")
    past_due_30_59: int = Field(0, ge=0, le=98, description="Times 30-59 days past due")
    debt_ratio: float = Field(..., ge=0, description="Monthly debt payments / monthly income. If monthly_income is null, " 
                                                     "enter the monthly debt AMOUNT here instead (mirrors the source dataset).",)
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income (null if unknown)")
    open_credit_lines: int = Field(..., ge=0, le=100, description="Open loans and credit lines")
    past_due_90: int = Field(0, ge=0, le=98, description="Times 90+ days late")
    real_estate_loans: int = Field(0, ge=0, le=100, description="Mortgage / real-estate loans")
    past_due_60_89: int = Field(0, ge=0, le=98, description="Times 60-89 days past due")
    dependents: Optional[int] = Field(None, ge=0, le=30, description="Number of dependents (null if unknown)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "revolving_utilization": 0.35,
                "age": 42,
                "past_due_30_59": 0,
                "debt_ratio": 0.42,
                "monthly_income": 5400,
                "open_credit_lines": 8,
                "past_due_90": 0,
                "real_estate_loans": 1,
                "past_due_60_89": 0,
                "dependents": 2,
            }
        }
    )


class Factor(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str  # "increases_risk" | "decreases_risk"


class ShapExplanation(BaseModel):
    base_value: float
    top_factors: List[Factor]
    note: str = "SHAP values are in log-odds units; positive values increase predicted risk."


class PredictResponse(BaseModel):
    risk_score: float
    decision: str  # "approve" | "decline"
    threshold: float
    shap_explanation: ShapExplanation
    model_version: str
    latency_ms: float


class ModelInfo(BaseModel):
    model_name: str
    model_kind: str
    threshold: float
    cost_ratio: float
    trained_at: str
    val_metrics: dict
    test_metrics: dict