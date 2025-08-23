"""Calibration API router."""

from fastapi import APIRouter, HTTPException, Depends
from grid.api.schemas import CalibrationResponse, CalibrationMetrics, CalibrationBucket
from grid.data.storage import DataStorage

router = APIRouter()

def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

@router.get("/weekly", response_model=CalibrationResponse)
async def get_weekly_calibration(
    storage: DataStorage = Depends(get_storage)
):
    """Get weekly calibration metrics."""
    # Placeholder implementation
    current_week = CalibrationMetrics(
        week=10,
        season=2024,
        buckets=[
            CalibrationBucket(
                bucket=0,
                predicted_prob=0.05,
                actual_rate=0.08,
                count=25,
                brier_contribution=0.001
            )
        ],
        overall_brier=0.235,
        overall_logloss=0.652,
        ece=0.028,
        accuracy=0.612,
        coverage_80=0.82,
        coverage_90=0.91
    )
    
    return CalibrationResponse(
        current_week=current_week,
        historical_trend=[current_week],
        model_health="good",
        recommendations=["Continue monitoring", "Consider ensemble reweighting"]
    )
