"""Exports API router."""

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from grid.api.schemas import ExportFormat, ExportResponse
from grid.data.storage import DataStorage

router = APIRouter()

def get_storage() -> DataStorage:
    from grid.api.main import grid_api
    return grid_api.get_storage()

@router.post("/", response_model=ExportResponse)
async def export_data(
    export_format: ExportFormat,
    storage: DataStorage = Depends(get_storage)
):
    """Export data in specified format."""
    try:
        # Generate export (placeholder)
        export_url = f"/downloads/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.format}"
        
        return ExportResponse(
            download_url=export_url,
            expires_at=datetime.now() + timedelta(hours=24),
            format=export_format.format,
            size_bytes=1024000,
            record_count=5000
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
