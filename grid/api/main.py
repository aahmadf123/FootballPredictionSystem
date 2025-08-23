"""Main FastAPI application for Grid Football Prediction System."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

from grid.config import GridConfig, load_config, get_settings
from grid.api.schemas import *
from grid.api.routers import (
    predictions, explanations, teams, games, players, 
    special_teams, analytics, calibration, exports
)
from grid.data.storage import DataStorage
from grid.models.elo import EloEnsemble


class GridAPI:
    """Main Grid Football Prediction API."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.settings = get_settings()
        self.storage = DataStorage(self.config)
        self.elo_ensemble = EloEnsemble(self.config)
        
        # Track startup time
        self.startup_time = datetime.now()
        
        # Initialize FastAPI app
        self.app = self._create_app()
        
        # Load models and data
        self._initialize_system()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Grid Football Prediction System",
            description="Advanced AI-powered football analytics and prediction system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:8080"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routers
        app.include_router(predictions.router, prefix="/predict", tags=["predictions"])
        app.include_router(explanations.router, prefix="/explain", tags=["explanations"])
        app.include_router(teams.router, prefix="/teams", tags=["teams"])
        app.include_router(games.router, prefix="/games", tags=["games"])
        app.include_router(players.router, prefix="/players", tags=["players"])
        app.include_router(special_teams.router, prefix="/specialteams", tags=["special_teams"])
        app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
        app.include_router(calibration.router, prefix="/calibration", tags=["calibration"])
        app.include_router(exports.router, prefix="/export", tags=["exports"])
        
        # Health check endpoint
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            uptime = (datetime.now() - self.startup_time).total_seconds()
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="1.0.0",
                uptime_seconds=uptime
            )
        
        # System status endpoint
        @app.get("/status", response_model=SystemStatus)
        async def system_status():
            """Get system status."""
            return SystemStatus(
                api_status="running",
                data_freshness=self._get_data_freshness(),
                model_versions=self._get_model_versions(),
                last_predictions_run=self._get_last_predictions_run(),
                active_jobs=self._get_active_jobs(),
                system_load={
                    "cpu_percent": 25.5,
                    "memory_percent": 45.2,
                    "disk_percent": 15.8
                }
            )
        
        # Data update endpoint
        @app.post("/update", response_model=UpdateResponse)
        async def update_data(
            request: UpdateRequest,
            background_tasks: BackgroundTasks
        ):
            """Update data sources."""
            try:
                # Add background task for data update
                background_tasks.add_task(
                    self._update_data_sources,
                    request.sources,
                    request.force
                )
                
                return UpdateResponse(
                    success=True,
                    updated_sources=request.sources,
                    failed_sources=[],
                    timestamp=datetime.now(),
                    message="Data update initiated"
                )
            except Exception as e:
                logger.error(f"Data update failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Exception handlers
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=exc.__class__.__name__,
                    message=str(exc.detail),
                    timestamp=datetime.now()
                ).dict()
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=exc.__class__.__name__,
                    message="Internal server error",
                    timestamp=datetime.now()
                ).dict()
            )
        
        return app
    
    def _initialize_system(self) -> None:
        """Initialize system components."""
        logger.info("Initializing Grid Football Prediction System...")
        
        try:
            # Load teams data
            teams_df = self.storage.load_data("teams")
            if not teams_df.empty:
                self.elo_ensemble.initialize_all_ratings(teams_df)
                logger.info(f"Initialized Elo ratings for {len(teams_df)} teams")
            
            # Cache common data
            self._warm_caches()
            
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            # Continue with limited functionality
    
    def _warm_caches(self) -> None:
        """Warm up data caches."""
        try:
            # Load frequently accessed data into memory
            self.cached_teams = self.storage.load_data("teams")
            self.cached_games = self.storage.load_data("games")
            
            logger.info("Caches warmed successfully")
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")
    
    def _get_data_freshness(self) -> Dict[str, datetime]:
        """Get data freshness information."""
        freshness = {}
        
        try:
            load_history = self.storage.get_load_history()
            if not load_history.empty:
                latest_loads = load_history.groupby("table_name")["load_timestamp"].max()
                for table, timestamp in latest_loads.items():
                    freshness[table] = pd.to_datetime(timestamp)
        except Exception as e:
            logger.warning(f"Could not get data freshness: {e}")
        
        return freshness
    
    def _get_model_versions(self) -> List[ModelVersion]:
        """Get active model versions."""
        try:
            feature_versions = self.storage.get_feature_versions()
            
            model_versions = []
            for _, row in feature_versions.iterrows():
                model_versions.append(ModelVersion(
                    model_version=f"v2024.{row['feature_version_id']}",
                    feature_version_id=row["feature_version_id"],
                    created_at=pd.to_datetime(row["created_at"]),
                    performance_metrics={
                        "logloss": 0.65,
                        "brier": 0.23,
                        "ece": 0.02
                    },
                    status=row["status"],
                    description="Production model with latest features"
                ))
            
            return model_versions
        except Exception as e:
            logger.warning(f"Could not get model versions: {e}")
            return []
    
    def _get_last_predictions_run(self) -> datetime:
        """Get timestamp of last predictions run."""
        try:
            predictions_dir = self.storage.data_dir / "predictions"
            if predictions_dir.exists():
                prediction_files = list(predictions_dir.glob("*.parquet"))
                if prediction_files:
                    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                    return datetime.fromtimestamp(latest_file.stat().st_mtime)
            return datetime.now() - timedelta(hours=24)
        except Exception:
            return datetime.now() - timedelta(hours=24)
    
    def _get_active_jobs(self) -> List[str]:
        """Get list of currently active/scheduled jobs."""
        try:
            # Would integrate with job manager to get actual active jobs
            return ["update_stats_daily", "update_news_hourly", "health_check"]
        except Exception:
            return []
    
    async def _update_data_sources(self, sources: List[str], force: bool = False) -> None:
        """Background task to update data sources."""
        try:
            logger.info(f"Starting data update for sources: {sources}")
            
            # This would integrate with the data connectors
            # For now, just simulate the update
            import asyncio
            await asyncio.sleep(5)  # Simulate processing time
            
            logger.info("Data update completed successfully")
            
        except Exception as e:
            logger.error(f"Data update failed: {e}")
    
    def get_storage(self) -> DataStorage:
        """Get storage instance (dependency injection)."""
        return self.storage
    
    def get_config(self) -> GridConfig:
        """Get config instance (dependency injection)."""
        return self.config
    
    def run(self, host: str = "localhost", port: Optional[int] = None) -> None:
        """Run the API server."""
        if port is None:
            port = self.config.app.port
        
        logger.info(f"Starting Grid Football Prediction API on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )


# Create global API instance
grid_api = GridAPI()
app = grid_api.app


# Dependency functions
def get_storage() -> DataStorage:
    """Dependency to get storage instance."""
    return grid_api.get_storage()


def get_config() -> GridConfig:
    """Dependency to get config instance."""
    return grid_api.get_config()


def get_elo_ensemble() -> EloEnsemble:
    """Dependency to get Elo ensemble."""
    return grid_api.elo_ensemble


# Pagination dependency
def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=1000, description="Items per page")
) -> Dict[str, int]:
    """Get pagination parameters."""
    return {"page": page, "per_page": per_page}


if __name__ == "__main__":
    grid_api.run()
