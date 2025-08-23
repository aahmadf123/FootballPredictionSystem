"""Automated job scheduling and execution for data updates and predictions."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from loguru import logger

from grid.config import GridConfig, get_settings
from grid.data.storage import DataStorage
from grid.data.connectors import CFBDConnector, NFLConnector, NewsConnector, WeatherConnector
from grid.features.situational import SituationalFeatures
from grid.features.team import TeamFeatureBuilder
from grid.features.player import PlayerFeatureBuilder
from grid.features.special_teams import SpecialTeamsFeatureBuilder
from grid.models.ensemble import EnsembleFramework
from grid.uncertainty.conformal import create_football_conformal_framework


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class JobResult:
    """Result of job execution."""
    job_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    message: Optional[str] = None
    error: Optional[str] = None
    records_processed: int = 0
    files_updated: List[str] = None
    
    def __post_init__(self):
        if self.files_updated is None:
            self.files_updated = []


class CircuitBreaker:
    """Circuit breaker for job failure handling."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable) -> Callable:
        """Wrap function with circuit breaker."""
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if (datetime.now() - self.last_failure_time).seconds > self.timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e
        
        return wrapper


class RetryManager:
    """Retry logic for failed jobs."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                
                wait_time = (self.backoff_factor ** attempt) * 60  # Wait in seconds
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)


class JobManager:
    """Central job manager for automated tasks."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.settings = get_settings()
        self.storage = DataStorage(config)
        
        # Job tracking
        self.job_results: Dict[str, JobResult] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        
        # Reliability components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        
        # Feature builders
        self.situational_features = SituationalFeatures(config)
        self.team_features = TeamFeatureBuilder(config)
        self.player_features = PlayerFeatureBuilder(config)
        self.st_features = SpecialTeamsFeatureBuilder(config)
        
        # Model framework
        self.ensemble_framework = EnsembleFramework(config)
        self.conformal_framework = create_football_conformal_framework()
        
        # Scheduler setup
        self.scheduler = AsyncIOScheduler(
            executors={'default': AsyncIOExecutor()},
            jobstores={'default': MemoryJobStore()},
            timezone='UTC'
        )
        
        self._setup_jobs()
    
    def _setup_jobs(self) -> None:
        """Setup scheduled jobs based on configuration."""
        intervals = self.config.data.update_intervals
        
        # Data update jobs
        self.scheduler.add_job(
            self.update_stats_job,
            CronTrigger.from_crontab(intervals.stats),
            id="update_stats",
            name="Update game statistics",
            replace_existing=True
        )
        
        self.scheduler.add_job(
            self.update_news_job,
            CronTrigger.from_crontab(intervals.news),
            id="update_news",
            name="Update news feeds",
            replace_existing=True
        )
        
        self.scheduler.add_job(
            self.update_weather_job,
            CronTrigger.from_crontab(intervals.weather),
            id="update_weather",
            name="Update weather data",
            replace_existing=True
        )
        
        # Feature engineering jobs
        self.scheduler.add_job(
            self.rebuild_features_job,
            CronTrigger(hour=2, minute=0),  # Daily at 2 AM
            id="rebuild_features",
            name="Rebuild feature store",
            replace_existing=True
        )
        
        # Prediction jobs
        self.scheduler.add_job(
            self.generate_predictions_job,
            CronTrigger(day_of_week=1, hour=6, minute=0),  # Tuesday 6 AM
            id="generate_predictions",
            name="Generate weekly predictions",
            replace_existing=True
        )
        
        # Model calibration jobs
        self.scheduler.add_job(
            self.recalibrate_models_job,
            CronTrigger(day_of_week=1, hour=6, minute=15),  # Tuesday 6:15 AM
            id="recalibrate_models",
            name="Recalibrate model ensemble",
            replace_existing=True
        )
        
        # Health check job
        self.scheduler.add_job(
            self.health_check_job,
            CronTrigger(minute="*/15"),  # Every 15 minutes
            id="health_check",
            name="System health monitoring",
            replace_existing=True
        )
    
    def start(self) -> None:
        """Start the job scheduler."""
        self.scheduler.start()
        logger.info("Job scheduler started")
    
    def stop(self) -> None:
        """Stop the job scheduler."""
        self.scheduler.shutdown()
        logger.info("Job scheduler stopped")
    
    def get_circuit_breaker(self, job_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for job."""
        if job_name not in self.circuit_breakers:
            self.circuit_breakers[job_name] = CircuitBreaker()
        return self.circuit_breakers[job_name]
    
    async def execute_job(self, job_func: Callable, job_id: str, *args, **kwargs) -> JobResult:
        """Execute a job with error handling and tracking."""
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.job_results[job_id] = job_result
        
        try:
            # Get circuit breaker for this job type
            circuit_breaker = self.get_circuit_breaker(job_id.split('_')[0])
            
            # Execute with circuit breaker and retry
            wrapped_func = circuit_breaker.call(job_func)
            result = await self.retry_manager.execute_with_retry(wrapped_func, *args, **kwargs)
            
            # Update job result
            job_result.status = JobStatus.COMPLETED
            job_result.end_time = datetime.now()
            job_result.message = "Job completed successfully"
            
            if isinstance(result, dict):
                job_result.records_processed = result.get('records_processed', 0)
                job_result.files_updated = result.get('files_updated', [])
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            job_result.status = JobStatus.FAILED
            job_result.end_time = datetime.now()
            job_result.error = str(e)
            
            logger.error(f"Job {job_id} failed: {e}")
        
        return job_result
    
    async def update_stats_job(self) -> Dict[str, Any]:
        """Update game statistics from external sources."""
        logger.info("Starting stats update job")
        
        total_records = 0
        files_updated = []
        
        try:
            # NFL data update
            async with NFLConnector(self.settings) as nfl_connector:
                current_season = datetime.now().year
                current_week = min(18, max(1, (datetime.now() - datetime(current_season, 9, 1)).days // 7))
                
                # Update teams
                teams_df = await nfl_connector.get_teams(current_season)
                if not teams_df.empty:
                    self.storage.upsert_data(teams_df, "teams", ["team_id", "season"])
                    total_records += len(teams_df)
                    files_updated.append("teams")
                
                # Update games for current week
                games_df = await nfl_connector.get_games(current_season, current_week)
                if not games_df.empty:
                    self.storage.upsert_data(games_df, "games", ["game_id"])
                    total_records += len(games_df)
                    files_updated.append("games")
            
            # CFBD data update (if API key available)
            if self.settings.cfbd_api_key:
                async with CFBDConnector(self.settings) as cfbd_connector:
                    # Update NCAA teams
                    ncaa_teams_df = await cfbd_connector.get_teams(current_season)
                    if not ncaa_teams_df.empty:
                        self.storage.upsert_data(ncaa_teams_df, "teams", ["team_id", "season"])
                        total_records += len(ncaa_teams_df)
                    
                    # Update NCAA games
                    ncaa_games_df = await cfbd_connector.get_games(current_season, current_week)
                    if not ncaa_games_df.empty:
                        self.storage.upsert_data(ncaa_games_df, "games", ["game_id"])
                        total_records += len(ncaa_games_df)
            
            return {
                'records_processed': total_records,
                'files_updated': files_updated
            }
            
        except Exception as e:
            logger.error(f"Stats update job failed: {e}")
            raise
    
    async def update_news_job(self) -> Dict[str, Any]:
        """Update news feeds."""
        logger.info("Starting news update job")
        
        try:
            async with NewsConnector(self.settings) as news_connector:
                news_df = await news_connector.get_all_news()
                
                if not news_df.empty:
                    self.storage.upsert_data(news_df, "news", ["news_id"])
                    
                    return {
                        'records_processed': len(news_df),
                        'files_updated': ['news']
                    }
                else:
                    return {
                        'records_processed': 0,
                        'files_updated': []
                    }
        
        except Exception as e:
            logger.error(f"News update job failed: {e}")
            raise
    
    async def update_weather_job(self) -> Dict[str, Any]:
        """Update weather data for upcoming games."""
        logger.info("Starting weather update job")
        
        try:
            # Get upcoming games
            games_df = self.storage.load_data("games")
            upcoming_games = games_df[
                (games_df["kickoff_utc"] > datetime.now()) &
                (games_df["kickoff_utc"] < datetime.now() + timedelta(days=7))
            ]
            
            weather_records = []
            
            if not upcoming_games.empty:
                weather_connector = WeatherConnector(self.settings)
                
                for _, game in upcoming_games.iterrows():
                    try:
                        # Get weather for game location and date
                        weather_data = await weather_connector.fetch_raw(
                            location=game.get("venue_id", "unknown"),
                            date=game["kickoff_utc"].strftime("%Y-%m-%d")
                        )
                        
                        if weather_connector.validate(weather_data):
                            weather_df = weather_connector.normalize(weather_data)
                            weather_df["game_id"] = game["game_id"]
                            weather_records.append(weather_df)
                    
                    except Exception as e:
                        logger.warning(f"Weather update failed for game {game['game_id']}: {e}")
            
            if weather_records:
                all_weather = pd.concat(weather_records, ignore_index=True)
                self.storage.upsert_data(all_weather, "weather", ["game_id"])
                
                return {
                    'records_processed': len(all_weather),
                    'files_updated': ['weather']
                }
            else:
                return {
                    'records_processed': 0,
                    'files_updated': []
                }
        
        except Exception as e:
            logger.error(f"Weather update job failed: {e}")
            raise
    
    async def rebuild_features_job(self) -> Dict[str, Any]:
        """Rebuild feature store with latest data."""
        logger.info("Starting feature rebuild job")
        
        try:
            # Load base data
            games_df = self.storage.load_data("games")
            teams_df = self.storage.load_data("teams")
            pbp_df = self.storage.load_data("pbp")
            players_df = self.storage.load_data("players")
            st_df = self.storage.load_data("special_teams")
            
            total_features = 0
            feature_version_id = f"fv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Build situational features
            if not pbp_df.empty and not games_df.empty:
                situations_df = self.situational_features.process_game_situations(pbp_df, games_df)
                if not situations_df.empty:
                    self.storage.save_gold_data(situations_df, "situations", feature_version_id)
                    total_features += len(situations_df)
            
            # Build team features
            if not pbp_df.empty and not games_df.empty:
                team_features_df = self.team_features.build_team_features(pbp_df, games_df, situations_df)
                if not team_features_df.empty:
                    self.storage.save_gold_data(team_features_df, "team_features", feature_version_id)
                    total_features += len(team_features_df)
            
            # Build player features
            if not pbp_df.empty and not players_df.empty and not games_df.empty:
                player_features_df = self.player_features.build_player_features(pbp_df, players_df, games_df)
                if not player_features_df.empty:
                    self.storage.save_gold_data(player_features_df, "player_features", feature_version_id)
                    total_features += len(player_features_df)
            
            # Build special teams features
            if not st_df.empty:
                st_features_df = self.st_features.build_special_teams_features(st_df)
                if not st_features_df.empty:
                    self.storage.save_gold_data(st_features_df, "st_features", feature_version_id)
                    total_features += len(st_features_df)
            
            # Create feature version record
            self.storage.create_feature_version(
                feature_version_id,
                config_hash="automated_rebuild",
                dependencies=["games", "teams", "pbp", "players", "special_teams"]
            )
            
            return {
                'records_processed': total_features,
                'files_updated': [f'feature_store/{feature_version_id}']
            }
        
        except Exception as e:
            logger.error(f"Feature rebuild job failed: {e}")
            raise
    
    async def generate_predictions_job(self) -> Dict[str, Any]:
        """Generate weekly predictions."""
        logger.info("Starting predictions generation job")
        
        try:
            # Load latest features
            feature_versions = self.storage.get_feature_versions()
            if feature_versions.empty:
                raise ValueError("No feature versions available")
            
            latest_version = feature_versions.iloc[0]["feature_version_id"]
            
            # Load data for predictions
            games_df = self.storage.load_data("games")
            teams_df = self.storage.load_data("teams")
            
            # Filter for upcoming games
            upcoming_games = games_df[
                (games_df["kickoff_utc"] > datetime.now()) &
                (games_df["status"] == "scheduled")
            ]
            
            if upcoming_games.empty:
                return {
                    'records_processed': 0,
                    'files_updated': []
                }
            
            # Load features
            team_features = self.storage.load_gold_data("team_features", latest_version)
            
            data_dict = {
                "games": upcoming_games,
                "teams": teams_df,
                "features": team_features
            }
            
            # Generate ensemble predictions
            predictions = self.ensemble_framework.predict(data_dict)
            
            # Apply conformal prediction
            conformal_intervals = self.conformal_framework.predict_intervals({
                "win_prob": predictions["ensemble_calibrated"]
            })
            
            # Save predictions
            predictions_df = pd.DataFrame({
                "game_id": upcoming_games["game_id"],
                "predicted_at": datetime.now(),
                "home_win_prob": predictions["ensemble_calibrated"],
                "model_version": latest_version,
                "confidence_lower": conformal_intervals["win_prob"][0],
                "confidence_upper": conformal_intervals["win_prob"][1]
            })
            
            predictions_path = self.storage.data_dir / "predictions" / f"weekly_{datetime.now().strftime('%Y%m%d')}.parquet"
            predictions_path.parent.mkdir(exist_ok=True)
            predictions_df.to_parquet(predictions_path, index=False)
            
            return {
                'records_processed': len(predictions_df),
                'files_updated': [str(predictions_path)]
            }
        
        except Exception as e:
            logger.error(f"Predictions generation job failed: {e}")
            raise
    
    async def recalibrate_models_job(self) -> Dict[str, Any]:
        """Recalibrate model ensemble based on recent performance."""
        logger.info("Starting model recalibration job")
        
        try:
            # Load recent games with results
            games_df = self.storage.load_data("games")
            recent_completed = games_df[
                (games_df["kickoff_utc"] > datetime.now() - timedelta(days=14)) &
                (games_df["home_score"].notna()) &
                (games_df["away_score"].notna())
            ]
            
            if len(recent_completed) < 10:
                logger.warning("Insufficient recent games for recalibration")
                return {
                    'records_processed': 0,
                    'files_updated': []
                }
            
            # Get corresponding predictions
            predictions_files = list((self.storage.data_dir / "predictions").glob("*.parquet"))
            recent_predictions = []
            
            for pred_file in predictions_files:
                pred_df = pd.read_parquet(pred_file)
                pred_df = pred_df[pred_df["game_id"].isin(recent_completed["game_id"])]
                recent_predictions.append(pred_df)
            
            if not recent_predictions:
                logger.warning("No recent predictions found for recalibration")
                return {
                    'records_processed': 0,
                    'files_updated': []
                }
            
            all_predictions = pd.concat(recent_predictions, ignore_index=True)
            
            # Merge with actual results
            evaluation_data = all_predictions.merge(
                recent_completed[["game_id", "home_score", "away_score"]],
                on="game_id"
            )
            
            if evaluation_data.empty:
                return {
                    'records_processed': 0,
                    'files_updated': []
                }
            
            # Calculate actual outcomes
            evaluation_data["home_win"] = (evaluation_data["home_score"] > evaluation_data["away_score"]).astype(int)
            
            # Recalibrate conformal predictor
            self.conformal_framework.fit(
                {"win_prob": evaluation_data["home_win_prob"].values},
                {"win_prob": evaluation_data["home_win"].values}
            )
            
            # Evaluate current performance
            from sklearn.metrics import log_loss, brier_score_loss
            
            logloss = log_loss(evaluation_data["home_win"], evaluation_data["home_win_prob"])
            brier = brier_score_loss(evaluation_data["home_win"], evaluation_data["home_win_prob"])
            
            # Save recalibration results
            calibration_result = {
                "recalibrated_at": datetime.now(),
                "games_evaluated": len(evaluation_data),
                "logloss": logloss,
                "brier_score": brier,
                "model_version": "ensemble_recalibrated"
            }
            
            cal_path = self.storage.data_dir / "calibration" / f"recalibration_{datetime.now().strftime('%Y%m%d')}.json"
            cal_path.parent.mkdir(exist_ok=True)
            
            import json
            with open(cal_path, "w") as f:
                json.dump(calibration_result, f, indent=2, default=str)
            
            logger.info(f"Model recalibrated. LogLoss: {logloss:.4f}, Brier: {brier:.4f}")
            
            return {
                'records_processed': len(evaluation_data),
                'files_updated': [str(cal_path)]
            }
        
        except Exception as e:
            logger.error(f"Model recalibration job failed: {e}")
            raise
    
    async def health_check_job(self) -> Dict[str, Any]:
        """Perform system health checks."""
        try:
            health_status = {
                "timestamp": datetime.now(),
                "data_freshness": {},
                "disk_usage": {},
                "job_statuses": {}
            }
            
            # Check data freshness
            load_history = self.storage.get_load_history()
            if not load_history.empty:
                latest_loads = load_history.groupby("table_name")["load_timestamp"].max()
                for table, timestamp in latest_loads.items():
                    hours_old = (datetime.now() - pd.to_datetime(timestamp)).total_seconds() / 3600
                    health_status["data_freshness"][table] = hours_old
            
            # Check recent job statuses
            recent_jobs = {
                job_id: result for job_id, result in self.job_results.items()
                if result.start_time > datetime.now() - timedelta(hours=24)
            }
            
            for job_id, result in recent_jobs.items():
                health_status["job_statuses"][job_id] = result.status.value
            
            # Save health report
            health_path = self.storage.data_dir / "health" / f"health_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            health_path.parent.mkdir(exist_ok=True)
            
            import json
            with open(health_path, "w") as f:
                json.dump(health_status, f, indent=2, default=str)
            
            return {
                'records_processed': len(health_status),
                'files_updated': [str(health_path)]
            }
        
        except Exception as e:
            logger.error(f"Health check job failed: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get status of a specific job."""
        return self.job_results.get(job_id)
    
    def get_all_job_statuses(self) -> Dict[str, JobResult]:
        """Get status of all jobs."""
        return self.job_results.copy()
    
    def get_recent_job_statuses(self, hours: int = 24) -> Dict[str, JobResult]:
        """Get recent job statuses."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return {
            job_id: result for job_id, result in self.job_results.items()
            if result.start_time > cutoff
        }
    
    async def manual_trigger_job(self, job_id: str) -> JobResult:
        """Manually trigger a specific job."""
        job_functions = {
            "update_stats": self.update_stats_job,
            "update_news": self.update_news_job,
            "update_weather": self.update_weather_job,
            "rebuild_features": self.rebuild_features_job,
            "generate_predictions": self.generate_predictions_job,
            "recalibrate_models": self.recalibrate_models_job,
            "health_check": self.health_check_job
        }
        
        if job_id not in job_functions:
            raise ValueError(f"Unknown job ID: {job_id}")
        
        logger.info(f"Manually triggering job: {job_id}")
        return await self.execute_job(job_functions[job_id], f"manual_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
