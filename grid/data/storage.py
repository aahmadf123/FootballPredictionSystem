"""Data storage and management utilities."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from grid.config import GridConfig


class DataStorage:
    """Data storage manager for bronze, silver, and gold layers."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.data_dir = Path(config.app.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create layer directories
        self.raw_dir = self.data_dir / "raw"
        self.bronze_dir = self.data_dir / "bronze"
        self.silver_dir = self.data_dir / "silver"
        self.gold_dir = self.data_dir / "gold"
        self.snapshots_dir = self.data_dir / "snapshots"
        
        for dir_path in [self.raw_dir, self.bronze_dir, self.silver_dir, 
                        self.gold_dir, self.snapshots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self.metadata_db = self.data_dir / "metadata.db"
        self._init_metadata_db()
    
    def _init_metadata_db(self) -> None:
        """Initialize SQLite metadata database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        # Table for tracking data loads
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_loads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                table_name TEXT NOT NULL,
                load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                records_count INTEGER,
                file_path TEXT,
                status TEXT DEFAULT 'success',
                error_message TEXT,
                checksum TEXT
            )
        """)
        
        # Table for feature versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_versions (
                feature_version_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_hash TEXT,
                dependencies TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Table for model versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_version TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature_version_id TEXT,
                model_config TEXT,
                performance_metrics TEXT,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (feature_version_id) REFERENCES feature_versions (feature_version_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_raw_data(self, data: Dict[str, Any], source: str, date: str) -> Path:
        """Save raw JSON data."""
        source_dir = self.raw_dir / source / date
        source_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%H%M%S")
        file_path = source_dir / f"{source}_{timestamp}.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved raw data to {file_path}")
        return file_path
    
    def save_bronze_data(self, df: pd.DataFrame, table_name: str) -> Path:
        """Save bronze layer data as Parquet."""
        file_path = self.bronze_dir / f"{table_name}.parquet"
        df.to_parquet(file_path, index=False)
        
        self._log_data_load("bronze", table_name, len(df), str(file_path))
        logger.info(f"Saved bronze data to {file_path} ({len(df)} records)")
        return file_path
    
    def save_silver_data(self, df: pd.DataFrame, table_name: str) -> Path:
        """Save silver layer data as Parquet."""
        file_path = self.silver_dir / f"{table_name}.parquet"
        df.to_parquet(file_path, index=False)
        
        self._log_data_load("silver", table_name, len(df), str(file_path))
        logger.info(f"Saved silver data to {file_path} ({len(df)} records)")
        return file_path
    
    def save_gold_data(self, df: pd.DataFrame, feature_name: str, 
                      feature_version_id: str) -> Path:
        """Save gold layer feature data."""
        feature_dir = self.gold_dir / "feature_store" / feature_version_id
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = feature_dir / f"{feature_name}.parquet"
        df.to_parquet(file_path, index=False)
        
        self._log_data_load("gold", feature_name, len(df), str(file_path))
        logger.info(f"Saved gold data to {file_path} ({len(df)} records)")
        return file_path
    
    def load_data(self, table_name: str, layer: str = "silver") -> pd.DataFrame:
        """Load data from specified layer."""
        if layer == "bronze":
            file_path = self.bronze_dir / f"{table_name}.parquet"
        elif layer == "silver":
            file_path = self.silver_dir / f"{table_name}.parquet"
        elif layer == "gold":
            raise ValueError("Gold layer requires feature_version_id")
        else:
            raise ValueError(f"Unknown layer: {layer}")
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    def load_gold_data(self, feature_name: str, feature_version_id: str) -> pd.DataFrame:
        """Load gold layer feature data."""
        file_path = self.gold_dir / "feature_store" / feature_version_id / f"{feature_name}.parquet"
        
        if not file_path.exists():
            logger.warning(f"Feature file not found: {file_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    def upsert_data(self, df: pd.DataFrame, table_name: str, 
                   key_columns: List[str], layer: str = "silver") -> None:
        """Upsert data (insert or update existing records)."""
        existing_df = self.load_data(table_name, layer)
        
        if existing_df.empty:
            # No existing data, just save new data
            if layer == "silver":
                self.save_silver_data(df, table_name)
            elif layer == "bronze":
                self.save_bronze_data(df, table_name)
        else:
            # Merge with existing data
            merged_df = self._merge_dataframes(existing_df, df, key_columns)
            if layer == "silver":
                self.save_silver_data(merged_df, table_name)
            elif layer == "bronze":
                self.save_bronze_data(merged_df, table_name)
    
    def _merge_dataframes(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, 
                         key_columns: List[str]) -> pd.DataFrame:
        """Merge dataframes with upsert logic."""
        # Remove existing records that match keys in new data
        mask = ~existing_df.set_index(key_columns).index.isin(
            new_df.set_index(key_columns).index
        )
        filtered_existing = existing_df[mask]
        
        # Combine with new data
        return pd.concat([filtered_existing, new_df], ignore_index=True)
    
    def _log_data_load(self, source: str, table_name: str, records_count: int, 
                      file_path: str, status: str = "success", 
                      error_message: Optional[str] = None) -> None:
        """Log data load to metadata database."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO data_loads 
            (source, table_name, records_count, file_path, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (source, table_name, records_count, file_path, status, error_message))
        
        conn.commit()
        conn.close()
    
    def get_load_history(self, table_name: Optional[str] = None) -> pd.DataFrame:
        """Get data load history."""
        conn = sqlite3.connect(self.metadata_db)
        
        if table_name:
            query = "SELECT * FROM data_loads WHERE table_name = ? ORDER BY load_timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(table_name,))
        else:
            query = "SELECT * FROM data_loads ORDER BY load_timestamp DESC"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def create_feature_version(self, feature_version_id: str, config_hash: str, 
                              dependencies: List[str]) -> None:
        """Create a new feature version."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feature_versions 
            (feature_version_id, config_hash, dependencies)
            VALUES (?, ?, ?)
        """, (feature_version_id, config_hash, json.dumps(dependencies)))
        
        conn.commit()
        conn.close()
    
    def get_feature_versions(self) -> pd.DataFrame:
        """Get all feature versions."""
        conn = sqlite3.connect(self.metadata_db)
        df = pd.read_sql_query("SELECT * FROM feature_versions ORDER BY created_at DESC", conn)
        conn.close()
        return df
    
    def create_snapshot(self, season: int, week: int, 
                       feature_version_id: str, model_version: str) -> Path:
        """Create a snapshot pack for a specific week."""
        pack_dir = self.snapshots_dir / "packs" / f"{season}wk{week:02d}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy feature data
        feature_src = self.gold_dir / "feature_store" / feature_version_id
        feature_dst = pack_dir / "gold" / "feature_store" / feature_version_id
        feature_dst.mkdir(parents=True, exist_ok=True)
        
        if feature_src.exists():
            import shutil
            shutil.copytree(feature_src, feature_dst, dirs_exist_ok=True)
        
        # Create manifest
        manifest = {
            "season": season,
            "week": week,
            "feature_version_id": feature_version_id,
            "model_version": model_version,
            "created_at": datetime.now().isoformat(),
            "files": []
        }
        
        # List all files in pack
        for file_path in pack_dir.rglob("*"):
            if file_path.is_file():
                manifest["files"].append({
                    "path": str(file_path.relative_to(pack_dir)),
                    "size": file_path.stat().st_size
                })
        
        manifest_path = pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created snapshot pack at {pack_dir}")
        return pack_dir


class DuckDBAnalytics:
    """DuckDB interface for ad-hoc analytics."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.conn = duckdb.connect()
        self._register_tables()
    
    def _register_tables(self) -> None:
        """Register Parquet files as DuckDB tables."""
        silver_dir = self.data_dir / "silver"
        
        for parquet_file in silver_dir.glob("*.parquet"):
            table_name = parquet_file.stem
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW {table_name} AS 
                SELECT * FROM parquet_scan('{parquet_file}')
            """)
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.conn.execute(sql).df()
    
    def get_tables(self) -> List[str]:
        """Get list of available tables."""
        result = self.conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]
    
    def close(self) -> None:
        """Close connection."""
        self.conn.close()
