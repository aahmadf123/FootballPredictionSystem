"""Data validation utilities using Great Expectations."""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset
from loguru import logger


class DataValidator:
    """Data validation using Great Expectations."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.expectation_suites: Dict[str, ExpectationSuite] = {}
        self._setup_expectations()
    
    def _setup_expectations(self) -> None:
        """Setup expectation suites for different data types."""
        
        # Teams expectation suite
        teams_suite = ExpectationSuite(expectation_suite_name="teams_suite")
        teams_expectations = [
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "team_id"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "team_id"}},
            {"expectation_type": "expect_column_values_to_be_in_set", "kwargs": {"column": "league", "value_set": ["NFL", "NCAA"]}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "altitude", "min_value": 0, "max_value": 10000}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "season", "min_value": 1900, "max_value": 2100}},
        ]
        
        for exp in teams_expectations:
            teams_suite.add_expectation(**exp)
        
        self.expectation_suites["teams"] = teams_suite
        
        # Games expectation suite
        games_suite = ExpectationSuite(expectation_suite_name="games_suite")
        games_expectations = [
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "game_id"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "game_id"}},
            {"expectation_type": "expect_column_values_to_be_unique", "kwargs": {"column": "game_id"}},
            {"expectation_type": "expect_column_values_to_be_in_set", "kwargs": {"column": "league", "value_set": ["NFL", "NCAA"]}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "week", "min_value": 0, "max_value": 25}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "season", "min_value": 1900, "max_value": 2100}},
        ]
        
        for exp in games_expectations:
            games_suite.add_expectation(**exp)
        
        self.expectation_suites["games"] = games_suite
        
        # Play-by-play expectation suite
        pbp_suite = ExpectationSuite(expectation_suite_name="pbp_suite")
        pbp_expectations = [
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "game_id"}},
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "play_id"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "game_id"}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "quarter", "min_value": 1, "max_value": 5}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "down", "min_value": 1, "max_value": 4}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "yardline_100", "min_value": 0, "max_value": 100}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "yards", "min_value": -50, "max_value": 110}},
        ]
        
        for exp in pbp_expectations:
            pbp_suite.add_expectation(**exp)
        
        self.expectation_suites["pbp"] = pbp_suite
        
        # Players expectation suite
        players_suite = ExpectationSuite(expectation_suite_name="players_suite")
        players_expectations = [
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": "player_id"}},
            {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {"column": "player_id"}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "height", "min_value": 60, "max_value": 90}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "weight", "min_value": 150, "max_value": 400}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "age", "min_value": 16.0, "max_value": 40.0}},
            {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"column": "snaps", "min_value": 0, "max_value": 2000}},
        ]
        
        for exp in players_expectations:
            players_suite.add_expectation(**exp)
        
        self.expectation_suites["players"] = players_suite
    
    def validate_dataframe(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Validate a DataFrame against its expectation suite."""
        if table_name not in self.expectation_suites:
            logger.warning(f"No expectation suite found for table: {table_name}")
            return {"success": False, "message": f"No expectation suite for {table_name}"}
        
        try:
            dataset = PandasDataset(df)
            suite = self.expectation_suites[table_name]
            
            validation_result = dataset.validate(expectation_suite=suite)
            
            return {
                "success": validation_result.success,
                "statistics": validation_result.statistics,
                "results": validation_result.results,
                "evaluated_expectations": validation_result.statistics.get("evaluated_expectations", 0),
                "successful_expectations": validation_result.statistics.get("successful_expectations", 0)
            }
            
        except Exception as e:
            logger.error(f"Validation error for {table_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_all_tables(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Validate multiple tables."""
        results = {}
        for table_name, df in data_dict.items():
            results[table_name] = self.validate_dataframe(df, table_name)
        return results
    
    def check_referential_integrity(self, data_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """Check referential integrity between tables."""
        issues = []
        
        if "games" in data_dict and "pbp" in data_dict:
            games_ids = set(data_dict["games"]["game_id"])
            pbp_game_ids = set(data_dict["pbp"]["game_id"])
            orphaned_pbp = pbp_game_ids - games_ids
            if orphaned_pbp:
                issues.append(f"PBP records with missing game_ids: {len(orphaned_pbp)}")
        
        if "teams" in data_dict and "games" in data_dict:
            team_ids = set(data_dict["teams"]["team_id"])
            games_df = data_dict["games"]
            home_teams = set(games_df["home_id"])
            away_teams = set(games_df["away_id"])
            missing_home = home_teams - team_ids
            missing_away = away_teams - team_ids
            if missing_home:
                issues.append(f"Games with missing home team_ids: {len(missing_home)}")
            if missing_away:
                issues.append(f"Games with missing away team_ids: {len(missing_away)}")
        
        if "players" in data_dict and "teams" in data_dict:
            team_ids = set(data_dict["teams"]["team_id"])
            player_team_ids = set(data_dict["players"]["team_id"])
            orphaned_players = player_team_ids - team_ids
            if orphaned_players:
                issues.append(f"Players with missing team_ids: {len(orphaned_players)}")
        
        return issues
    
    def get_data_quality_report(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        validation_results = self.validate_all_tables(data_dict)
        integrity_issues = self.check_referential_integrity(data_dict)
        
        summary = {
            "total_tables": len(data_dict),
            "validation_results": validation_results,
            "referential_integrity_issues": integrity_issues,
            "overall_success": all(
                result.get("success", False) for result in validation_results.values()
            ) and len(integrity_issues) == 0
        }
        
        # Add row counts
        summary["row_counts"] = {
            table: len(df) for table, df in data_dict.items()
        }
        
        return summary
