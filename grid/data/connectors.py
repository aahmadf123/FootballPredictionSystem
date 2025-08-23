"""Data connectors for various external sources."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel

from grid.config import Settings


class ConnectorError(Exception):
    """Custom exception for connector errors."""
    pass


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    async def acquire(self) -> None:
        """Acquire rate limit permission."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class BaseConnector(ABC):
    """Base class for all data connectors."""
    
    def __init__(self, settings: Settings, rate_limit: float = 1.0):
        self.settings = settings
        self.rate_limiter = RateLimiter(rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def fetch_raw(self, **kwargs) -> Dict[str, Any]:
        """Fetch raw data from source."""
        pass
    
    @abstractmethod
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate raw data structure."""
        pass
    
    @abstractmethod
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize raw data to standard schema."""
        pass
    
    async def fetch_and_process(self, **kwargs) -> Optional[pd.DataFrame]:
        """Complete fetch -> validate -> normalize pipeline."""
        try:
            await self.rate_limiter.acquire()
            raw_data = await self.fetch_raw(**kwargs)
            
            if not self.validate(raw_data):
                logger.error(f"Validation failed for {self.__class__.__name__}")
                return None
            
            normalized_data = self.normalize(raw_data)
            logger.info(f"Successfully processed {len(normalized_data)} records from {self.__class__.__name__}")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise ConnectorError(f"Failed to fetch and process data: {e}")


class CFBDConnector(BaseConnector):
    """College Football Data API connector."""
    
    BASE_URL = "https://api.collegefootballdata.com"
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=1.0)  # CFBD rate limit
        if not settings.cfbd_api_key:
            raise ConnectorError("CFBD API key not provided")
        self.api_key = settings.cfbd_api_key
    
    async def fetch_raw(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch raw data from CFBD API."""
        if not self.session:
            raise ConnectorError("Session not initialized")
        
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with self.session.get(url, headers=headers, params=params or {}) as response:
            if response.status == 429:
                logger.warning("Rate limit hit, waiting...")
                await asyncio.sleep(60)
                return await self.fetch_raw(endpoint, params)
            
            response.raise_for_status()
            return await response.json()
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate CFBD response structure."""
        return isinstance(raw_data, (list, dict))
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize CFBD data to standard schema."""
        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)
        return pd.DataFrame([raw_data])
    
    async def get_teams(self, year: int) -> pd.DataFrame:
        """Get team data for a specific year."""
        params = {"year": year}
        raw_data = await self.fetch_raw("teams", params)
        df = self.normalize(raw_data)
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "team_id": df["school"],
            "league": "NCAA",
            "conf": df["conference"],
            "division": df.get("division", "FBS"),
            "venue_id": df["school"],  # Simplified
            "tz": "UTC",  # Default, should be enhanced
            "altitude": 0,  # Default, should be enhanced
            "surface": "grass",  # Default, should be enhanced
            "coach_id": df["school"] + "_coach",  # Simplified
            "season": year
        })
        
        return df_mapped
    
    async def get_games(self, year: int, week: Optional[int] = None) -> pd.DataFrame:
        """Get game data."""
        params = {"year": year}
        if week:
            params["week"] = week
        
        raw_data = await self.fetch_raw("games", params)
        df = self.normalize(raw_data)
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "game_id": "NCAA_" + df["season"].astype(str) + "_W" + df["week"].astype(str) + "_" + df["away_team"] + "_" + df["home_team"],
            "league": "NCAA",
            "season": df["season"],
            "week": df["week"],
            "home_id": df["home_team"],
            "away_id": df["away_team"],
            "kickoff_utc": pd.to_datetime(df["start_date"]),
            "venue_id": df.get("venue", "unknown"),
            "ref_crew_id": "unknown",  # Not available in CFBD
            "weather_id": None,
            "line_implied": df.get("lines", [{}])[0].get("spread") if df.get("lines") else None,
            "total_implied": None,
            "home_score": df.get("home_points"),
            "away_score": df.get("away_points"),
            "status": "completed" if df.get("home_points") is not None else "scheduled"
        })
        
        return df_mapped
    
    async def get_plays(self, year: int, week: int, team: Optional[str] = None) -> pd.DataFrame:
        """Get play-by-play data."""
        params = {"year": year, "week": week}
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("plays", params)
        df = self.normalize(raw_data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "game_id": "NCAA_" + df["year"].astype(str) + "_W" + df["week"].astype(str) + "_" + df["away"] + "_" + df["home"],
            "play_id": df["id"],
            "sec_left": df["clock_minutes"] * 60 + df["clock_seconds"],
            "quarter": df["period"],
            "down": df["down"],
            "dist": df["distance"],
            "yardline_100": 100 - df["yard_line"],
            "offense_id": df["offense"],
            "defense_id": df["defense"],
            "play_type": df["play_type"],
            "yards": df.get("yards_gained", 0),
            "ep_before": df.get("ep_before", 0.0),
            "ep_after": df.get("ep_after", 0.0),
            "epa": df.get("epa", 0.0),
            "wpa": df.get("wpa", 0.0)
        })
        
        return df_mapped


class NFLConnector(BaseConnector):
    """NFL data connector using ESPN API."""
    
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=2.0)  # Conservative rate limit
    
    async def fetch_raw(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch raw data from ESPN NFL API."""
        if not self.session:
            raise ConnectorError("Session not initialized")
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        async with self.session.get(url, params=params or {}) as response:
            response.raise_for_status()
            return await response.json()
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate ESPN NFL response structure."""
        return isinstance(raw_data, dict) and "events" in raw_data
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize ESPN NFL data to standard schema."""
        events = raw_data.get("events", [])
        return pd.DataFrame(events)
    
    async def get_teams(self, year: int) -> pd.DataFrame:
        """Get NFL team data."""
        raw_data = await self.fetch_raw("teams")
        teams = raw_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        
        df = pd.DataFrame([{
            "team_id": team["team"]["abbreviation"],
            "league": "NFL",
            "conf": team["team"].get("conference", {}).get("name"),
            "division": team["team"].get("division", {}).get("name"),
            "venue_id": team["team"].get("venue", {}).get("id", "unknown"),
            "tz": "UTC",  # Default
            "altitude": 0,  # Default
            "surface": "unknown",  # Default
            "coach_id": f"{team['team']['abbreviation']}_coach",
            "season": year
        } for team in teams])
        
        return df
    
    async def get_games(self, year: int, week: Optional[int] = None) -> pd.DataFrame:
        """Get NFL game data."""
        params = {"season": year, "seasontype": 2}  # Regular season
        if week:
            params["week"] = week
        
        raw_data = await self.fetch_raw("scoreboard", params)
        events = raw_data.get("events", [])
        
        games_data = []
        for event in events:
            competitions = event.get("competitions", [])
            for comp in competitions:
                competitors = comp.get("competitors", [])
                home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})
                
                games_data.append({
                    "game_id": f"NFL_{year}_W{week or 1}_{away_team.get('team', {}).get('abbreviation', 'UNK')}_{home_team.get('team', {}).get('abbreviation', 'UNK')}",
                    "league": "NFL",
                    "season": year,
                    "week": week or 1,
                    "home_id": home_team.get("team", {}).get("abbreviation"),
                    "away_id": away_team.get("team", {}).get("abbreviation"),
                    "kickoff_utc": pd.to_datetime(event.get("date")),
                    "venue_id": comp.get("venue", {}).get("id", "unknown"),
                    "ref_crew_id": "unknown",
                    "weather_id": None,
                    "line_implied": None,  # Not available in free ESPN API
                    "total_implied": None,
                    "home_score": int(home_team.get("score", 0)) if home_team.get("score") else None,
                    "away_score": int(away_team.get("score", 0)) if away_team.get("score") else None,
                    "status": comp.get("status", {}).get("type", {}).get("name", "scheduled")
                })
        
        return pd.DataFrame(games_data)


class NewsConnector(BaseConnector):
    """News feed connector for RSS feeds."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=0.5)  # 0.5 requests per second
        self.news_feeds = self._parse_news_feeds(settings.news_feeds or "")
    
    def _parse_news_feeds(self, feeds_str: str) -> List[str]:
        """Parse semicolon-separated news feeds."""
        return [feed.strip() for feed in feeds_str.split(";") if feed.strip()]
    
    async def fetch_raw(self, feed_url: str) -> Dict[str, Any]:
        """Fetch raw RSS feed data."""
        try:
            # Using feedparser synchronously as it doesn't have async support
            feed = feedparser.parse(feed_url)
            return {"entries": feed.entries, "feed": feed.feed}
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            return {"entries": [], "feed": {}}
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate RSS feed structure."""
        return "entries" in raw_data and isinstance(raw_data["entries"], list)
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize RSS feed data to news schema."""
        entries = raw_data.get("entries", [])
        
        news_data = []
        for entry in entries:
            news_data.append({
                "news_id": entry.get("id", entry.get("link", "unknown")),
                "published_at": pd.to_datetime(entry.get("published", datetime.now())),
                "url": entry.get("link", ""),
                "team_ids": [],  # Will be filled by NER
                "player_ids": [],  # Will be filled by NER
                "coach_ids": [],  # Will be filled by NER
                "summary": entry.get("summary", entry.get("title", "")),
                "impact": {},  # Will be filled by analysis
                "sentiment": 0.0  # Will be filled by sentiment analysis
            })
        
        return pd.DataFrame(news_data)
    
    async def get_all_news(self) -> pd.DataFrame:
        """Get news from all configured feeds."""
        all_news = []
        
        for feed_url in self.news_feeds:
            try:
                raw_data = await self.fetch_raw(feed_url)
                if self.validate(raw_data):
                    news_df = self.normalize(raw_data)
                    all_news.append(news_df)
            except Exception as e:
                logger.error(f"Error processing feed {feed_url}: {e}")
        
        if all_news:
            return pd.concat(all_news, ignore_index=True)
        return pd.DataFrame()


class WeatherConnector(BaseConnector):
    """Weather data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=1.0)
        if not settings.weather_api_key:
            logger.warning("Weather API key not provided")
        self.api_key = settings.weather_api_key
    
    async def fetch_raw(self, location: str, date: str) -> Dict[str, Any]:
        """Fetch weather data for location and date."""
        if not self.api_key:
            # Return default weather when no API key
            return {
                "location": location,
                "date": date,
                "temperature": 70,
                "wind_speed": 5,
                "precipitation": 0,
                "conditions": "clear",
                "dome": False,
                "altitude": 0
            }
        
        # Weather API integration would go here
        # For now, return realistic default values
        import random
        return {
            "location": location,
            "date": date,
            "temperature": random.randint(20, 85),
            "wind_speed": random.randint(0, 15),
            "precipitation": random.choice([0, 0, 0, 0.1, 0.2]),
            "conditions": random.choice(["clear", "cloudy", "rain", "snow"]),
            "dome": "dome" in location.lower(),
            "altitude": 0
        }
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate weather data structure."""
        required_fields = ["location", "date", "temperature", "wind_speed"]
        return all(field in raw_data for field in required_fields)
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize weather data."""
        return pd.DataFrame([raw_data])


async def main():
    """Example usage of connectors."""
    settings = Settings()
    
    # Example: Fetch CFBD data
    async with CFBDConnector(settings) as cfbd:
        teams = await cfbd.get_teams(2023)
        logger.info(f"Fetched {len(teams)} teams")
        
        games = await cfbd.get_games(2023, 1)
        logger.info(f"Fetched {len(games)} games")


if __name__ == "__main__":
    asyncio.run(main())
