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
    
    async def get_roster(self, year: int, team: Optional[str] = None) -> pd.DataFrame:
        """Get roster data for players."""
        params = {"year": year}
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("roster", params)
        df = self.normalize(raw_data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "player_id": df["athlete_id"].astype(str),
            "team_id": df["team"],
            "pos": df["position"],
            "depth": df.get("depth_chart_order", 99),
            "height": df.get("height", 0),
            "weight": df.get("weight", 0),
            "age": year - pd.to_datetime(df.get("birth_date", "1900-01-01")).dt.year,
            "snaps": 0,  # Will be filled from advanced stats
            "season": year,
            "class_year": df.get("year", "FR"),
            "jersey_number": df.get("jersey", 0)
        })
        
        return df_mapped
    
    async def get_recruiting(self, year: int, team: Optional[str] = None) -> pd.DataFrame:
        """Get recruiting rankings and talent composite data."""
        params = {"year": year}
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("recruiting/teams", params)
        df = self.normalize(raw_data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "team_id": df["team"],
            "season": year,
            "composite_talent": df.get("talent", 0.0),
            "class_rank": df.get("rank", 999),
            "total_commits": df.get("commits", 0),
            "avg_rating": df.get("averageRating", 0.0),
            "total_points": df.get("points", 0.0)
        })
        
        return df_mapped
    
    async def get_transfer_portal(self, year: int, team: Optional[str] = None) -> pd.DataFrame:
        """Get transfer portal data."""
        params = {"year": year}
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("player/portal", params)
        df = self.normalize(raw_data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "player_id": df["athlete_id"].astype(str),
            "season": year,
            "team_id": df.get("destination", ""),
            "direction": "in" if pd.notna(df.get("destination")) else "out",
            "prev_team": df.get("origin", ""),
            "proj_role": df.get("rating", "unknown"),
            "portal_date": pd.to_datetime(df.get("transferDate")),
            "eligibility_year": df.get("eligibility", "")
        })
        
        return df_mapped
    
    async def get_advanced_stats(self, year: int, week: Optional[int] = None, team: Optional[str] = None) -> pd.DataFrame:
        """Get advanced team statistics."""
        params = {"year": year}
        if week:
            params["week"] = week
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("stats/game/advanced", params)
        df = self.normalize(raw_data)
        
        return df
    
    async def get_sp_ratings(self, year: int) -> pd.DataFrame:
        """Get S&P+ ratings data."""
        params = {"year": year}
        
        raw_data = await self.fetch_raw("ratings/sp", params)
        df = self.normalize(raw_data)
        
        return df
    
    async def get_betting_lines(self, year: int, week: Optional[int] = None, team: Optional[str] = None) -> pd.DataFrame:
        """Get betting lines and spreads."""
        params = {"year": year}
        if week:
            params["week"] = week
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("lines", params)
        df = self.normalize(raw_data)
        
        return df
    
    async def get_coaching_data(self, year: int, team: Optional[str] = None) -> pd.DataFrame:
        """Get coaching staff data."""
        params = {"year": year}
        if team:
            params["team"] = team
        
        raw_data = await self.fetch_raw("coaches", params)
        df = self.normalize(raw_data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Map to our schema
        df_mapped = pd.DataFrame({
            "coach_id": df["first_name"] + "_" + df["last_name"] + "_" + df["school"],
            "name": df["first_name"] + " " + df["last_name"],
            "role": df["position"],
            "tenure": year - pd.to_datetime(df.get("hire_date", "2023-01-01")).dt.year,
            "season": year,
            "team_id": df["school"]
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
    
    async def get_pbp_data(self, game_id: str) -> pd.DataFrame:
        """Get NFL play-by-play data using ESPN NFL API."""
        # Extract game info from game_id format: NFL_YYYY_WXX_AWAY_HOME
        parts = game_id.split("_")
        if len(parts) < 5:
            logger.error(f"Invalid game_id format: {game_id}")
            return pd.DataFrame()
        
        season = parts[1]
        week = parts[2][1:]  # Remove 'W' prefix
        
        # For now, return empty DataFrame as ESPN's free API doesn't provide detailed PBP
        # In production, this would integrate with ESPN's premium API or other NFL data sources
        logger.warning(f"NFL PBP data not available for {game_id} - requires premium ESPN API or NFL Next Gen Stats")
        return pd.DataFrame()
    
    async def get_player_stats(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Get NFL player statistics."""
        params = {"season": season, "seasontype": 2}
        if week:
            params["week"] = week
        
        # This would require ESPN's premium API for detailed player stats
        # For now, return empty DataFrame
        logger.warning("NFL player stats require premium ESPN API access")
        return pd.DataFrame()
    
    async def get_depth_charts(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Get NFL depth chart data."""
        # This requires specialized NFL APIs or web scraping
        logger.warning("NFL depth charts require specialized data sources")
        return pd.DataFrame()


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
    
    async def get_forecast(self, location: str, date: str) -> pd.DataFrame:
        """Get weather forecast for specific location and date."""
        if not self.api_key:
            logger.warning("Weather API key not provided, using default forecast")
            return self.normalize(await self.fetch_raw(location, date))
        
        # OpenWeatherMap API integration
        if not self.session:
            raise ConnectorError("Session not initialized")
        
        # Get coordinates first
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
        geo_params = {
            "q": location,
            "limit": 1,
            "appid": self.api_key
        }
        
        async with self.session.get(geo_url, params=geo_params) as response:
            if response.status != 200:
                logger.error(f"Geocoding failed for {location}")
                return self.normalize(await self.fetch_raw(location, date))
            
            geo_data = await response.json()
            if not geo_data:
                logger.error(f"No coordinates found for {location}")
                return self.normalize(await self.fetch_raw(location, date))
        
        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
        
        # Get forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"
        }
        
        async with self.session.get(forecast_url, params=forecast_params) as response:
            if response.status != 200:
                logger.error(f"Weather forecast failed for {location}")
                return self.normalize(await self.fetch_raw(location, date))
            
            forecast_data = await response.json()
            
            # Find forecast closest to game date
            target_date = pd.to_datetime(date)
            closest_forecast = None
            min_diff = float('inf')
            
            for forecast in forecast_data.get("list", []):
                forecast_date = pd.to_datetime(forecast["dt"], unit='s')
                diff = abs((forecast_date - target_date).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_forecast = forecast
            
            if closest_forecast:
                weather_data = {
                    "location": location,
                    "date": date,
                    "temperature": closest_forecast["main"]["temp"],
                    "wind_speed": closest_forecast["wind"]["speed"],
                    "precipitation": closest_forecast.get("rain", {}).get("3h", 0),
                    "conditions": closest_forecast["weather"][0]["main"].lower(),
                    "dome": "dome" in location.lower(),
                    "altitude": 0,  # Would need additional API for altitude
                    "humidity": closest_forecast["main"]["humidity"],
                    "pressure": closest_forecast["main"]["pressure"]
                }
                return pd.DataFrame([weather_data])
        
        # Fallback to default if API fails
        return self.normalize(await self.fetch_raw(location, date))


async def main():
    """Example usage of connectors."""
    settings = Settings()
    
    # Example: Fetch CFBD data
    async with CFBDConnector(settings) as cfbd:
        teams = await cfbd.get_teams(2023)
        logger.info(f"Fetched {len(teams)} teams")
        
        games = await cfbd.get_games(2023, 1)
        logger.info(f"Fetched {len(games)} games")


class RecruitingConnector(BaseConnector):
    """247Sports and Rivals recruiting data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=0.5)  # Conservative rate limit
        self.api_key_247 = getattr(settings, 'recruiting_247_api_key', None)
        self.api_key_rivals = getattr(settings, 'recruiting_rivals_api_key', None)
    
    async def fetch_raw(self, source: str = "247", year: int = 2024, team: Optional[str] = None) -> Dict[str, Any]:
        """Fetch recruiting data from 247Sports or Rivals."""
        # This would integrate with 247Sports or Rivals APIs when available
        # For now, return mock data structure
        logger.warning(f"Recruiting data connector not fully implemented for {source}")
        
        return {
            "teams": [
                {
                    "team": team or "Alabama",
                    "year": year,
                    "rank": 1,
                    "points": 325.5,
                    "avg_rating": 94.2,
                    "commits": 25,
                    "five_stars": 8,
                    "four_stars": 15,
                    "three_stars": 2
                }
            ]
        }
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate recruiting data structure."""
        return "teams" in raw_data and isinstance(raw_data["teams"], list)
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize recruiting data."""
        teams = raw_data.get("teams", [])
        return pd.DataFrame(teams)


class InjuryConnector(BaseConnector):
    """Injury and transaction data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=1.0)
        self.espn_injury_key = getattr(settings, 'espn_injury_api_key', None)
        self.fantasypros_key = getattr(settings, 'fantasypros_api_key', None)
    
    async def fetch_raw(self, league: str = "NFL", week: Optional[int] = None) -> Dict[str, Any]:
        """Fetch injury reports."""
        # Mock injury data structure
        return {
            "injuries": [
                {
                    "player_id": "12345",
                    "game_id": "NFL_2024_W10_BUF_MIA",
                    "status": "questionable",
                    "designation": "ankle",
                    "note": "Sprained ankle in practice",
                    "impact": 3,
                    "as_of": "2024-11-01T15:30:00Z"
                }
            ]
        }
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate injury data structure."""
        return "injuries" in raw_data and isinstance(raw_data["injuries"], list)
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize injury data."""
        injuries = raw_data.get("injuries", [])
        df = pd.DataFrame(injuries)
        if not df.empty:
            df["as_of"] = pd.to_datetime(df["as_of"])
        return df


class RefereeConnector(BaseConnector):
    """Referee crew data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=0.5)
        self.fo_api_key = getattr(settings, 'fo_api_key', None)
    
    async def fetch_raw(self, season: int, week: Optional[int] = None) -> Dict[str, Any]:
        """Fetch referee crew data."""
        # Mock referee data
        return {
            "crews": [
                {
                    "ref_crew_id": "CREW_001",
                    "season": season,
                    "referee_name": "John Smith",
                    "penalties_per_game": 12.5,
                    "std": 3.2,
                    "pace_adj": 1.05,
                    "games_officiated": 156
                }
            ]
        }
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate referee data structure."""
        return "crews" in raw_data and isinstance(raw_data["crews"], list)
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize referee data."""
        crews = raw_data.get("crews", [])
        return pd.DataFrame(crews)


class VenueConnector(BaseConnector):
    """Venue and travel data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=1.0)
        self.google_maps_key = getattr(settings, 'google_maps_api_key', None)
        self.timezone_key = getattr(settings, 'timezone_api_key', None)
    
    async def fetch_raw(self, home_venue: str, away_venue: str) -> Dict[str, Any]:
        """Fetch venue and travel data."""
        # Mock venue data
        return {
            "home_venue": {
                "venue_id": home_venue,
                "latitude": 42.3601,
                "longitude": -71.0589,
                "altitude": 150,
                "surface": "fieldturf",
                "dome": False,
                "capacity": 65000,
                "timezone": "America/New_York"
            },
            "away_venue": {
                "venue_id": away_venue,
                "latitude": 25.9581,
                "longitude": -80.2389,
                "altitude": 8,
                "surface": "grass",
                "dome": False,
                "capacity": 67000,
                "timezone": "America/New_York"
            },
            "travel_distance": 1255.7,
            "timezone_delta": 0,
            "short_week": False
        }
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate venue data structure."""
        return "home_venue" in raw_data and "away_venue" in raw_data
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize venue data."""
        return pd.DataFrame([raw_data])


class MarketDataConnector(BaseConnector):
    """Betting market data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=1.0)
        self.odds_api_key = getattr(settings, 'odds_api_key', None)
        self.fanduel_key = getattr(settings, 'fanduel_api_key', None)
        self.draftkings_key = getattr(settings, 'draftkings_api_key', None)
    
    async def fetch_raw(self, league: str = "NFL", upcoming_only: bool = True) -> Dict[str, Any]:
        """Fetch betting market data."""
        if not self.odds_api_key:
            logger.warning("Odds API key not provided, using mock data")
            return {
                "data": [
                    {
                        "game_id": "NFL_2024_W10_BUF_MIA",
                        "home_team": "MIA",
                        "away_team": "BUF",
                        "home_price": 2.1,
                        "away_price": 1.75,
                        "point_spread": -3.5,
                        "total": 47.5,
                        "bookmaker": "fanduel",
                        "last_update": "2024-11-01T12:00:00Z"
                    }
                ]
            }
        
        # The Odds API integration
        if not self.session:
            raise ConnectorError("Session not initialized")
        
        sport_key = "americanfootball_nfl" if league == "NFL" else "americanfootball_ncaaf"
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        
        params = {
            "api_key": self.odds_api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Odds API request failed: {response.status}")
                return {"data": []}
            
            return await response.json()
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate market data structure."""
        return isinstance(raw_data, (dict, list))
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize market data."""
        if isinstance(raw_data, dict):
            data = raw_data.get("data", raw_data)
        else:
            data = raw_data
        
        return pd.DataFrame(data)


class SpecialTeamsConnector(BaseConnector):
    """Detailed special teams data connector."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, rate_limit=1.0)
    
    async def fetch_raw(self, league: str, season: int, week: Optional[int] = None) -> Dict[str, Any]:
        """Fetch special teams data."""
        # Mock special teams data
        return {
            "special_teams": [
                {
                    "game_id": f"{league}_2024_W{week or 1}_TEAM1_TEAM2",
                    "team_id": "TEAM1",
                    "fg_att": 3,
                    "fg_made": 2,
                    "fg_dist_bins": {"20-29": 1, "30-39": 1, "40-49": 1, "50+": 0},
                    "punt_net_avg": 42.5,
                    "pr_ypa": 8.2,
                    "kr_ypa": 23.1,
                    "blocks": 0,
                    "fg_long": 47,
                    "punt_inside_20": 3,
                    "touchbacks": 4
                }
            ]
        }
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """Validate special teams data structure."""
        return "special_teams" in raw_data and isinstance(raw_data["special_teams"], list)
    
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize special teams data."""
        st_data = raw_data.get("special_teams", [])
        return pd.DataFrame(st_data)


if __name__ == "__main__":
    asyncio.run(main())
