"""Data factory for orchestrating all connectors and providing unified data access."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from loguru import logger

from grid.config import Settings, GridConfig
from grid.data.connectors import (
    CFBDConnector,
    NFLConnector,
    NewsConnector,
    WeatherConnector,
    RecruitingConnector,
    InjuryConnector,
    RefereeConnector,
    VenueConnector,
    MarketDataConnector,
    SpecialTeamsConnector,
    ConnectorError
)
from grid.data.storage import DataStorage


class DataFactory:
    """Factory for managing and orchestrating all data connectors."""
    
    def __init__(self, settings: Settings, config: GridConfig):
        self.settings = settings
        self.config = config
        self.storage = DataStorage(config)
        
        # Initialize all connectors
        self.connectors = {}
        self._init_connectors()
    
    def _init_connectors(self):
        """Initialize all available connectors."""
        try:
            # Core football data connectors
            if self.settings.cfbd_api_key:
                self.connectors['cfbd'] = CFBDConnector(self.settings)
                logger.info("CFBD connector initialized")
            else:
                logger.warning("CFBD API key not found - NCAA data will be limited")
            
            self.connectors['nfl'] = NFLConnector(self.settings)
            logger.info("NFL connector initialized")
            
            # Weather connector
            self.connectors['weather'] = WeatherConnector(self.settings)
            logger.info("Weather connector initialized")
            
            # News connector
            if self.settings.news_feeds:
                self.connectors['news'] = NewsConnector(self.settings)
                logger.info("News connector initialized")
            
            # Recruiting connector
            if self.settings.recruiting_247_api_key or self.settings.recruiting_rivals_api_key:
                self.connectors['recruiting'] = RecruitingConnector(self.settings)
                logger.info("Recruiting connector initialized")
            
            # Injury connector
            if self.settings.espn_injury_api_key or self.settings.fantasypros_api_key:
                self.connectors['injury'] = InjuryConnector(self.settings)
                logger.info("Injury connector initialized")
            
            # Referee connector
            if self.settings.fo_api_key:
                self.connectors['referee'] = RefereeConnector(self.settings)
                logger.info("Referee connector initialized")
            
            # Venue connector
            if self.settings.google_maps_api_key:
                self.connectors['venue'] = VenueConnector(self.settings)
                logger.info("Venue connector initialized")
            
            # Market data connector
            if self.settings.odds_api_key:
                self.connectors['market'] = MarketDataConnector(self.settings)
                logger.info("Market data connector initialized")
            
            # Special teams connector
            self.connectors['special_teams'] = SpecialTeamsConnector(self.settings)
            logger.info("Special teams connector initialized")
            
        except Exception as e:
            logger.error(f"Error initializing connectors: {e}")
            raise ConnectorError(f"Failed to initialize data factory: {e}")
    
    async def fetch_all_data(
        self, 
        league: str, 
        season: int, 
        week: Optional[int] = None,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all available data for a given league, season, and optionally week."""
        results = {}
        
        logger.info(f"Fetching all data for {league} {season}" + (f" Week {week}" if week else ""))
        
        # Core game and team data
        results.update(await self._fetch_core_data(league, season, week))
        
        # Player and roster data
        results.update(await self._fetch_player_data(league, season, week))
        
        # Special teams data
        results.update(await self._fetch_special_teams_data(league, season, week))
        
        # Recruiting and transfer portal data (NCAA only)
        if league == "NCAA":
            results.update(await self._fetch_recruiting_data(season))
        
        # Contextual data
        results.update(await self._fetch_contextual_data(league, season, week))
        
        # Market data
        results.update(await self._fetch_market_data(league))
        
        # Store results
        if results:
            await self._store_results(results, league, season, week)
        
        return results
    
    async def _fetch_core_data(self, league: str, season: int, week: Optional[int]) -> Dict[str, pd.DataFrame]:
        """Fetch core team and game data."""
        results = {}
        
        if league == "NFL" and 'nfl' in self.connectors:
            async with self.connectors['nfl'] as nfl:
                try:
                    # Fetch teams
                    teams = await nfl.get_teams(season)
                    if not teams.empty:
                        results['teams'] = teams
                        logger.info(f"Fetched {len(teams)} NFL teams")
                    
                    # Fetch games
                    games = await nfl.get_games(season, week)
                    if not games.empty:
                        results['games'] = games
                        logger.info(f"Fetched {len(games)} NFL games")
                        
                except Exception as e:
                    logger.error(f"Error fetching NFL core data: {e}")
        
        elif league == "NCAA" and 'cfbd' in self.connectors:
            async with self.connectors['cfbd'] as cfbd:
                try:
                    # Fetch teams
                    teams = await cfbd.get_teams(season)
                    if not teams.empty:
                        results['teams'] = teams
                        logger.info(f"Fetched {len(teams)} NCAA teams")
                    
                    # Fetch games
                    games = await cfbd.get_games(season, week)
                    if not games.empty:
                        results['games'] = games
                        logger.info(f"Fetched {len(games)} NCAA games")
                    
                    # Fetch play-by-play data
                    if week:
                        pbp = await cfbd.get_plays(season, week)
                        if not pbp.empty:
                            results['pbp'] = pbp
                            logger.info(f"Fetched {len(pbp)} NCAA plays")
                    
                    # Fetch advanced stats
                    advanced_stats = await cfbd.get_advanced_stats(season, week)
                    if not advanced_stats.empty:
                        results['advanced_stats'] = advanced_stats
                        logger.info(f"Fetched NCAA advanced stats")
                    
                    # Fetch ratings
                    sp_ratings = await cfbd.get_sp_ratings(season)
                    if not sp_ratings.empty:
                        results['sp_ratings'] = sp_ratings
                        logger.info(f"Fetched S&P+ ratings")
                    
                    # Fetch coaching data
                    coaches = await cfbd.get_coaching_data(season)
                    if not coaches.empty:
                        results['coaches'] = coaches
                        logger.info(f"Fetched {len(coaches)} coaching records")
                        
                except Exception as e:
                    logger.error(f"Error fetching NCAA core data: {e}")
        
        return results
    
    async def _fetch_player_data(self, league: str, season: int, week: Optional[int]) -> Dict[str, pd.DataFrame]:
        """Fetch player-level data."""
        results = {}
        
        if league == "NCAA" and 'cfbd' in self.connectors:
            async with self.connectors['cfbd'] as cfbd:
                try:
                    # Fetch roster data
                    roster = await cfbd.get_roster(season)
                    if not roster.empty:
                        results['players'] = roster
                        logger.info(f"Fetched {len(roster)} NCAA player records")
                        
                except Exception as e:
                    logger.error(f"Error fetching NCAA player data: {e}")
        
        # Injury data for both leagues
        if 'injury' in self.connectors:
            async with self.connectors['injury'] as injury:
                try:
                    injuries = await injury.fetch_and_process(league=league, week=week)
                    if injuries is not None and not injuries.empty:
                        results['injuries'] = injuries
                        logger.info(f"Fetched {len(injuries)} injury records")
                        
                except Exception as e:
                    logger.error(f"Error fetching injury data: {e}")
        
        return results
    
    async def _fetch_special_teams_data(self, league: str, season: int, week: Optional[int]) -> Dict[str, pd.DataFrame]:
        """Fetch special teams data."""
        results = {}
        
        if 'special_teams' in self.connectors:
            async with self.connectors['special_teams'] as st:
                try:
                    st_data = await st.fetch_and_process(
                        league=league, 
                        season=season, 
                        week=week
                    )
                    if st_data is not None and not st_data.empty:
                        results['special_teams'] = st_data
                        logger.info(f"Fetched {len(st_data)} special teams records")
                        
                except Exception as e:
                    logger.error(f"Error fetching special teams data: {e}")
        
        return results
    
    async def _fetch_recruiting_data(self, season: int) -> Dict[str, pd.DataFrame]:
        """Fetch recruiting and transfer portal data."""
        results = {}
        
        if 'recruiting' in self.connectors:
            async with self.connectors['recruiting'] as recruiting:
                try:
                    recruiting_data = await recruiting.fetch_and_process(year=season)
                    if recruiting_data is not None and not recruiting_data.empty:
                        results['recruiting'] = recruiting_data
                        logger.info(f"Fetched {len(recruiting_data)} recruiting records")
                        
                except Exception as e:
                    logger.error(f"Error fetching recruiting data: {e}")
        
        # Transfer portal data
        if 'cfbd' in self.connectors:
            async with self.connectors['cfbd'] as cfbd:
                try:
                    transfers = await cfbd.get_transfer_portal(season)
                    if not transfers.empty:
                        results['transfers'] = transfers
                        logger.info(f"Fetched {len(transfers)} transfer records")
                        
                except Exception as e:
                    logger.error(f"Error fetching transfer data: {e}")
        
        return results
    
    async def _fetch_contextual_data(self, league: str, season: int, week: Optional[int]) -> Dict[str, pd.DataFrame]:
        """Fetch contextual data like weather, venues, referees."""
        results = {}
        
        # Weather data (will be fetched per game as needed)
        # Venue data (will be fetched per matchup as needed)
        
        # Referee data
        if 'referee' in self.connectors:
            async with self.connectors['referee'] as referee:
                try:
                    ref_data = await referee.fetch_and_process(season=season, week=week)
                    if ref_data is not None and not ref_data.empty:
                        results['referees'] = ref_data
                        logger.info(f"Fetched {len(ref_data)} referee crew records")
                        
                except Exception as e:
                    logger.error(f"Error fetching referee data: {e}")
        
        # News data
        if 'news' in self.connectors:
            async with self.connectors['news'] as news:
                try:
                    news_data = await news.get_all_news()
                    if not news_data.empty:
                        # Filter recent news (last 7 days)
                        recent_cutoff = datetime.now() - timedelta(days=7)
                        recent_news = news_data[
                            pd.to_datetime(news_data['published_at']) >= recent_cutoff
                        ]
                        if not recent_news.empty:
                            results['news'] = recent_news
                            logger.info(f"Fetched {len(recent_news)} recent news articles")
                            
                except Exception as e:
                    logger.error(f"Error fetching news data: {e}")
        
        return results
    
    async def _fetch_market_data(self, league: str) -> Dict[str, pd.DataFrame]:
        """Fetch betting market data."""
        results = {}
        
        if 'market' in self.connectors:
            async with self.connectors['market'] as market:
                try:
                    market_data = await market.fetch_and_process(league=league)
                    if market_data is not None and not market_data.empty:
                        results['market_data'] = market_data
                        logger.info(f"Fetched {len(market_data)} market records")
                        
                except Exception as e:
                    logger.error(f"Error fetching market data: {e}")
        
        return results
    
    async def _store_results(
        self, 
        results: Dict[str, pd.DataFrame], 
        league: str, 
        season: int, 
        week: Optional[int]
    ):
        """Store fetched data using the storage system."""
        try:
            for data_type, df in results.items():
                if not df.empty:
                    await self.storage.store_bronze_data(
                        data=df,
                        source=data_type,
                        league=league,
                        season=season,
                        week=week
                    )
                    logger.info(f"Stored {len(df)} {data_type} records")
                    
        except Exception as e:
            logger.error(f"Error storing results: {e}")
    
    async def get_weather_for_game(self, game_id: str, venue: str, game_date: str) -> Optional[pd.DataFrame]:
        """Get weather data for a specific game."""
        if 'weather' not in self.connectors:
            return None
        
        async with self.connectors['weather'] as weather:
            try:
                weather_data = await weather.get_forecast(venue, game_date)
                if not weather_data.empty:
                    logger.info(f"Fetched weather for {game_id} at {venue}")
                    return weather_data
                    
            except Exception as e:
                logger.error(f"Error fetching weather for {game_id}: {e}")
        
        return None
    
    async def get_venue_data(self, home_venue: str, away_venue: str) -> Optional[pd.DataFrame]:
        """Get venue and travel data for a matchup."""
        if 'venue' not in self.connectors:
            return None
        
        async with self.connectors['venue'] as venue:
            try:
                venue_data = await venue.fetch_and_process(
                    home_venue=home_venue, 
                    away_venue=away_venue
                )
                if venue_data is not None and not venue_data.empty:
                    logger.info(f"Fetched venue data for {home_venue} vs {away_venue}")
                    return venue_data
                    
            except Exception as e:
                logger.error(f"Error fetching venue data: {e}")
        
        return None
    
    def get_available_connectors(self) -> List[str]:
        """Get list of available/initialized connectors."""
        return list(self.connectors.keys())
    
    def get_connector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connectors."""
        status = {}
        
        for name, connector in self.connectors.items():
            status[name] = {
                "initialized": True,
                "type": type(connector).__name__,
                "rate_limit": connector.rate_limiter.min_interval,
                "last_call": connector.rate_limiter.last_call
            }
        
        # Add missing connectors
        missing_connectors = {
            'cfbd': not bool(self.settings.cfbd_api_key),
            'news': not bool(self.settings.news_feeds),
            'recruiting': not (self.settings.recruiting_247_api_key or self.settings.recruiting_rivals_api_key),
            'injury': not (self.settings.espn_injury_api_key or self.settings.fantasypros_api_key),
            'referee': not bool(self.settings.fo_api_key),
            'venue': not bool(self.settings.google_maps_api_key),
            'market': not bool(self.settings.odds_api_key)
        }
        
        for name, is_missing in missing_connectors.items():
            if is_missing and name not in status:
                status[name] = {
                    "initialized": False,
                    "reason": "API key not provided",
                    "type": f"{name}Connector"
                }
        
        return status


async def main():
    """Example usage of DataFactory."""
    from grid.config import Settings, load_config
    
    settings = Settings()
    config = load_config()
    
    factory = DataFactory(settings, config)
    
    # Show available connectors
    print("Available connectors:", factory.get_available_connectors())
    print("Connector status:", factory.get_connector_status())
    
    # Fetch all data for NFL Week 10 2024
    results = await factory.fetch_all_data("NFL", 2024, 10)
    
    for data_type, df in results.items():
        print(f"{data_type}: {len(df)} records")


if __name__ == "__main__":
    asyncio.run(main())
