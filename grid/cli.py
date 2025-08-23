"""Command line interface for Grid Football Prediction System."""

import click
import asyncio
from pathlib import Path
from datetime import datetime

from grid.config import load_config, get_settings
from grid.data.storage import DataStorage
from grid.data.connectors import CFBDConnector, NFLConnector, NewsConnector
from grid.models.elo import EloEnsemble
from grid.api.main import GridAPI
from loguru import logger


@click.group()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Grid Football Prediction System CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the Grid system."""
    config_path = ctx.obj['config_path']
    
    try:
        config = load_config(config_path)
        storage = DataStorage(config)
        
        click.echo("Initializing Grid Football Prediction System...")
        
        # Create data structure
        from grid.config import ensure_data_structure, get_data_dir
        data_dir = get_data_dir(config)
        ensure_data_structure(data_dir)
        
        click.echo(f"Data directories created at {data_dir}")
        
        # Initialize Elo ratings
        elo_ensemble = EloEnsemble(config)
        
        click.echo("System initialized successfully")
        click.echo(f"API will run on http://localhost:{config.app.port}")
        
    except Exception as e:
        click.echo(f"Initialization failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--sources', multiple=True, help='Data sources to update')
@click.option('--league', help='Specific league (NFL/NCAA)')
@click.option('--season', type=int, help='Specific season')
@click.option('--week', type=int, help='Specific week')
@click.pass_context
def update(ctx, sources, league, season, week):
    """Update data from external sources."""
    config_path = ctx.obj['config_path']
    
    try:
        config = load_config(config_path)
        settings = get_settings()
        storage = DataStorage(config)
        
        click.echo("Updating data...")
        
        # Run async update
        asyncio.run(_update_data(config, settings, storage, sources, league, season, week))
        
        click.echo("Data update completed")
        
    except Exception as e:
        click.echo(f"Update failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--league', help='Specific league (NFL/NCAA)')
@click.option('--season', type=int, help='Specific season')
@click.option('--week', type=int, help='Specific week')
@click.pass_context
def predict(ctx, league, season, week):
    """Generate predictions."""
    config_path = ctx.obj['config_path']
    
    try:
        config = load_config(config_path)
        storage = DataStorage(config)
        
        click.echo("Generating predictions...")
        
        # Load data
        games_df = storage.load_data("games")
        teams_df = storage.load_data("teams")
        
        if games_df.empty:
            click.echo("No games data found. Run 'grid update' first.")
            return
        
        # Filter by parameters
        if league:
            games_df = games_df[games_df["league"] == league]
        if season:
            games_df = games_df[games_df["season"] == season]
        if week is not None:
            games_df = games_df[games_df["week"] == week]
        
        if games_df.empty:
            click.echo("No games found matching criteria.")
            return
        
        # Initialize Elo and generate predictions
        elo_ensemble = EloEnsemble(config)
        elo_ensemble.initialize_all_ratings(teams_df)
        
        predictions_df = elo_ensemble.predict_games(games_df)
        
        # Save predictions
        predictions_path = storage.data_dir / "predictions" / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions_path.parent.mkdir(exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        
        click.echo(f" Generated {len(predictions_df)} predictions")
        click.echo(f" Saved to {predictions_path}")
        
        # Show sample
        if len(predictions_df) > 0:
            click.echo("\nSample predictions:")
            for _, pred in predictions_df.head().iterrows():
                click.echo(f"  {pred['home_team']} vs {pred['away_team']}: {pred['home_win_prob']:.3f}")
        
    except Exception as e:
        click.echo(f"Error: Prediction failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', type=int, help='Port to bind to')
@click.pass_context
def serve(ctx, host, port):
    """Start the API server."""
    config_path = ctx.obj['config_path']
    
    try:
        click.echo("Starting Grid Football Prediction API...")
        
        api = GridAPI(config_path)
        api.run(host=host, port=port)
        
    except Exception as e:
        click.echo(f"Error: Server failed to start: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status."""
    config_path = ctx.obj['config_path']
    
    try:
        config = load_config(config_path)
        storage = DataStorage(config)
        
        click.echo("Grid Football Prediction System Status")
        click.echo("=" * 40)
        
        # Data status
        load_history = storage.get_load_history()
        if not load_history.empty:
            latest_loads = load_history.groupby("table_name")["load_timestamp"].max()
            click.echo("\n Data Freshness:")
            for table, timestamp in latest_loads.items():
                click.echo(f"  {table}: {timestamp}")
        else:
            click.echo("\n No data loaded yet")
        
        # Feature versions
        feature_versions = storage.get_feature_versions()
        if not feature_versions.empty:
            click.echo(f"\n🔧 Feature Versions: {len(feature_versions)}")
            for _, version in feature_versions.head().iterrows():
                click.echo(f"  {version['feature_version_id']}: {version['status']}")
        
        # Configuration
        click.echo(f"\n  Configuration:")
        click.echo(f"  Data dir: {config.app.data_dir}")
        click.echo(f"  Leagues: {', '.join(config.app.leagues)}")
        click.echo(f"  Port: {config.app.port}")
        
    except Exception as e:
        click.echo(f"Error: Status check failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--season', type=int, required=True, help='Season to backtest')
@click.option('--league', help='Specific league (NFL/NCAA)')
@click.pass_context
def backtest(ctx, season, league):
    """Run backtesting analysis."""
    config_path = ctx.obj['config_path']
    
    try:
        config = load_config(config_path)
        storage = DataStorage(config)
        
        click.echo(f"Running backtest for {season}...")
        
        # Load historical data
        games_df = storage.load_data("games")
        teams_df = storage.load_data("teams")
        
        if league:
            games_df = games_df[games_df["league"] == league]
            teams_df = teams_df[teams_df["league"] == league]
        
        season_games = games_df[games_df["season"] == season]
        
        if season_games.empty:
            click.echo(f"Error: No games found for {season}")
            return
        
        # Run Elo backtest
        elo_ensemble = EloEnsemble(config)
        elo_ensemble.initialize_all_ratings(teams_df)
        
        predictions_df = elo_ensemble.predict_games(season_games)
        
        # Calculate accuracy (simplified)
        completed_games = predictions_df.merge(
            season_games[["game_id", "home_score", "away_score"]],
            on="game_id"
        )
        completed_games = completed_games.dropna(subset=["home_score", "away_score"])
        
        if not completed_games.empty:
            completed_games["home_win"] = completed_games["home_score"] > completed_games["away_score"]
            completed_games["predicted_home_win"] = completed_games["home_win_prob"] > 0.5
            accuracy = (completed_games["home_win"] == completed_games["predicted_home_win"]).mean()
            
            click.echo(f" Backtest completed")
            click.echo(f" Accuracy: {accuracy:.3f}")
            click.echo(f" Games analyzed: {len(completed_games)}")
        else:
            click.echo("Error: No completed games found for analysis")
        
    except Exception as e:
        click.echo(f"Error: Backtest failed: {e}", err=True)
        raise click.Abort()


async def _update_data(config, settings, storage, sources, league, season, week):
    """Async helper for data updates."""
    
    if not sources:
        sources = ["cfbd", "nfl", "news"]
    
    for source in sources:
        try:
            click.echo(f"Updating {source}...")
            
            if source == "cfbd" and settings.cfbd_api_key:
                async with CFBDConnector(settings) as connector:
                    if season:
                        teams_df = await connector.get_teams(season)
                        if not teams_df.empty:
                            storage.save_silver_data(teams_df, "teams")
                        
                        games_df = await connector.get_games(season, week)
                        if not games_df.empty:
                            storage.save_silver_data(games_df, "games")
            
            elif source == "nfl":
                async with NFLConnector(settings) as connector:
                    if season:
                        teams_df = await connector.get_teams(season)
                        if not teams_df.empty:
                            storage.save_silver_data(teams_df, "teams")
                        
                        games_df = await connector.get_games(season, week)
                        if not games_df.empty:
                            storage.save_silver_data(games_df, "games")
            
            elif source == "news":
                async with NewsConnector(settings) as connector:
                    news_df = await connector.get_all_news()
                    if not news_df.empty:
                        storage.save_silver_data(news_df, "news")
            
            click.echo(f"   {source} updated")
            
        except Exception as e:
            click.echo(f"  Error: {source} failed: {e}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
