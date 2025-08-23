# Grid Football Prediction System

Advanced AI-powered football analytics and prediction system for NFL and NCAA, designed for superfans and analysts.

## Features

- Multi-League Support: NFL and NCAA FBS predictions
- Advanced Modeling: Elo baseline, GBDT, Deep Learning tabular models
- Player-Level Analytics: EPA, matchups, aging curves, development tracking
- Special Teams Analysis: Field goal modeling, return efficiency, ST-EPA
- Real-Time Predictions: Live game updates and in-game probability changes
- Uncertainty Quantification: Conformal prediction intervals and calibration
- Explainable AI: SHAP explanations and counterfactual analysis
- Production Ready: FastAPI, data validation, automated pipelines

## System Architecture

```
Data Plane:    Connectors → Normalizers → Feature Store → Snapshots
Model Plane:   Elo/SRS → DL Tabular → Ensemble → Calibration
Serving Plane: FastAPI → ONNX Runtime → Explanations → UI
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd FootballPredictionSystem

# Install dependencies
pip install -e .

# Set up configuration
cp .env.template .env
# Edit .env with your API keys
```

### Initialize System

```bash
# Initialize data directories and system
grid init

# Update data sources
grid update --sources cfbd nfl news

# Generate predictions
grid predict --league NFL --season 2024 --week 10

# Start API server
grid serve
```

### API Usage

```bash
# Health check
curl http://localhost:8787/health

# Get game prediction
curl http://localhost:8787/predict/game/NFL_2024_W10_BUF_MIA

# Get week predictions
curl http://localhost:8787/predict/week/NFL/2024/10

# Get explanations
curl http://localhost:8787/explain/game/NFL_2024_W10_BUF_MIA

# Counterfactual analysis
curl -X POST http://localhost:8787/predict/counterfactual \
  -H "Content-Type: application/json" \
  -d '{"game_id": "NFL_2024_W10_BUF_MIA", "deltas": {"QB_out": true, "wind": 15}}'
```

## Configuration

The system uses `config.yaml` for configuration:

```yaml
app:
  port: 8787
  data_dir: ./data
  leagues: [NFL, NCAA]

features:
  enable_player_level: true
  enable_special_teams: true
  enable_garbage_time_filter: true
  
models:
  use_text: true
  pbp_transformer:
    enabled: true
    d_model: 256
```

## Data Sources

### Required API Keys

- **CFBD_API_KEY**: College Football Data API for NCAA data
- **WEATHER_API_KEY**: Weather service for game conditions
- **NEWS_FEEDS**: RSS feeds for news analysis

### Supported Sources

- **NFL**: ESPN API for games, scores, basic stats
- **NCAA**: College Football Data API for comprehensive FBS data
- **Weather**: Game-day conditions for field goal modeling
- **News**: RSS feeds for injury reports and team news
- **Recruiting**: Team talent and transfer portal data

## Core Features

### Garbage Time Detection

Configurable thresholds by league and quarter:
- **NFL**: Q1/Q2: 28 points, Q3: 21 points, Q4: 17 points
- **NCAA**: Q1/Q2: 35 points, Q3: 28 points, Q4: 24 points

### Player Features

- **EPA by involvement**: QB dropbacks, RB carries/targets, WR/TE targets
- **Matchup analysis**: WR vs CB, OL vs DL strength comparisons
- **Development curves**: Position-specific aging and experience factors
- **Injury impact**: Automated injury report processing

### Special Teams

- **Field Goal Model**: Distance, weather, venue, kicker factors
- **Return Efficiency**: Expected vs actual return yards
- **ST-EPA**: Comprehensive special teams expected points added

### Models

1. **Elo Baseline**: Rating system with home field, recency decay
2. **GBDT**: XGBoost/LightGBM with feature importance
3. **Deep Tabular**: Multi-task neural network with uncertainty
4. **Ensemble**: Weighted combination with calibration

## API Endpoints

### Predictions
- `GET /predict/game/{game_id}` - Single game prediction
- `GET /predict/week/{league}/{season}/{week}` - Week predictions
- `POST /predict/counterfactual` - Counterfactual analysis

### Analytics
- `GET /teams/{team_id}` - Team information
- `GET /players/{player_id}/projections` - Player projections
- `GET /specialteams/summary` - Special teams analysis
- `GET /analytics/talent/{team_id}` - Recruiting/talent metrics

### System
- `GET /health` - Health check
- `GET /status` - System status
- `POST /update` - Trigger data updates
- `GET /calibration/weekly` - Model calibration metrics

## CLI Commands

```bash
# System management
grid init                    # Initialize system
grid status                  # Show system status
grid serve                   # Start API server

# Data operations
grid update                  # Update all data sources
grid update --sources cfbd   # Update specific source
grid update --league NFL     # Update specific league

# Predictions
grid predict                 # Generate all predictions
grid predict --league NCAA   # League-specific predictions
grid predict --season 2024   # Season-specific predictions

# Analysis
grid backtest --season 2023  # Run backtesting analysis
```

## Development

### Project Structure

```
grid/
├── api/              # FastAPI routers and schemas
├── data/             # Connectors, validators, storage
├── features/         # Feature engineering modules
├── models/           # Model implementations
├── utils/            # Utility functions
└── cli.py            # Command line interface
```

### Data Flow

1. **Raw Data**: JSON files from external APIs
2. **Bronze Layer**: Validated Parquet files
3. **Silver Layer**: Normalized schema-compliant data
4. **Gold Layer**: Feature-engineered data with versioning
5. **Snapshots**: Reproducible weekly prediction packages

### Adding New Features

1. Create feature builder in `grid/features/`
2. Add schema validation in `grid/data/schemas.py`
3. Update feature pipeline in main processing
4. Add unit tests in `tests/`

## Model Performance

### Promotion Criteria (POC)

- **Accuracy**: Beat Elo baseline by ≥2% LogLoss
- **Calibration**: ECE ≤ 0.03
- **Coverage**: 90% conformal prediction within 88-92%
- **Speed**: Full weekly run < 30 seconds on 8-core CPU

### Monitoring

- Automated calibration tracking
- Performance drift detection
- A/B testing framework for features/models
- Model lineage and reproducibility

## Production Deployment

### Requirements

- **Python**: 3.11+
- **CPU**: 8+ cores recommended
- **Memory**: 16GB+ for full dataset
- **Storage**: 10GB+ for historical data

### Scaling

- Containerized with Docker
- Horizontal scaling via load balancer
- Separate prediction and serving workloads
- Caching layer for frequently accessed data

## Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues for bugs and features
- Documentation at `/docs` when API is running
- Performance monitoring at `/status` endpoint