# MarketSense-Lumina

A real-time risk management platform that uses market-implied signals and game-theoretic modeling to predict systemic financial risk.

## Overview

MarketSense-Lumina replaces traditional Monte Carlo simulations with machine learning models to provide instant Minimum Credit Requirement (MCR) calculations and systemic resilience scoring. The platform uses real-time market data (stock prices, volatility) rather than delayed self-reported bank data to assess financial institution health and systemic risk.

## Features

- **Real-time Market Data Ingestion**: Live stock data processing with rolling volatility calculations
- **Game-Theoretic Simulation**: Bank agent modeling with autonomous decision-making capabilities
- **Machine Learning Intelligence**: CoreML-powered MCR prediction and anomaly detection
- **Contagion Analysis**: Hub-and-spoke network modeling for shock propagation
- **Interactive Dashboard**: Real-time risk visualization and scenario analysis
- **RESTful API**: Comprehensive API layer for system integration

## Architecture

The system follows a four-phase implementation approach:

1. **Phase I**: Market Oracle and Data Engineering
2. **Phase II**: Game-Theoretic Engine
3. **Phase III**: Machine Learning and CoreML Integration
4. **Phase IV**: Contagion Analysis and Dashboard

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/marketsense/lumina.git
cd lumina
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Configuration

1. Copy the example configuration:
```bash
cp config/config.example.json config/config.json
```

2. Set up environment variables:
```bash
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
export QUANDL_API_KEY="your_api_key_here"
export FRED_API_KEY="your_api_key_here"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests
pytest tests/property/ -m property

# Run with coverage
pytest --cov=market_sense_lumina
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

Install development dependencies:
```bash
pip install -e ".[dev]"
pre-commit install
```

### Testing Strategy

The project uses a dual testing approach:

- **Unit Tests**: Specific examples and edge cases
- **Property-Based Tests**: Universal properties using Hypothesis framework

Each property test runs a minimum of 100 iterations to ensure statistical confidence.

## Usage

### Basic Example

```python
from market_sense_lumina import MarketOracle, GameTheoreticEngine

# Initialize market data oracle
oracle = MarketOracle()
data = oracle.fetch_historical_data(['JPM', 'BAC', 'WFC'], period='1y')

# Set up game-theoretic simulation
engine = GameTheoreticEngine()
mcr_value = engine.calculate_mcr_realtime(data)

print(f"Current MCR: ${mcr_value:,.2f}")
```

## API Documentation

The RESTful API provides endpoints for:

- Market data retrieval
- Risk calculations
- Real-time alerts
- Historical analysis

API documentation is available at `/docs` when running the server.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the MarketSense team at team@marketsense.ai.