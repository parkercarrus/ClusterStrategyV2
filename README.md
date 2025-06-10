# Adaptive Clustering for Equity Selection

A machine learning framework for equity selection that combines adaptive clustering with cluster-relative feature engineering to identify outperforming S&P 500 stocks.

## Results

- **CAGR**: 32.91% | **Sharpe Ratio**: 1.32
- **Strategy**: Quarterly rebalanced top quartile selection
- **Universe**: S&P 500 stocks

## Methodology

### 1. Adaptive Clustering
Quarterly k-means clustering of stocks using fundamental metrics creates dynamic peer groups that adapt to market conditions.

### 2. Cluster-Relative Features
Instead of raw fundamentals, features measure each stock's distance from its cluster centroid and temporal changes in relative position.

### 3. Global Classification
Random Forest model predicts top quartile performers using cluster-relative features, trained across all clusters quarterly.
q

## Getting Started

```bash
# Configure strategy parameters
vim config/parameters.json

# Run the strategy
python main.py
```

**Example parameters.json:**
```json
{
  "n_clusters": 8,
    ...
  "threshold": 0.75,
}
```

## Technical Details

- **Clustering**: k-means on standardized fundamentals (P/E, P/B, ROE, etc.)
- **Features**: Cluster-relative distances and temporal changes
- **Model**: Random Forest with expanding window training
- **Rebalancing**: Quarterly with walk-forward validation

## Future Work

- Compare against cluster-specific models
- Test on international markets
- Incorporate alternative data sources
- Add transaction cost modeling

---

*Demonstrates end-to-end quantitative research: strategy development, vectorized backtesting, and performance analysis.*