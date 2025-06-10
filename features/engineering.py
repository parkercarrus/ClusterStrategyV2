import pandas as pd
from config.config import market_prices, quarters_dict, params
from features.utils import prev_quarter
import logging

PARAMS = params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def get_price_data(symbols, quarter_id):
    prev_q = prev_quarter(quarter_id)
    prev_date = quarters_dict[prev_q]
    curr_date = quarters_dict[quarter_id]
    
    prices_prev = market_prices[market_prices['date'] == prev_date].copy()
    prices_curr = market_prices[market_prices['date'] == curr_date].copy()

    for symbol in symbols:
        if symbol not in prices_prev.columns:
            prices_prev[symbol] = 0
        if symbol not in prices_curr.columns:
            prices_curr[symbol] = 0

    df = pd.concat(
        [prices_prev[symbols], prices_curr[symbols]],
        axis=0
    )
    
    df.index = ['t-1', 't']
    df = df.fillna(0)
    return df.T




def rel_pos(X0: pd.DataFrame, X1: pd.DataFrame, centroid: pd.Series) -> pd.DataFrame:
    numeric_cols = X0.select_dtypes(include='number').columns.intersection(X1.select_dtypes(include='number').columns) # operate only on numeric columns

    rel_pos_0 = X0[numeric_cols] - centroid[numeric_cols]  # Δ position
    rel_pos_1 = X1[numeric_cols] - centroid[numeric_cols]  # new Δ position
    delta_rel_pos = rel_pos_1 - rel_pos_0                  # Δ(Δ position)

    # Rename columns
    rel_pos_0.columns = [f"{col}_relpos" for col in rel_pos_0.columns]
    delta_rel_pos.columns = [f"{col}_deltarelpos" for col in delta_rel_pos.columns]

    return pd.concat([rel_pos_0, delta_rel_pos], axis=1)

def targets(price_data: pd.DataFrame) -> pd.Series:
    if PARAMS.relative_performance == True:
        movement = price_data['t2'] - price_data['t0']
        avg_movement = movement.mean()
        return (movement - avg_movement) / avg_movement
    
    if PARAMS.relative_performance == False:
        return (price_data['t2'] - price_data['t0']) / price_data['t0']