from config.config import market_prices
import pandas as pd
from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_date_range(quarter):
    return

def get_price_data(symbols, quarter):
    """Return a DataFrame of symbols index, with price t-1, t, t+1"""
    start_date, end_date = get_date_range(quarter)
    return market_prices[symbols, start_date, end_date]

def next_quarter(quarter_id: str) -> str:
    q, year = quarter_id.split('_')
    quarter_num = int(q[1])
    year = int(year)
    if quarter_num == 4:
        return f"Q1_{year + 1}"
    else:
        return f"Q{quarter_num + 1}_{year}"
    
def prev_quarter(quarter_id: str) -> str:
    q, year = quarter_id.split('_')
    quarter_num = int(q[1])
    year = int(year)
    if quarter_num == 1:
        return f"Q4_{year - 1}"
    else:
        return f"Q{quarter_num - 1}_{year}"

    
def get_quarter_data_OLD(quarterly_data: Dict[str, pd.DataFrame], quarter_id: str) -> Dict[str, pd.DataFrame]:
    """
    Get data for a specific quarter and the next quarter.
    
    Args:
        quarterly_data: Dictionary mapping quarter IDs to DataFrames
        quarter_id: Current quarter ID
        
    Returns:
        Dictionary with current (X0) and next quarter (X1) data
    """
    next_q = next_quarter(quarter_id)
    
    if quarter_id not in quarterly_data:
        logger.error(f"Quarter {quarter_id} not found in quarterly data")
        return {}
        
    if next_q not in quarterly_data:
        logger.error(f"Next quarter {next_q} not found in quarterly data")
        return {}
    
    return quarterly_data[quarter_id]

def get_quarter_data(data_dict, quarter_id):
    return {
        'X0': data_dict[quarter_id],
        'X1': data_dict[next_quarter(quarter_id)]
    }

def shift_quarters(quarterly_data: Dict[str, pd.DataFrame], shift: int = 2) -> Dict[str, pd.DataFrame]:
    keys = list(quarterly_data.keys())
    shifted_eval_data = {
        keys[i]: quarterly_data[keys[i - shift]] if i - shift >= 0 else quarterly_data[keys[0]]
        for i in range(len(keys))
    }
    return shifted_eval_data