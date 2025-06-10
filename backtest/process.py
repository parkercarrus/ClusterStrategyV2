from config.config import constituent_dict, market_prices, quarters_dict, params
import pandas as pd
import logging

logger = logging.getLogger()


def check_signal(signal_matrix: pd.DataFrame) -> bool:
    signal_valid = True

    # Check if system is purchasing non-constituent equities
    count = 0
    for quarter in signal_matrix.iterrows():
        quarter_consts = constituent_dict[quarter[0]]
        for equity in quarter[1].index:
            if quarter[1][equity] == 1:
                if equity not in quarter_consts:
                    count+=1
    
    if count != 0:
        logger.warning('BACKTEST INVALID - SYSTEM PURCHASING NON-CONSTITUENT EQUITIES')
        signal_valid = False
    
    return signal_valid

def get_price_matrix(signal_matrix: pd.DataFrame) -> pd.DataFrame:
    price_matrix = market_prices.copy()

    # Convert market_prices index to Datetime
    price_matrix['date'] = pd.to_datetime(price_matrix['date'])
    price_matrix = price_matrix.drop(columns=['index', 'Unnamed: 0'], errors='ignore')
    price_matrix = price_matrix.set_index('date')

    missing_cols = sum([col not in signal_matrix for col in price_matrix])
    logger.info(f"There are {missing_cols} columns present in market_prices but not present in signal")

    # Drop Irrelevant Columns
    cols = []
    for col in price_matrix:
        if col not in signal_matrix.columns:
            cols.append(col)

    price_matrix_cleaned = price_matrix.drop(columns=cols)

    return price_matrix_cleaned

def process_signal(signal_matrix: pd.DataFrame) -> pd.DataFrame:
    """Returns a dated version of signal_matrix"""
    # Convert signal_matrix index to Datetime
    quarter_dates = [pd.to_datetime(quarters_dict[q]) for q in signal_matrix.index]
    signal_dated = signal_matrix.copy()
    signal_dated.index = pd.DatetimeIndex(quarter_dates, name='date')
    
    return signal_dated

def filter_price_matrix(price_matrix: pd.DataFrame, signal_matrix: pd.DataFrame) -> pd.DataFrame:
    price_matrix_filtered = price_matrix.loc[signal_matrix.index.min():]
    return price_matrix_filtered

def pad_signal(signal_matrix: pd.DataFrame, price_matrix: pd.DataFrame) -> pd.DataFrame:
    # Print each date between the first and last index in signal_dated
    for date in pd.date_range(signal_matrix.index.min() + pd.Timedelta(days=1), price_matrix.index.max(), freq='D'):
        if date not in signal_matrix.index and date in price_matrix.index:
            # Add a row with all zero values for the columns in signal_dated at this date
            signal_matrix.loc[date] = 0
            signal_matrix = signal_matrix.sort_index()
    
    if signal_matrix.shape != price_matrix.shape:
        logger.warning(f'signal_matrix and price_matrix not aligned {signal_matrix.shape} vs {price_matrix.shape}')
    else:
        assert signal_matrix.shape == price_matrix.shape

    return signal_matrix

def scale_buy_signal(stretched_signal, price_data, purchase_amount):
    mask = stretched_signal == 1
    scaled_signal = stretched_signal.copy().astype(float)
    scaled_signal[mask] = purchase_amount / price_data[mask]
    return scaled_signal


def get_portfolio_value(signal_matrix: pd.DataFrame):
    """
    Calculates the portfolio value over time based on trading signals and corresponding price data.
    Args:
        signal_matrix (pd.DataFrame): A DataFrame containing trading signals, where rows represent time indices and columns represent different assets.
    Returns:
        pd.Series: A Series representing the total portfolio value at each time index. Returns an empty DataFrame if the signal_matrix fails validation.
    """
    if not check_signal(signal_matrix):
        return pd.DataFrame()
    
    price_matrix = get_price_matrix(signal_matrix)
    processed_signal = process_signal(signal_matrix)
    price_matrix_filtered = filter_price_matrix(price_matrix, processed_signal)
    
    signal_padded = pad_signal(processed_signal, price_matrix_filtered)
    signal_scaled = scale_buy_signal(signal_padded, price_matrix, params['buy_amount'])

    signal_cumsum = signal_scaled.cumsum()
    ledger = signal_cumsum * price_matrix_filtered
    portfolio_value = ledger.sum(axis=1)

    return portfolio_value

