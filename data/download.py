import pandas as pd
import requests
import logging
from collections import defaultdict

# Logging
logging.basicConfig(level=logging.INFO)

def get_all_symbols(constituent_dict):
    all_stocks = [stock for stocks in constituent_dict.values() for stock in stocks]
    unique_stocks = list(set(all_stocks))
    return unique_stocks

def get_symbol_data(symbol, api_key):
    """Fetch all available ratio data for a single symbol."""
    url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?period=quarterly&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.warning(f"Failed to fetch {symbol}: {e}")
        return []

def get_quarter_key(date_str, quarter_dict):
    """Map arbitrary date to quarter key like 'Q1_2022' based on release cutoff."""
    date = pd.to_datetime(date_str)
    for quarter, cutoff_str in sorted(quarter_dict.items(), key=lambda x: pd.to_datetime(x[1])):
        if date <= pd.to_datetime(cutoff_str):
            return quarter
    return None

def build_data_cache(symbols, quarter_dict, fmp_api_key):
    """Call API once per symbol and cache ratio data as {quarter -> {symbol -> features}}"""
    quarter_cache = defaultdict(dict)
    for symbol in symbols:
        print(symbol)
        raw_data = get_symbol_data(symbol, fmp_api_key)
        for entry in raw_data:
            if not isinstance(entry, dict) or 'date' not in entry:
                continue
            quarter = get_quarter_key(entry['date'], quarter_dict)
            if not quarter:
                continue
            features = {k: v for k, v in entry.items() if isinstance(v, (int, float))}
            quarter_cache[quarter][symbol] = features
    return quarter_cache

def aggregate_quarter_data(cache, constituent_dict):
    """Filter to only include requested tickers per quarter"""
    data_dict = defaultdict(dict)
    for quarter, ticker_set in constituent_dict.items():
        if quarter in cache:
            for symbol in ticker_set:
                if symbol in cache[quarter]:
                    data_dict[quarter][symbol] = cache[quarter][symbol]
    return data_dict

def get_quarterly_dataframes(constituent_dict, quarter_dict, fmp_api_key):
    """Main pipeline to produce one DataFrame per quarter."""
    all_symbols = get_all_symbols(constituent_dict)
    full_cache = build_data_cache(all_symbols, quarter_dict, fmp_api_key)
    filtered_data = aggregate_quarter_data(full_cache, constituent_dict)
    df_dict = {
        quarter: pd.DataFrame.from_dict(symbol_data, orient='index')
        for quarter, symbol_data in filtered_data.items()
    }
    return df_dict
