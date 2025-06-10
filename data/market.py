import requests
import pandas as pd

API_KEY = "1a9b1b90e9207a9f2bc82d39e9bd5a6f" 

def download(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download adjusted close prices from FinancialModelingPrep"""
    all_data = []

    for symbol in symbols:
        print(f'Trying {symbol}')
        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            f"?from={start.date()}&to={end.date()}&apikey={API_KEY}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed for {symbol}")
            continue

        json_data = response.json()
        if "historical" not in json_data:
            print(f"No historical data for {symbol}")
            continue

        df = pd.DataFrame(json_data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        df = df[["adjClose"]].rename(columns={"adjClose": symbol})
        all_data.append(df)

    if not all_data:
        raise ValueError("No data downloaded.")
        
    return pd.concat(all_data, axis=1)

def get(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Wrapper to get filtered time-windowed adjusted close prices"""
    print(symbols)
    return download(symbols, start, end)
