import config.config as config
from typing import Dict
import pandas as pd

constituent_dict = config.constituent_dict

def get_viable_symbols(quarter_id: str) -> list[str]:
    try:
        return constituent_dict[quarter_id]
    except:
        raise KeyError(f'Trading universe constituents not found for {quarter_id}')
    

def prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    # Drop Columns
    X = X.drop(
        columns=['cluster_id', 'quarter_id', 'Unnamed: 0_relpos', 'ticker_relpos', 'ticker_deltarelpos', 'Unnamed: 0_deltarelpos'],
        errors='ignore'
    )

    X = X.fillna(0)

    return X

def construct_decision_matrix(signal_list: list[pd.DataFrame], idx_list: list[str]) -> pd.DataFrame:
    symbols = set()
    for s in signal_list:
        symbols.update(idx for idx in s.index if not str(idx).isdigit())
    symbols_idx = list(symbols)

    # Create a DataFrame for each (sig, id) pair with symbols_idx as columns
    df = pd.DataFrame(index=idx_list, columns=symbols_idx)

    for (sig, id) in zip(signal_list, idx_list):
        for symbol in symbols_idx:
            if symbol in sig.index:
                df.at[id, symbol] = sig[symbol]
            else:
                df.at[id, symbol] = None

    # Fill NA Values with 0
    df = df.fillna(0)

    return df
