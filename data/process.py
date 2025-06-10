from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np

def symmetric_log_transform(X):
    return pd.DataFrame(
        np.sign(X) * np.log1p(np.abs(X)),
        index=X.index,
        columns=X.columns
    )

def scale_data(X, scaler_class=RobustScaler):
    scaler = scaler_class()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

def process_data(df_dict):
    """1. Symmetric-Log-Scale data. 2. Scale Data (RobustScaler default) on numeric columns only."""
    processed_df_dict = {}

    for quarter in df_dict:
        original_df = df_dict[quarter]
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns

        # Transform only numeric columns
        log_transformed = original_df.copy()
        log_transformed[numeric_cols] = symmetric_log_transform(original_df[numeric_cols])
        scaled = log_transformed.copy()
        scaled[numeric_cols] = scale_data(log_transformed[numeric_cols])

        if 'ticker' in scaled.columns:
            scaled = scaled.rename(columns={'ticker': 'symbol'})

        if 'Unnamed: 0' in scaled.columns:
            scaled = scaled.drop(columns=['Unnamed: 0'])

        processed_df_dict[quarter] = scaled
    


    return processed_df_dict