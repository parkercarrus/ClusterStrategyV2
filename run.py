import logging
import pandas as pd
import config.config as config
import data.process as processing
import features.pipeline as pipeline
from features.utils import shift_quarters
from config.config import quarters_dict, params
import models.pipeline as model_pipeline
import backtest.process as backtest
import evaluate.stats as eval

def run():
    # Gather data from Config
    df_dict = config.df_dict.copy()

    # Preprocess and feature engineer
    processed_df_dict = processing.process_data(df_dict)
    engineered_data = pipeline.process_data(processed_df_dict)

    # Construct Evaluation Data
    eval_data = shift_quarters(engineered_data, shift=2)

    # Construct strategies and gather signal
    strategies = model_pipeline.process_models(engineered_data)
    signal = model_pipeline.make_predict(strategies, eval_data)
    
    # Backtest
    portfolio_value = backtest.get_portfolio_value(signal)

    max_drawdown, sharpe, cagr = eval.compute_performance_metrics(portfolio_value)
    
    return {
        "portfolio_value": portfolio_value,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "cagr": cagr
    }

res = run()
print(res)