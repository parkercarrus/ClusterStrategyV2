import models.long as long
import pandas as pd
from config.config import params
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import defaultdict
from models.utils import get_viable_symbols, prepare_X, construct_decision_matrix
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)

# IMPLEMENTATION OF LONG-ONLY STRATEGY

class Strategy:
    def __init__(self, training_data: pd.DataFrame, quarter_id: str):
        self.training_data = training_data
        self.quarter_id = quarter_id
        self.model_metrics = None
        self.model_object = RandomForestClassifier()
        self.probas = pd.Series(dtype=float)
        self.viable_symbols = get_viable_symbols(self.quarter_id)

    def fit(self):
        self.model_object = long.train_model(self.training_data)
    
    def rank(self, X: pd.DataFrame):
        X_eval = X[self.model_object.feature_names_in_]
        probas = self.model_object.predict_proba(X_eval)

        return pd.DataFrame(probas)

    def f(self, X: pd.DataFrame) -> pd.Series:
        symbol_idx = X['index']
        
        # Predict probabilities using the model
        self.probas = self.rank(X)
        self.probas.index = symbol_idx.values  # Set symbol index for probabilities
        
        # Zero out probabilities for non-viable symbols
        self.probas.loc[~self.probas.index.isin(self.viable_symbols)] = 0

        # Pick top-N based on probability
        top_n = self.probas.nlargest(params['num_stocks'], columns=1)

        # Create binary signals (1 if in top_n, else 0), indexed by symbol
        binary_signals = pd.Series(
            (symbol_idx.isin(top_n.index)).astype(int).values,
            index=symbol_idx
        )

        return binary_signals
    
    def fill_data(self):
        self.training_data = self.training_data.fillna(0).infer_objects(copy=False)

    def process(self):
        self.training_data = prepare_X(self.training_data)
        self.fit()

def process_single_strategy(args: Tuple[str, pd.DataFrame]) -> Strategy:

    quarter_id, data = args
    strategy = Strategy(data, quarter_id)
    strategy.process()
    return strategy

def process_models(quarterly_data: Dict[str, pd.DataFrame]) -> List[Strategy]:
    if params['parallel']:
        # Execute quarterly processing in parallel
        with ThreadPoolExecutor() as executor:
            strategies = list(executor.map(process_single_strategy, quarterly_data.items()))
        return strategies
    
    else:
        return [process_single_strategy(args) for args in quarterly_data.items()]
    
def make_predict(strategies: List[Strategy], eval_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    for i, (strategy, (q_id, _)) in enumerate(zip(strategies, eval_data.items())):
        if strategy.quarter_id != q_id:
            logger.error(f"Mismatch at index {i}: strategy.quarter_id = {strategy.quarter_id}, expected = {q_id}")
            raise ValueError(f"Quarter ID mismatch at index {i}")

    
    signal_rows = []
    idx_list = []

    eval_data = {q_id: prepare_X(X) for q_id, X in eval_data.items()}

    for (strategy, item) in zip(strategies, eval_data.items()):
        q_id, X = item
        signal = strategy.f(X)
        signal_rows.append(signal)
        idx_list.append(q_id)

    df = construct_decision_matrix(signal_rows, idx_list)
    
    return df