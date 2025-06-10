import json
import pandas as pd
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

with open('config/parameters.json', 'r') as file:
    PARAMS = json.load(file)

with open('data/objects/quarter_dict.json', 'r') as file:
    quarters_dict = json.load(file)

with open('data/objects/sp500_constituents.json', 'r') as file:
    constituent_dict = json.load(file)


market_prices = pd.read_csv('data/datasets/price_matrix.csv')

df_dict = {}
for quarter in quarters_dict:
    try:
        csv_path = f"data/datasets/{quarter}_strategy_data.csv"
        df_dict[quarter] = pd.read_csv(csv_path)
    except:
        logger.warning(f'No data available for {quarter}')

params = PARAMS
logger.info(params)