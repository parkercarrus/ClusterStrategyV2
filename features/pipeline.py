import pandas as pd
from sklearn.cluster import KMeans
import logging
import config.config as config
import features.engineering as engineering
from typing import Dict, List
from features.utils import get_quarter_data, next_quarter
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PARAMS = config.params


class ClusterData:
    """
    Represents data for a single cluster across a specific quarter.
    
    Attributes:
        cluster_id: Unique identifier for the cluster
        quarter_id: Quarter identifier (e.g., 'Q1_2023')
        symbols: List of financial symbols in this cluster
        price_data: DataFrame containing price data for all symbols
        centroid: Series representing the centroid of the cluster
        processed_data: List to store processed data for each symbol
        processed: Boolean flag indicating if processing is complete
    """

    def __init__(self, cluster_id: int, quarter_id: str, cluster_data: Dict[str, pd.DataFrame]):
        """
        Initialize a ClusterData object.
        
        Args:
            cluster_id: Unique identifier for this cluster
            quarter_id: Quarter identifier (e.g., 'Q1_2023')
            cluster_data: DataFrame containing data for this cluster
        """
        self.cluster_data = cluster_data
        self.X0 = self.cluster_data['X0']
        self.X1 = self.cluster_data['X1']
        self.cluster_id = cluster_id
        self.quarter_id = quarter_id
        self.symbols = self._get_symbols()
        self.price_data = engineering.get_price_data(self.symbols, self.quarter_id)
        self.future_price_data = engineering.get_price_data(self.symbols, next_quarter(self.quarter_id))

        self.target_series = self._get_future_performance()
        
        self.centroid = self._get_centroid()
        self.average_performance = self._get_average_performance()
        self.processed_data = []
        self.processed = False

    def _get_symbols(self) -> List[str]:
        """Extract symbols from X0 index."""
        try:
            return list(self.X0['symbol'])
        except Exception as e:
            logger.warning(f"Unable to extract symbols: {e}")
            return []

    
    def _get_centroid(self) -> pd.Series:
        """Compute centroid from X0 data, excluding non-numeric or irrelevant columns."""
        try:
            X0_df = self.cluster_data['X0']
            ignore_cols = ['symbol', 'Unnamed: 0']
            numeric_cols = X0_df.select_dtypes(include='number').columns
            numeric_cols = [col for col in numeric_cols if col not in ignore_cols]

            centroid = X0_df[numeric_cols].mean()
            centroid.name = f'centroid_cluster_{self.cluster_id}'

            return centroid
        except Exception as e:
            logger.error(f"Failed to compute centroid for cluster {self.cluster_id}: {str(e)}")
            return pd.Series(dtype='float64')

    def _get_future_performance(self) -> pd.Series:
        if self.price_data.empty or self.future_price_data.empty:
            logger.warning('Price data series empty')
            return pd.Series(dtype='float64')

        logger.info(self.future_price_data)
        future_prices = self.future_price_data['t']
        current_prices = self.price_data['t']

        return (future_prices - current_prices) / current_prices

    def _get_average_performance(self) -> float:
        """Calculate average performance across all symbols in the cluster."""
        if self.price_data.empty:
            logger.warning(f"No price data available for performance calculation in cluster {self.cluster_id}")
            return 0.0
            
        if 't' not in self.price_data.columns or 't-1' not in self.price_data.columns:
            logger.warning(f"Missing required price columns 't' or 't-1' in cluster {self.cluster_id}")
            return 0.0
            
        returns = self.price_data['t'] - self.price_data['t-1']
        return returns.mean() if not returns.empty else 0.0

    def get_relative_performance(self) -> pd.Series:
        """Calculate performance of each symbol relative to cluster average."""
        if self.price_data.empty or 't' not in self.price_data.columns:
            logger.warning(f"Missing price data for relative performance in cluster {self.cluster_id}")
            return pd.Series(dtype='float64')
            
        prices_t = self.price_data['t']
        avg_perf = self.average_performance
        return prices_t - avg_perf
    
    def construct_data_matrix(self) -> pd.DataFrame:
        try:
            # Align X0 and X1 by symbol
            X0_df = self.X0.set_index('symbol')
            X1_df = self.X1.set_index('symbol')
            X0_df, X1_df = X0_df.align(X1_df, join='inner', axis=0)

            # Update symbols to aligned set
            aligned_symbols = X0_df.index.tolist()
            self.symbols = aligned_symbols

            # Reset index for easier handling
            X0_df = X0_df.reset_index()
            X1_df = X1_df.reset_index()

            # Get usable columns
            ignore_cols = ['symbol']
            rel_cols = [col for col in self.centroid.index if col not in ignore_cols]

            # Compute relative positions
            rel_pos_0_raw = X0_df[rel_cols].sub(self.centroid[rel_cols], axis=1)
            rel_pos_1 = X1_df[rel_cols].sub(self.centroid[rel_cols], axis=1)

            # Rename columns for clarity
            rel_pos_0_raw.columns = [f"{col}_relpos" for col in rel_cols]
            rel_pos_1.columns = [f"{col}_relpos" for col in rel_cols]

            # Subtract to get deltas (positionally safe)
            delta_vals = rel_pos_1.to_numpy() - rel_pos_0_raw.to_numpy()
            delta_rel_pos = pd.DataFrame(
                delta_vals,
                columns=[col.replace("_relpos", "_deltarelpos") for col in rel_pos_0_raw.columns],
                index=range(len(delta_vals))
            )

            
            # Get performance metrics
            relative_performance = self.get_relative_performance().reindex(self.symbols)
            target_series = self.target_series.reindex(self.symbols)

            # Core metadata
            result = pd.DataFrame({
                'symbol': self.symbols,
                'cluster_id': self.cluster_id,
                'quarter_id': self.quarter_id,
                'relative_performance': relative_performance.values,
                'target': target_series.values
            }).set_index('symbol')

            rel_pos_0_raw.index = self.symbols
            delta_rel_pos.index = self.symbols
            
            # Final combined frame
            return pd.concat([result, rel_pos_0_raw, delta_rel_pos], axis=1).reset_index()

        except Exception as e:
            logger.error(f"Error constructing matrix data for cluster {self.cluster_id}: {str(e)}")
            return pd.DataFrame()



    
    def process(self) -> None:
        """Process data for all symbols in this cluster using vectorized operations."""
        if not self.symbols:
            logger.warning(f"No symbols to process in cluster {self.cluster_id}")
            self.processed = True
            return
        
        try:
            # Process all symbols at once using matrix operations
            result_df = self.construct_data_matrix()
            if result_df.empty:
                logger.warning(f"No data generated for cluster {self.cluster_id}")
                self.processed = True
                return
                
            # Convert DataFrame to list of dictionaries for compatibility
            self.processed_data = result_df.to_dict('records')
            
            logger.info(f"Processed {len(self.processed_data)}/{len(self.symbols)} symbols in cluster {self.cluster_id} using matrix operations")
            self.processed = True
            
        except Exception as e:
            logger.error(f"Error in batch processing for cluster {self.cluster_id}: {str(e)}")
            self.processed = True
    
    def to_df(self) -> pd.DataFrame:
        """Convert processed data to DataFrame."""
        if not self.processed:
            logger.warning("Trying to convert to DataFrame without processing first")
            return pd.DataFrame()
        return pd.DataFrame(self.processed_data)

class QuarterData:
    """
    Represents and processes all data for a specific quarter.
    
    Attributes:
        quarter_df: DataFrame containing data for the quarter
        quarter_id: Identifier for the quarter (e.g., 'Q1_2023')
        symbols: Series containing symbols in this quarter
        clusters: List of ClusterData objects
    """
    
    def __init__(self, quarter_df: pd.DataFrame, quarter_id: str):
        """
        Initialize a QuarterData object.
        
        Args:
            quarter_df: DataFrame containing data for this quarter
            quarter_id: Quarter identifier
        """
        self.quarter_dict = quarter_df
        self.X0 = quarter_df['X0']
        self.X1 = quarter_df['X1']
        self.quarter_id = quarter_id
        self.symbols = self._get_symbols()
        self.clustering_object = None
        self.cluster_mapping = None
        self.future_quarter_id = next_quarter(next_quarter(self.quarter_id))
        self.clustering_method = PARAMS['clustering_method']
        self.clusters = []

    def _get_symbols(self) -> List[str]:
        """Extract symbols from X0 portion of cluster data."""
        try:
            return pd.Series(self.X0['symbol'])
        
        except:
            logger.warning("Unable to extract symbols")
            return []
    
    def train_kmeans(self) -> None:
        """Train K-means clustering model on quarter data."""
        try:
            # Select only numeric columns for clustering
            numeric_df = self.X0.select_dtypes(include='number')
                
            # Handle NaN values - either fill or drop depending on preference
            numeric_df = numeric_df.fillna(0)  # Replace with appropriate strategy
            
            model = KMeans(
                n_clusters=PARAMS['n_clusters'], 
                random_state=PARAMS['seed'], 
                n_init=PARAMS['n_init']
            )
            model.fit(numeric_df)
            self.clustering_object = model
            logger.info(f"KMeans clustering complete for quarter {self.quarter_id}")
            
        except Exception as e:
            logger.error(f"Error training KMeans for quarter {self.quarter_id}: {str(e)}")

    def map_clusters(self) -> None:
        """Map data points to clusters based on clustering method."""
        
        if self.clustering_method == 'KMeans':
            self.train_kmeans()
            if self.clustering_object is None:
                logger.error(f"Clustering failed for quarter {self.quarter_id}")
                return
                
            # Select same numeric columns used for training
            numeric_df = self.X0.select_dtypes(include='number')
            numeric_df = numeric_df.fillna(0)  # Use same strategy as in training
            
            clusters = self.clustering_object.predict(numeric_df)
            
            # Create mapping DataFrame
            self.cluster_mapping = pd.DataFrame({
                'symbol': self.symbols.values,
                'cluster': clusters
            })
            logger.info(f"Mapped {len(self.symbols)} symbols to clusters in quarter {self.quarter_id}")
            
        else:
            logger.error(f"No functionality for clustering method {self.clustering_method}")

    def get_clusters(self) -> List[ClusterData]:
        """
        Create ClusterData objects for each cluster.
        
        Returns:
            List of ClusterData objects
        """
        if self.cluster_mapping is None:
            logger.warning(f"No clusters mapped for quarter {self.quarter_id}")
            return []
            
        # Clear existing clusters to avoid duplicates
        self.clusters = []
        
        # Group by cluster
        grouped = self.cluster_mapping.groupby('cluster')

        for cluster_id, group_df in grouped:

            try:
                symbols = group_df['symbol'].values
                cluster_data = {'X0': self.X0[self.X0['symbol'].isin(symbols)].copy(), 'X1': self.X1[self.X1['symbol'].isin(symbols)].copy()}
                
                # Create cluster object
                cluster_obj = ClusterData(cluster_id, self.quarter_id, cluster_data)
                self.clusters.append(cluster_obj)
                logger.debug(f"Created cluster {cluster_id} with {len(symbols)} symbols in quarter {self.quarter_id}")
                
            except Exception as e:
                logger.error(f"Error creating cluster {cluster_id} in quarter {self.quarter_id}: {str(e)}")
        
        logger.info(f"Created {len(self.clusters)} clusters for quarter {self.quarter_id}")
        return self.clusters

    def concatenate_clusters(self) -> pd.DataFrame:
        """
        Concatenate processed data from all clusters.
        
        Returns:
            DataFrame containing combined data from all clusters
        """
        if not self.clusters:
            logger.warning(f"No clusters available to concatenate for quarter {self.quarter_id}")
            return pd.DataFrame()
            
        unprocessed = [i for i, c in enumerate(self.clusters) if not c.processed]
        if unprocessed:
            logger.warning(f"Clusters {unprocessed} not processed before concatenation in quarter {self.quarter_id}")
            
        # Filter to only processed clusters
        processed_clusters = [c for c in self.clusters if c.processed]

        if not processed_clusters:
            logger.warning(f"No processed clusters to concatenate for quarter {self.quarter_id}")
            return pd.DataFrame()
            
        # Get dataframes for each cluster
        cluster_dfs = []
        for cluster in processed_clusters:
            df = pd.DataFrame(cluster.processed_data)
            if not df.empty:
                cluster_dfs.append(df)
                
        if not cluster_dfs:
            logger.warning(f"No data to concatenate for quarter {self.quarter_id}")
            return pd.DataFrame()
            
        # Concatenate dataframes
        result = pd.concat(cluster_dfs, ignore_index=True)
        logger.info(f"Concatenated {len(processed_clusters)} clusters with {len(result)} rows total")
        return result

    def process_all_clusters(self):
        """Process all clusters for this quarter."""
        if not self.clusters:
            self.get_clusters()
            
        if not self.clusters:
            logger.warning(f"No clusters to process for quarter {self.quarter_id}")
            return
            
        for i, cluster in enumerate(self.clusters):
            try:
                logger.info(f"Processing cluster {cluster.cluster_id} ({i+1}/{len(self.clusters)}) in quarter {self.quarter_id}")
                cluster.process()
                logger.info(f"processed data {cluster.processed_data}")
            except Exception as e:
                logger.error(f"Error processing cluster {cluster.cluster_id} in quarter {self.quarter_id}: {str(e)}")

def process_quarter(quarterly_data: Dict[str, pd.DataFrame], quarter_id: str) -> pd.DataFrame:
    """
    Process data for a specific quarter.
    
    Args:
        quarterly_data: Dictionary mapping quarter IDs to DataFrames
        quarter_id: Quarter ID to process
        
    Returns:
        DataFrame containing processed quarter data
    """
    try:
        logger.info(f"Processing quarter {quarter_id}")
        
        # Get quarter data
        quarter_data = get_quarter_data(quarterly_data, quarter_id)
        if quarter_data is None:
            logger.warning(f"No data available for quarter {quarter_id}")
            return pd.DataFrame()
        
        q_next = next_quarter(quarter_id)
        q_next2 = next_quarter(q_next)
        

        qd = QuarterData(quarter_data, quarter_id)
        qd.future_quarter_id = q_next2

        qd.map_clusters()
        qd.get_clusters()
        qd.process_all_clusters()
        
        # Combine results
        result = qd.concatenate_clusters()
        logger.info(f"Completed processing quarter {quarter_id}: generated {len(result)} rows of data")
        return result
        
    except Exception as e:
        logger.error(f"Error processing quarter {quarter_id}: {str(e)}")
        return pd.DataFrame()

def process_data(quarterly_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Process all quarters in the provided data.
    
    Args:
        quarterly_data: Dictionary mapping quarter IDs to DataFrames
        
    Returns:
        Dictionary mapping quarter IDs to processed DataFrames
    """
    if not quarterly_data:
        logger.warning("No quarterly data provided")
        return {}
        
    keys = list(quarterly_data.keys())
    
    if len(keys) <= 1:
        logger.warning("Need at least two quarters to process data (for X0 and X1)")
        return {}
    
    logger.info(f"Processing {len(keys)-1} quarters of data")
    
    # Process each quarter except the last two (which we need for target values)
    results = {}
    for i, q_id in enumerate(keys[:-1]):
        logger.info(f"Processing quarter {i+1}/{len(keys)-2}: {q_id}")
        results[q_id] = process_quarter(quarterly_data, q_id)
        
    logger.info(f"Completed processing {len(results)} quarters")
    return results