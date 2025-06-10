import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config.config import params

def train_model(data: pd.DataFrame) -> RandomForestClassifier:
    clf = RandomForestClassifier()

    # Create binary target: 1 if in top 25%, else 0
    threshold = data["target"].quantile((1-params['threshold']))
    y_train = (data["target"] >= threshold).astype(int)
    
    X_train = data.drop(columns=['target', 'index'])
    
    clf.fit(X_train, y_train)

    return clf