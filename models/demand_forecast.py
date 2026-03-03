import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os

# Set up tracking
mlflow.set_experiment("Supply_Chain_Demand_Forecasting")

def create_features(df):
    """Creates time-series features using the exact column names from your CSV."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # FIX 1: Match the lowercase 'sku' and 'warehouse' from your terminal
    df = df.sort_values(['sku', 'warehouse', 'date'])

    # Time-based triggers
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # FIX 2: Lag Features using correct grouping names
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df.groupby(['sku', 'warehouse'])['demand'].shift(lag)

    return df.dropna()

def train_model():
    print("🚀 Loading data and engineering features...")
    df_raw = pd.read_csv('data/raw/demand_history.csv')
    df = create_features(df_raw)

    # Features we want the AI to learn from
    features = ['day_of_week', 'month', 'lag_1', 'lag_7', 'lag_30']
    X = df[features]
    y = df['demand']

    # Split: Keep the most recent 20% of data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    with mlflow.start_run():
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train)

        # Evaluate performance
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Log to MLflow (The 'Elite' way to track models)
        mlflow.log_metric("MAE", mae)
        mlflow.xgboost.log_model(model, "demand_xgb_model")
        
        print(f"✅ Training complete! Mean Absolute Error: {mae:.2f}")
        
        # Save the model to your models/saved folder
        os.makedirs('models/saved', exist_ok=True)
        model.save_model('models/saved/demand_forecast.json')

if __name__ == "__main__":
    train_model()