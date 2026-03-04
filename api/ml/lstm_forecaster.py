"""
LSTM-based Demand Forecasting Service
Complete implementation for supply chain optimization
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow scikit-learn")


class LSTMForecaster:
    """
    LSTM Neural Network for time-series demand forecasting
    
    Features:
    - Multi-step ahead forecasting
    - Confidence interval estimation
    - Model persistence (save/load)
    - Early stopping and learning rate scheduling
    - Evaluation metrics (MAPE, RMSE, MAE)
    """
    
    def __init__(self, sequence_length: int = 30, features: int = 1):
        """
        Initialize LSTM Forecaster
        
        Args:
            sequence_length: Number of past timesteps to use for prediction
            features: Number of input features (1 for univariate)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
            
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def build_model(
        self, 
        lstm_units: List[int] = [64, 32], 
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Build LSTM model architecture
        
        Args:
            lstm_units: List of units for each LSTM layer
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        model = keras.Sequential([
            # First LSTM layer
            keras.layers.LSTM(
                lstm_units[0],
                return_sequences=True,
                input_shape=(self.sequence_length, self.features)
            ),
            keras.layers.Dropout(dropout),
            
            # Second LSTM layer
            keras.layers.LSTM(lstm_units[1], return_sequences=False),
            keras.layers.Dropout(dropout),
            
            # Dense layers
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def prepare_data(
        self, 
        data: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for LSTM training
        
        Args:
            data: Time series data (pandas Series)
            
        Returns:
            X, y: Training sequences and targets
        """
        # Normalize data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        train_data: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model
        
        Args:
            train_data: Training time series data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X, y = self.prepare_data(train_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=verbose
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        self.history = history.history
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }
    
    def predict(
        self, 
        historical_data: pd.Series, 
        forecast_horizon: int = 30,
        return_intervals: bool = True
    ) -> Dict:
        """
        Generate demand forecast
        
        Args:
            historical_data: Historical demand data
            forecast_horizon: Number of periods to forecast
            return_intervals: Whether to return confidence intervals
            
        Returns:
            Dictionary with predictions and optional confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare last sequence for prediction
        scaled_data = self.scaler.transform(historical_data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        # Generate multi-step forecast
        for _ in range(forecast_horizon):
            # Reshape for model input
            input_seq = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            pred = self.model.predict(input_seq, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence (sliding window)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        result = {
            'predictions': predictions.tolist(),
            'forecast_horizon': forecast_horizon,
            'model': 'LSTM'
        }
        
        # Calculate confidence intervals
        if return_intervals:
            std_dev = np.std(historical_data.values)
            lower_bound = predictions - 1.96 * std_dev
            upper_bound = predictions + 1.96 * std_dev
            
            result['lower_bound'] = lower_bound.tolist()
            result['upper_bound'] = upper_bound.tolist()
            result['confidence_level'] = 0.95
        
        return result
    
    def evaluate(self, test_data: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            test_data: Test time series data
            
        Returns:
            Evaluation metrics (MSE, RMSE, MAE, MAPE)
        """
        X, y = self.prepare_data(test_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform
        y_actual = self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        y_pred = self.scaler.inverse_transform(predictions).flatten()
        
        # Calculate metrics
        mse = np.mean((y_actual - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_actual - y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-10))) * 100
        
        # R-squared
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: str, scaler_path: str):
        """Load model and scaler from disk"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM Demand Forecasting - Demo")
    print("=" * 60)
    
    # Generate sample demand data
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    
    # Simulate realistic demand with trend, seasonality, and noise
    t = np.arange(365)
    trend = 100 + 0.3 * t
    seasonality = 30 * np.sin(2 * np.pi * t / 365)
    weekly_pattern = 15 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 10, 365)
    
    demand = trend + seasonality + weekly_pattern + noise
    demand_series = pd.Series(demand, index=dates)
    
    print(f"\n📊 Sample Data Generated:")
    print(f"   - Time period: {dates[0].date()} to {dates[-1].date()}")
    print(f"   - Data points: {len(demand_series)}")
    print(f"   - Mean demand: {demand_series.mean():.2f} units")
    print(f"   - Std deviation: {demand_series.std():.2f} units")
    
    # Split into train and test
    train_size = int(len(demand_series) * 0.8)
    train_data = demand_series[:train_size]
    test_data = demand_series[train_size:]
    
    print(f"\n📈 Training LSTM Model...")
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Test samples: {len(test_data)}")
    
    # Initialize and train model
    forecaster = LSTMForecaster(sequence_length=30)
    
    history = forecaster.train(
        train_data,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating Model Performance...")
    metrics = forecaster.evaluate(test_data)
    
    print(f"\n✅ Test Results:")
    print(f"   - RMSE: {metrics['rmse']:.2f} units")
    print(f"   - MAE: {metrics['mae']:.2f} units")
    print(f"   - MAPE: {metrics['mape']:.2f}%")
    print(f"   - R²: {metrics['r2']:.4f}")
    
    # Generate forecast
    print(f"\n🔮 Generating 30-day Forecast...")
    forecast = forecaster.predict(train_data, forecast_horizon=30)
    
    print(f"\n📅 Forecast Preview (first 7 days):")
    for i in range(7):
        pred = forecast['predictions'][i]
        lower = forecast['lower_bound'][i]
        upper = forecast['upper_bound'][i]
        print(f"   Day {i+1}: {pred:.1f} units (95% CI: {lower:.1f} - {upper:.1f})")
    
    # Save model
    print(f"\n💾 Saving Model...")
    forecaster.save('lstm_model.h5', 'scaler.pkl')
    
    print(f"\n✅ Demo Complete!")
    print("=" * 60)