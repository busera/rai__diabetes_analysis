"""
RAI Toolkit Ready Pipeline Model for Diabetes Dataset

This script creates a pipeline compatible with the Microsoft Responsible AI Toolkit
for preprocessing and modeling the diabetes dataset.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger

# Set up logging
logger.add("../logs/model_training.log", rotation="500 MB")

class QuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=10):
        self.n_quantiles = n_quantiles
    
    def fit(self, X, y=None):
        self.quantiles_ = {col: np.percentile(X[col], np.linspace(0, 100, self.n_quantiles))
                           for col in X.columns}
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = pd.cut(X[col], bins=self.quantiles_[col], labels=False, include_lowest=True)
        return X_transformed

def load_data():
    """Load the diabetes dataset from Parquet files."""
    X_train = pq.read_table('../data/interim/X_train.parquet').to_pandas()
    X_val = pq.read_table('../data/interim/X_val.parquet').to_pandas()
    X_test = pq.read_table('../data/interim/X_test.parquet').to_pandas()
    y_train = pq.read_table('../data/interim/y_train.parquet').to_pandas()['target']
    y_val = pq.read_table('../data/interim/y_val.parquet').to_pandas()['target']
    y_test = pq.read_table('../data/interim/y_test.parquet').to_pandas()['target']
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_pipeline(X):
    """Create a pipeline compatible with the RAI Toolkit."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('quantile', QuantileTransformer(n_quantiles=10))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def evaluate_model(model, X, y, dataset_name):
    """Evaluate the model and log the results."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    logger.info(f"{dataset_name} - MSE: {mse:.4f}, R2: {r2:.4f}")

def main():
    """Main function to run the entire pipeline."""
    logger.info("Starting model training")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    logger.info(f"Data loaded. Training set shape: {X_train.shape}")

    # Create and train the model
    model = create_pipeline(X_train)
    model.fit(X_train, y_train)
    logger.info("Model trained")

    # Evaluate the model
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    # Get feature importance
    feature_names = (model.named_steps['preprocessor']
                     .named_transformers_['num']
                     .named_steps['quantile']
                     .quantiles_.keys())
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.named_steps['regressor'].feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("Top 5 important features:")
    logger.info(feature_importance.head().to_string(index=False))

    logger.success("Model training and evaluation complete.")

    return model, X_test, y_test

if __name__ == "__main__":
    main()