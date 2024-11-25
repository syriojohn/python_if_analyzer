import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationForestAnalyzer:
    def __init__(self, 
                 contamination: float = 0.1,
                 random_state: int = 42,
                 n_estimators: int = 100):
        """
        Initialize the Isolation Forest Analyzer
        
        Args:
            contamination: The proportion of outliers in the dataset
            random_state: Random state for reproducibility
            n_estimators: Number of trees in the forest
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators
        )
        self.scaler = StandardScaler()
        self.features_used = None
        self.anomaly_scores = None
        
    def fit_predict(self, 
                   data: pd.DataFrame,
                   features_to_use: List[str] = None,
                   exclude_features: List[str] = None) -> np.ndarray:
        """
        Fit the model and predict anomalies
        
        Args:
            data: DataFrame containing the data
            features_to_use: List of column names to use for detection
            exclude_features: List of column names to exclude
            
        Returns:
            Array of predictions where -1 indicates anomalies and 1 indicates normal data
        """
        # Determine features to use
        if features_to_use is None:
            # Use all numeric columns by default
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            features_to_use = [col for col in numeric_cols]
        
        if exclude_features:
            features_to_use = [f for f in features_to_use if f not in exclude_features]
            
        logger.info(f"Using features: {features_to_use}")
        self.features_used = features_to_use
        
        # Prepare the data
        X = data[features_to_use].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit and predict
        predictions = self.model.fit_predict(X_scaled)
        
        # Calculate anomaly scores
        self.anomaly_scores = self.model.score_samples(X_scaled)
        
        return predictions
    
    def get_anomaly_details(self, 
                           data: pd.DataFrame,
                           predictions: np.ndarray,
                           additional_info_cols: List[str] = None) -> pd.DataFrame:
        """
        Get detailed information about detected anomalies
        
        Args:
            data: Original DataFrame
            predictions: Array of predictions from fit_predict
            additional_info_cols: Additional columns to include in the output
            
        Returns:
            DataFrame containing anomaly details
        """
        anomaly_df = data.copy()
        anomaly_df['is_anomaly'] = predictions == -1
        anomaly_df['anomaly_score'] = self.anomaly_scores
        
        # Sort by anomaly score (most anomalous first)
        anomaly_df = anomaly_df.sort_values('anomaly_score')
        
        # Select columns to display
        display_cols = ['is_anomaly', 'anomaly_score']
        if self.features_used:
            display_cols.extend(self.features_used)
        if additional_info_cols:
            display_cols.extend(additional_info_cols)
            
        return anomaly_df[display_cols]
    
    def plot_anomalies(self, 
                      data: pd.DataFrame,
                      predictions: np.ndarray,
                      feature_x: str,
                      feature_y: str,
                      figsize: tuple = (10, 6)):
        """
        Create a scatter plot highlighting anomalies
        
        Args:
            data: Original DataFrame
            predictions: Array of predictions
            feature_x: Feature to plot on x-axis
            feature_y: Feature to plot on y-axis
            figsize: Size of the figure
        """
        plt.figure(figsize=figsize)
        
        # Plot normal points
        normal_mask = predictions == 1
        plt.scatter(data[feature_x][normal_mask], 
                   data[feature_y][normal_mask],
                   c='blue', label='Normal')
        
        # Plot anomalies
        anomaly_mask = predictions == -1
        plt.scatter(data[feature_x][anomaly_mask],
                   data[feature_y][anomaly_mask],
                   c='red', label='Anomaly')
        
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title('Anomaly Detection Results')
        plt.legend()
        plt.show()
        
    def plot_feature_importances(self, data: pd.DataFrame):
        """
        Plot the relative importance of features based on anomaly scores
        
        Args:
            data: Original DataFrame
        """
        if self.features_used is None:
            logger.warning("No features available. Run fit_predict first.")
            return
            
        # Calculate correlation between features and anomaly scores
        correlations = {}
        for feature in self.features_used:
            corr = np.abs(np.corrcoef(data[feature], self.anomaly_scores)[0, 1])
            correlations[feature] = corr
            
        # Create bar plot
        plt.figure(figsize=(10, 6))
        features = list(correlations.keys())
        scores = list(correlations.values())
        
        plt.bar(features, scores)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation with Anomaly Score')
        plt.title('Feature Importance in Anomaly Detection')
        plt.tight_layout()
        plt.show()
