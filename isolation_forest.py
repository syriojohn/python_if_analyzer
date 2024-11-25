import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from typing import List, Union, Optional


class DynamicIsolationForest:
    """
    A dynamic implementation of Isolation Forest that allows flexible feature selection
    and easy configuration of parameters.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = 'auto',
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize the Dynamic Isolation Forest.

        Args:
            n_estimators: Number of isolation trees
            max_samples: Number of samples to draw for each tree
            contamination: Expected proportion of outliers in the dataset
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs to run. -1 means using all processors
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.features = None
        self.feature_names = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        features: Optional[List[str]] = None
    ) -> 'DynamicIsolationForest':
        """
        Fit the isolation forest model.

        Args:
            X: Input data, can be DataFrame or numpy array
            features: List of feature names to use. If None, uses all features

        Returns:
            self: The fitted model
        """
        # Handle different input types
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            if features is not None:
                self.features = features
                X = X[features]
            else:
                self.features = self.feature_names
            X = X.values
        else:
            if features is not None:
                if len(features) != X.shape[1]:
                    raise ValueError("Number of features doesn't match data dimensions")
                self.features = features
            else:
                self.features = [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_names = self.features

        # Initialize and fit the model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.model.fit(X)
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict if instances are anomalies.

        Args:
            X: Input data

        Returns:
            np.ndarray: 1 for inliers, -1 for outliers
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X[self.features].values
            
        return self.model.predict(X)
    
    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute anomaly scores for each sample.

        Args:
            X: Input data

        Returns:
            np.ndarray: Anomaly scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")
            
        if isinstance(X, pd.DataFrame):
            X = X[self.features].values
            
        return -self.model.score_samples(X)  # Negative scores so higher = more anomalous
    
    def plot_anomaly_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> None:
        """
        Plot the anomaly scores distribution.

        Args:
            X: Input data
            threshold: Optional threshold to mark anomalies
        """
        scores = self.score_samples(X)
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, density=True, alpha=0.7)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        
        if threshold is not None:
            plt.axvline(x=threshold, color='r', linestyle='--',
                       label=f'Threshold: {threshold:.2f}')
            plt.legend()
            
        plt.show()
        
    def get_feature_importance(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """
        Calculate feature importance based on average depth of splits.

        Args:
            X: Input data

        Returns:
            pd.Series: Feature importance scores
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.features].values
            
        # Calculate average path length for each feature
        n_samples = X.shape[0]
        depths = np.zeros((n_samples, len(self.features)))
        
        for tree in self.model.estimators_:
            for i, sample in enumerate(X):
                path_length = 0
                current_node = 0
                
                while tree.tree_.children_left[current_node] != -1:
                    feature = tree.tree_.feature[current_node]
                    depths[i, feature] += 1
                    if sample[feature] <= tree.tree_.threshold[current_node]:
                        current_node = tree.tree_.children_left[current_node]
                    else:
                        current_node = tree.tree_.children_right[current_node]
                        
        importance = np.mean(depths, axis=0)
        return pd.Series(importance, index=self.features).sort_values(ascending=False)
