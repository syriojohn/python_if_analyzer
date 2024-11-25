from isolation_forest_analyzer import IsolationForestAnalyzer
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisResult:
    def __init__(self, df, predictions, target_feature, analysis_features):
        self.df = df
        self.predictions = predictions
        self.target_feature = target_feature
        self.analysis_features = analysis_features

class AdvancedAnomalyAnalyzer:
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the Advanced Anomaly Analyzer
        
        Args:
            contamination: The proportion of outliers in the dataset
        """
        self.analyzer = IsolationForestAnalyzer(contamination=contamination)
        self.column_mappings = {}
        self.target_feature = None
        self.analysis_features = None
        
    def _create_engineered_features(self, 
                                  df: pd.DataFrame, 
                                  target_col: str,
                                  analysis_cols: List[str]) -> pd.DataFrame:
        """
        Create engineered features based on ratios with target feature
        """
        df = df.copy()
        
        # Create ratios between target and each analysis feature
        for col in analysis_cols:
            if col != target_col:
                feature_name = f"{target_col}_to_{col}_ratio"
                # Avoid division by zero by adding small constant
                df[feature_name] = (df[target_col] / 
                                  df[col].clip(lower=0.0001))
                
        return df
    
    def analyze(self,
               data: pd.DataFrame,
               target_feature: str,
               analysis_features: List[str],
               output_dir: Optional[str] = None) -> AnalysisResult:
        """
        Perform anomaly analysis focused on a specific feature
        
        Args:
            data: Input DataFrame
            target_feature: Main feature to analyze anomalies for
            analysis_features: Other features to include in analysis
            output_dir: Directory to save results and plots
            
        Returns:
            AnalysisResult: Object containing analysis results
        """
        # Validate features exist in dataset
        all_features = [target_feature] + analysis_features
        missing_features = [f for f in all_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataset: {missing_features}")
            
        self.target_feature = target_feature
        self.analysis_features = analysis_features
        
        # Create engineered features
        df_engineered = self._create_engineered_features(
            data, target_feature, analysis_features
        )
        
        # Combine original and engineered features
        features_for_analysis = analysis_features
        
        # Run Isolation Forest
        predictions = self.analyzer.fit_predict(
            df_engineered,
            features_to_use=features_for_analysis
        )
        
        # Create results DataFrame
        results = data.copy()
        results['is_anomaly'] = predictions == -1
        results['anomaly_score'] = self.analyzer.anomaly_scores
        
        # Save results if output directory provided
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(output_dir) / f"anomaly_results_{timestamp}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        # Create and return result object
        result = AnalysisResult(
            df=results,
            predictions=predictions,
            target_feature=target_feature,
            analysis_features=analysis_features
        )
        
        return result
    
    def _generate_visualizations(self, 
                               df: pd.DataFrame,
                               predictions: np.ndarray,
                               output_dir: str):
        """
        Generate and save all visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Individual feature plots against target
        for feature in self.analysis_features:
            plt.figure(figsize=(12, 6))
            plt.scatter(df[feature][predictions == 1],
                      df[self.target_feature][predictions == 1],
                      c='blue', label='Normal')
            plt.scatter(df[feature][predictions == -1],
                      df[self.target_feature][predictions == -1],
                      c='red', label='Anomaly')
            plt.xlabel(feature)
            plt.ylabel(self.target_feature)
            plt.title(f'{self.target_feature} vs {feature}')
            plt.legend()
            plt.savefig(output_dir / f'plot_{self.target_feature}_vs_{feature}.png')
            plt.close()
            
        # 2. Anomaly score distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=self.analyzer.anomaly_scores, bins=50)
        plt.xlabel('Anomaly Score')
        plt.title('Distribution of Anomaly Scores')
        plt.savefig(output_dir / 'anomaly_score_distribution.png')
        plt.close()
    
    def _save_feature_contributions(self, output_dir: str):
        """
        Save feature contributions to a text file
        """
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(output_dir / f'feature_contributions_{timestamp}.txt', 'w') as f:
            f.write("Feature Contributions to Anomaly Detection\n")
            f.write("=" * 40 + "\n\n")
            
            # Removed feature contribution calculation logic

if __name__ == "__main__":
    # Example usage
    data_path = r"C:\Users\syrio\OneDrive\Documents\ai_use\ml_analysis\reduced_swaption_data.csv"
    df = pd.read_csv(data_path)
    
    # Specify output directory
    output_dir = "anomaly_results"
    
    # Create analyzer
    analyzer = AdvancedAnomalyAnalyzer(contamination=0.1)
    
    # Run analysis with Theta as target feature
    result = analyzer.analyze(
        data=df,
        target_feature='Theta',
        analysis_features=['IRDelta', 'IRVega', 'NPV', 'Notional', 'DaysToExpiry'],
        output_dir=output_dir
    )
    
    # Print summary
    print("\nAnomaly Analysis Results:")
    print(f"Total trades analyzed: {len(df)}")
    print(f"Number of anomalies detected: {result.df['is_anomaly'].sum()}")
    
    print("\nMost anomalous trades:")
    anomalies = result.df[result.df['is_anomaly']].sort_values('anomaly_score')
    print(anomalies.head(10))
