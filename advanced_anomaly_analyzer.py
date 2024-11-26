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
import json

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
        self.feature_options = {}  # Store feature analysis options
        
    def analyze(self, 
                df: pd.DataFrame,
                target_feature: str,
                analysis_features: List[str],
                feature_options: Dict[str, Dict[str, bool]],
                output_dir: Optional[str] = None) -> AnalysisResult:
        """
        Analyze the dataset using Isolation Forest
        
        Args:
            df: Input DataFrame
            target_feature: Target feature to analyze
            analysis_features: List of features to use in analysis
            feature_options: Dict of feature options {feature_name: {'use_raw': bool, 'use_ratio': bool}}
            output_dir: Directory to save results and plots
        """
        self.target_feature = target_feature
        self.analysis_features = analysis_features
        self.feature_options = feature_options
        
        # Create copy of dataframe
        df_processed = df.copy()
        
        # List to keep track of features used in analysis
        features_for_analysis = []
        
        # Add raw features if selected
        for feature in analysis_features:
            if feature_options[feature]['use_raw']:
                features_for_analysis.append(feature)
                logging.info(f"Using raw feature: {feature}")
        
        # Add ratio features if selected
        df_processed, engineered_features = self._create_engineered_features(
            df_processed, 
            target_feature,
            analysis_features
        )
        
        # Add engineered features to analysis features
        features_for_analysis.extend(engineered_features)
        
        logging.info(f"Total features used in analysis: {features_for_analysis}")
        
        # Prepare features for isolation forest
        X = df_processed[features_for_analysis].values
        
        # Run isolation forest analysis
        predictions = self.analyzer.fit_predict(
            df_processed,
            features_to_use=features_for_analysis
        )
        
        # Create results DataFrame with all original and engineered features
        results = df_processed.copy()
        results['is_anomaly'] = predictions == -1
        results['anomaly_score'] = self.analyzer.anomaly_scores
        
        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir) / "analysis_results.csv"
            results.to_csv(output_path, index=False)
            logging.info(f"Saved results with {len(engineered_features)} engineered features to {output_path}")
            
            # Save feature information
            feature_info = {
                'target_feature': target_feature,
                'raw_features_used': [f for f in analysis_features if feature_options[f]['use_raw']],
                'engineered_features': engineered_features,
                'total_features_used': features_for_analysis
            }
            
            feature_info_path = Path(output_dir) / "feature_info.json"
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=4)
            logging.info(f"Saved feature information to {feature_info_path}")
        
        # Create and return result object
        result = AnalysisResult(
            df=results,
            predictions=predictions,
            target_feature=target_feature,
            analysis_features=features_for_analysis
        )
        
        return result
    
    def _create_engineered_features(self, 
                                  df: pd.DataFrame, 
                                  target_col: str,
                                  analysis_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create engineered features based on ratios with target feature
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame with engineered features and list of created feature names
        """
        df = df.copy()
        engineered_features = []
        
        # Create ratios between target and each analysis feature
        for col in analysis_cols:
            if col != target_col and self.feature_options[col]['use_ratio']:
                feature_name = f"{target_col}_to_{col}_ratio"
                # Avoid division by zero by adding small constant
                df[feature_name] = (df[target_col] / 
                                  df[col].clip(lower=0.0001))
                
                engineered_features.append(feature_name)
                logging.info(f"Created engineered feature: {feature_name}")
        
        return df, engineered_features
    
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
    
    # Define feature options
    feature_options = {
        'IRDelta': {'use_raw': True, 'use_ratio': True},
        'IRVega': {'use_raw': True, 'use_ratio': True},
        'NPV': {'use_raw': True, 'use_ratio': False},
        'Notional': {'use_raw': True, 'use_ratio': False},
        'DaysToExpiry': {'use_raw': True, 'use_ratio': False}
    }
    
    # Run analysis with Theta as target feature
    result = analyzer.analyze(
        data=df,
        target_feature='Theta',
        analysis_features=['IRDelta', 'IRVega', 'NPV', 'Notional', 'DaysToExpiry'],
        feature_options=feature_options,
        output_dir=output_dir
    )
    
    # Print summary
    print("\nAnomaly Analysis Results:")
    print(f"Total trades analyzed: {len(df)}")
    print(f"Number of anomalies detected: {result.df['is_anomaly'].sum()}")
    
    print("\nMost anomalous trades:")
    anomalies = result.df[result.df['is_anomaly']].sort_values('anomaly_score')
    print(anomalies.head(10))
