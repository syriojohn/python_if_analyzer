# Financial Anomaly Detection Tool

## Project Overview
Advanced anomaly detection tool for financial trading data, specifically focused on swaption analysis. The tool uses Isolation Forest algorithm combined with a modern PySide6-based GUI interface for interactive analysis and visualization. Particularly useful for detecting Theta blowouts and other anomalies in trading data.

## Features
- Interactive GUI with tabbed interface for analysis and visualization
- Real-time anomaly detection using Isolation Forest
- Dynamic feature selection with granular control:
  - Select target feature (e.g., Theta)
  - Choose analysis features
  - Per-feature options:
    - Use raw feature data
    - Create and use ratio with target feature
- Comprehensive output:
  - Full results CSV with original and engineered features
  - Feature information JSON file
  - Detailed visualizations
- Interactive plot viewer with auto-refresh capability
- Comprehensive logging system
- Thread-safe plot generation
- Configuration management via JSON

## Installation

### Prerequisites
- Python 3.13.0 or later
- Windows OS (tested on Windows 10/11)

### Required Packages
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
PySide6>=6.5.0
```

### Setup
```bash
# Clone the repository (if using git)
git clone [repository-url]
cd isolation_forest_tool

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Launch the application:
```bash
python anomaly_detector_gui.py
```

2. Configure Analysis:
   - Input File: Select your CSV file containing trading data
   - Output Directory: Choose where to save results and plots
   - Target Feature: Select the feature to analyze (e.g., Theta)
   - Analysis Features: Choose features for anomaly detection
   - Feature Options:
     - Use Raw: Include raw feature values in analysis
     - Use Ratio: Create and include ratio with target feature
   - Contamination Factor: Adjust expected anomaly percentage (default: 0.1)

3. Run Analysis:
   - Click "Run Analysis" button
   - Monitor progress in the log window
   - Results will automatically display in the Plots tab

4. View Results:
   - Check the output directory for:
     - `analysis_results.csv`: Full dataset with original features, engineered features, and anomaly predictions
     - `feature_info.json`: Details of features used in analysis
     - Visualization plots in the Plots tab

## Output Files

### analysis_results.csv
Contains:
- All original features from input
- Engineered features (ratios if selected)
- `is_anomaly`: Boolean indicating anomaly detection
- `anomaly_score`: Isolation Forest anomaly score

### feature_info.json
Documents:
- Target feature used
- Raw features included in analysis
- Engineered features created
- Complete list of features used in analysis

## Usage Tips

1. Feature Selection:
   - Start with raw features that might influence your target
   - Use ratios when relationships between features are important
   - Monitor the log window for feature creation information

2. Analyzing Theta Blowouts:
   - Set Theta as target feature
   - Include relevant Greeks (Delta, Vega) and market data
   - Consider both raw values and ratios for comprehensive analysis

3. Visualization:
   - Use the Plots tab to examine relationships
   - Look for patterns in anomaly distribution
   - Compare different feature combinations

## Troubleshooting

If you encounter issues:
1. Check the log window for error messages
2. Verify input data format
3. Ensure output directory is writable
4. Confirm feature selections are appropriate

## Contributing

Feel free to submit issues and enhancement requests!
