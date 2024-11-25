# Financial Anomaly Detection Tool

## Project Overview
Advanced anomaly detection tool for financial trading data, specifically focused on swaption analysis. The tool uses Isolation Forest algorithm combined with a modern PySide6-based GUI interface for interactive analysis and visualization.

## Features
- Interactive GUI with tabbed interface for analysis and visualization
- Real-time anomaly detection using Isolation Forest
- Dynamic feature selection and target variable configuration
- Interactive plot viewer with auto-refresh capability
- Comprehensive logging system with file and console output
- Thread-safe plot generation
- Configuration management via JSON
- Detailed error tracking and reporting

## Installation

### Prerequisites
- Python 3.13.0 or later
- Windows OS (tested on Windows 10/11)

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
   - Analysis Features: Choose features for anomaly detection (e.g., IRDelta, NPV)
   - Contamination Factor: Adjust expected anomaly percentage (default: 0.1)

3. Run Analysis:
   - Click "Run Analysis" button
   - Monitor progress in the log window
   - Results will automatically display in the Plots tab

4. View Results:
   - Switch between "Analysis" and "Plots" tabs
   - Use "Refresh Plots" button to update visualizations
   - Examine various plot types:
     * Individual feature scatter plots with trend lines and anomaly highlighting
     * Correlation matrix showing relationships between features
     * Anomaly distribution plots showing normal vs anomalous points

## Project Structure

### Essential Files
- `anomaly_detector_gui.py`: Main GUI Application
  * Primary entry point of the application
  * Implements the PySide6-based user interface
  * Handles user interactions and file operations
  * Manages analysis workflow and visualization
  * Contains the PlotViewer for result visualization

- `advanced_anomaly_analyzer.py`: Core Analysis Engine
  * Implements the main anomaly detection logic
  * Manages data preprocessing and analysis
  * Generates analysis results and metrics
  * Handles result visualization

- `isolation_forest.py`: Base Algorithm Implementation
  * Contains the core Isolation Forest algorithm
  * Implements the base anomaly detection methods
  * Provides the foundational machine learning functionality
  * Handles data splitting and scoring

- `isolation_forest_analyzer.py`: Extended Analyzer
  * Extends the base Isolation Forest implementation
  * Handles model training and prediction
  * Manages anomaly score computation
  * Provides interface for advanced analysis

- `config.json`: Configuration File
  * Stores default input/output paths
  * Contains user preferences
  * Manages application settings

- `requirements.txt`: Dependencies List
  * Lists all required Python packages
  * Specifies version requirements
  * Ensures consistent environment setup

## Key Components

### Analysis Results
The tool provides comprehensive analysis results including:
- Anomaly detection results for each data point
- Scatter plots showing relationships between features and target variable
- Correlation matrix to understand feature relationships
- Distribution of normal vs anomalous points

### Visualization
- Individual feature plots with:
  * Normal points in blue
  * Anomalous points in red
  * Trend lines showing relationships
- Correlation matrix heatmap
- Anomaly distribution bar chart

### Threading and Performance
- Asynchronous analysis execution
- Thread-safe plot generation
- Progress updates via signals
- Safe GUI updates via signals and slots

## Known Issues
1. Occasional GUI lag during heavy analysis
2. Memory usage with large datasets

## Future Enhancements
1. Support for additional anomaly detection algorithms
2. Enhanced visualization options
3. Export capabilities for analysis results
4. Custom feature engineering options
5. Time-series specific analysis features

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
