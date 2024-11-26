import sys
import os
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                            QScrollArea, QFrame, QListWidget, QLineEdit,
                            QTabWidget, QCheckBox, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal, QMutex
from PySide6.QtGui import QPixmap
import pandas as pd
import numpy as np
from pathlib import Path
from advanced_anomaly_analyzer import AdvancedAnomalyAnalyzer
import logging
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from queue import Queue
import traceback
from datetime import datetime

print("Starting application...")

class QTextEditLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.widget.setReadOnly(True)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)

class AnalysisResult:
    def __init__(self, df, predictions, target_feature, analysis_features):
        self.df = df
        self.predictions = predictions
        self.target_feature = target_feature
        self.analysis_features = analysis_features

class AnalysisWorker(QThread):
    finished = Signal(object)  # Will emit AnalysisResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, input_file, output_dir, target_feature, analysis_features, contamination, feature_options):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.target_feature = target_feature
        self.analysis_features = analysis_features
        self.contamination = contamination
        self.feature_options = feature_options
        self.mutex = QMutex()
        self.abort = False

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.mutex.unlock()

    def run(self):
        """Run the analysis in a separate thread"""
        try:
            # Load data
            self.progress.emit("Loading data...")
            df = pd.read_csv(self.input_file)
            
            # Convert numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if self.abort:
                return
            
            # Remove target feature from analysis features if present
            if self.target_feature in self.analysis_features:
                self.analysis_features.remove(self.target_feature)
            
            # Run analysis
            self.progress.emit("Running analysis...")
            analyzer = AdvancedAnomalyAnalyzer(contamination=self.contamination)
            result = analyzer.analyze(
                df=df,  # Changed from data=df to df=df
                target_feature=self.target_feature,
                analysis_features=self.analysis_features,
                feature_options=self.feature_options,
                output_dir=self.output_dir
            )
            
            if self.abort:
                return
            
            # Emit the result
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Error in analysis: {str(e)}\n{traceback.format_exc()}")

class PlotViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Create scroll area for plots
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container for plots
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        
        self.scroll.setWidget(self.plot_container)
        self.layout.addWidget(self.scroll)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Plots")
        self.refresh_btn.clicked.connect(self.load_plots)
        self.layout.addWidget(self.refresh_btn)
        
        self.plots_dir = None
        
    def set_plots_directory(self, directory):
        """Set the directory containing plot images"""
        self.plots_dir = directory
        self.load_plots()
        
    def load_plots(self):
        """Load and display all plot images from the plots directory"""
        # Clear existing plots
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        if not self.plots_dir or not os.path.exists(self.plots_dir):
            return
            
        # Load each PNG file
        for filename in sorted(os.listdir(self.plots_dir)):
            if filename.endswith('.png'):
                # Create label for plot title
                title = filename.replace('.png', '').replace('_', ' ').title()
                title_label = QLabel(title)
                title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.plot_layout.addWidget(title_label)
                
                # Create label for plot image
                plot_label = QLabel()
                pixmap = QPixmap(os.path.join(self.plots_dir, filename))
                scaled_pixmap = pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
                plot_label.setPixmap(scaled_pixmap)
                plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.plot_layout.addWidget(plot_label)
                
                # Add spacing between plots
                self.plot_layout.addSpacing(20)

class FeatureOptionsWidget(QWidget):
    def __init__(self, feature_name, parent=None):
        super().__init__(parent)
        self.feature_name = feature_name
        layout = QHBoxLayout(self)
        
        # Feature name label
        self.label = QLabel(feature_name)
        layout.addWidget(self.label)
        
        # Checkboxes for options
        self.use_raw = QCheckBox("Use Raw")
        self.use_raw.setChecked(True)  # Default to True
        self.use_ratio = QCheckBox("Use Ratio")
        self.use_ratio.setChecked(True)  # Default to True
        
        layout.addWidget(self.use_raw)
        layout.addWidget(self.use_ratio)
        
    def get_options(self):
        return {
            'use_raw': self.use_raw.isChecked(),
            'use_ratio': self.use_ratio.isChecked()
        }

class AnomalyDetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Financial Anomaly Detector")
        self.setMinimumSize(1200, 800)  # Increased window size
        
        # Load config if exists
        self.config = self.load_config()
        
        # Create main widget with tab widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Analysis tab
        self.analysis_tab = QWidget()
        self.setup_analysis_tab()
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        
        # Plot viewer tab
        self.plot_viewer = PlotViewer()
        self.tab_widget.addTab(self.plot_viewer, "Plots")
        
        self.main_layout.addWidget(self.tab_widget)
        
    def setup_analysis_tab(self):
        """Set up the analysis tab with existing GUI elements"""
        layout = QVBoxLayout(self.analysis_tab)
        
        # File selection
        file_layout = QHBoxLayout()
        self.input_file_label = QLabel("Input File:")
        self.input_file_path = QLineEdit()
        self.input_file_button = QPushButton("Browse")
        file_layout.addWidget(self.input_file_label)
        file_layout.addWidget(self.input_file_path)
        file_layout.addWidget(self.input_file_button)
        layout.addLayout(file_layout)
        
        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_dir_label = QLabel("Output Directory:")
        self.output_dir_path = QLineEdit()
        self.output_dir_button = QPushButton("Browse")
        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(self.output_dir_path)
        output_layout.addWidget(self.output_dir_button)
        layout.addLayout(output_layout)
        
        # Set paths from config if available
        if self.config:
            if 'input_file' in self.config:
                self.input_file_path.setText(self.config['input_file'])
                self.input_file = self.config['input_file']
                if os.path.exists(self.input_file):
                    self.load_features()
            
            if 'output_dir' in self.config:
                self.output_dir_path.setText(self.config['output_dir'])
                self.output_dir = self.config['output_dir']
        
        # Feature selection area
        feature_layout = QHBoxLayout()
        
        # Target feature selection
        target_layout = QVBoxLayout()
        self.target_label = QLabel("Target Feature:")
        self.target_combo = QComboBox()
        target_layout.addWidget(self.target_label)
        target_layout.addWidget(self.target_combo)
        feature_layout.addLayout(target_layout)
        
        # Analysis features selection
        analysis_layout = QVBoxLayout()
        self.features_label = QLabel("Analysis Features:")
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        analysis_layout.addWidget(self.features_label)
        analysis_layout.addWidget(self.features_list)
        feature_layout.addLayout(analysis_layout)
        
        layout.addLayout(feature_layout)
        
        # Feature options area
        self.feature_options_group = QGroupBox("Feature Analysis Options")
        self.feature_options_layout = QVBoxLayout(self.feature_options_group)
        self.feature_options_widgets = {}  # Store widgets for each feature
        layout.addWidget(self.feature_options_group)
        
        # Contamination factor
        contamination_layout = QHBoxLayout()
        self.contamination_label = QLabel("Contamination Factor:")
        self.contamination_spin = QDoubleSpinBox()
        self.contamination_spin.setRange(0.01, 0.5)
        self.contamination_spin.setSingleStep(0.01)
        self.contamination_spin.setValue(0.1)
        contamination_layout.addWidget(self.contamination_label)
        contamination_layout.addWidget(self.contamination_spin)
        layout.addLayout(contamination_layout)
        
        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)
        
        # Log viewer
        self.log_text = QTextEdit()
        layout.addWidget(self.log_text)
        
        # Connect signals
        self.input_file_button.clicked.connect(self.browse_input_file)
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        self.run_button.clicked.connect(self.run_analysis)
        self.features_list.itemSelectionChanged.connect(self.update_feature_options)
        self.target_combo.currentTextChanged.connect(self.check_run_button)
        self.features_list.itemSelectionChanged.connect(self.check_run_button)
        
        # Initialize worker
        self.worker = None

    def load_config(self):
        """Load configuration from config.json if it exists"""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading config: {str(e)}")
        return None

    def generate_plots(self, result, output_dir):
        """Generate visualization plots in the main thread"""
        try:
            # Create plots directory if it doesn't exist
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Set plot style
            plt.style.use('default')  # Use default style instead of seaborn
            
            # 1. Individual feature plots with trend lines
            for feature in result.analysis_features:
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Normal points
                    normal_mask = result.predictions == 1
                    plt.scatter(result.df[feature][normal_mask],
                              result.df[result.target_feature][normal_mask],
                              c='blue', label='Normal', alpha=0.6)
                    
                    # Anomaly points
                    anomaly_mask = result.predictions == -1
                    plt.scatter(result.df[feature][anomaly_mask],
                              result.df[result.target_feature][anomaly_mask],
                              c='red', label='Anomaly', alpha=0.8)
                    
                    # Add trend line
                    z = np.polyfit(result.df[feature].fillna(result.df[feature].mean()), 
                                 result.df[result.target_feature].fillna(result.df[result.target_feature].mean()), 1)
                    p = np.poly1d(z)
                    plt.plot(result.df[feature], p(result.df[feature]), "k--", alpha=0.5, label='Trend')
                    
                    plt.xlabel(feature)
                    plt.ylabel(result.target_feature)
                    plt.title(f'{result.target_feature} vs {feature}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(plots_dir, f'plot_{result.target_feature}_vs_{feature}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logging.info(f"Generated plot for {feature}")
                except Exception as e:
                    logging.warning(f"Could not generate plot for {feature}: {str(e)}")
                    continue

            # 2. Correlation Matrix
            try:
                plt.figure(figsize=(10, 8))
                features_to_correlate = [result.target_feature] + result.analysis_features
                correlation_matrix = result.df[features_to_correlate].corr()
                
                sns.heatmap(correlation_matrix, 
                           annot=True, 
                           cmap='RdYlBu', 
                           center=0,
                           fmt='.2f')
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info("Generated correlation matrix")
            except Exception as e:
                logging.warning(f"Could not generate correlation matrix: {str(e)}")

            # 3. Anomaly Score Distribution
            try:
                plt.figure(figsize=(10, 6))
                scores = pd.Series(result.predictions).map({1: 'Normal', -1: 'Anomaly'})
                scores.value_counts().plot(kind='bar')
                plt.title('Distribution of Normal vs Anomaly Points')
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'anomaly_distribution.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info("Generated distribution plot")
            except Exception as e:
                logging.warning(f"Could not generate distribution plot: {str(e)}")

        except Exception as e:
            logging.error(f"Error in plot generation: {str(e)}")
        finally:
            plt.close('all')

        # Update plot viewer
        self.plot_viewer.set_plots_directory(plots_dir)
        self.tab_widget.setCurrentWidget(self.plot_viewer)  # Switch to plot viewer tab

    def analysis_complete(self, result):
        """Handle analysis completion in the main thread"""
        try:
            # Generate plots in the main thread
            self.generate_plots(result, self.output_dir_path.text())
            logging.info("Analysis and visualization complete! Check output directory for results.")
        except Exception as e:
            logging.error(f"Error in analysis completion: {str(e)}")
        finally:
            self.run_button.setEnabled(True)
            if self.worker:
                self.worker.stop()
                self.worker = None

    def browse_input_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input CSV File",
            self.input_file_path.text(),
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if file_name:
            self.input_file = file_name
            self.input_file_path.setText(file_name)
            self.load_features()
            self.check_run_button()

    def browse_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_path.text()
        )
        if dir_name:
            self.output_dir = dir_name
            self.output_dir_path.setText(dir_name)
            self.check_run_button()

    def load_features(self):
        """Load features from the input CSV file"""
        try:
            if self.input_file and os.path.exists(self.input_file):
                df = pd.read_csv(self.input_file)
                features = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Update target feature combo box
                self.target_combo.clear()
                self.target_combo.addItems(features)
                
                # Update analysis features list
                self.features_list.clear()
                self.features_list.addItems(features)
                
                logging.info(f"Loaded {len(features)} features from input file")
                self.check_run_button()
        except Exception as e:
            logging.error(f"Error loading features: {str(e)}")

    def update_feature_options(self):
        """Update feature options when selection changes"""
        # Clear existing options
        for i in reversed(range(self.feature_options_layout.count())):
            self.feature_options_layout.itemAt(i).widget().setParent(None)
        
        # Add options for selected features
        selected_items = self.features_list.selectedItems()
        self.feature_options_widgets = {}
        
        for item in selected_items:
            feature_name = item.text()
            option_widget = FeatureOptionsWidget(feature_name)
            self.feature_options_widgets[feature_name] = option_widget
            self.feature_options_layout.addWidget(option_widget)
            
    def get_feature_options(self):
        """Get the current feature options for all selected features"""
        options = {}
        for feature_name, widget in self.feature_options_widgets.items():
            options[feature_name] = widget.get_options()
        return options

    def check_run_button(self):
        """Enable run button only if all selections are valid"""
        input_valid = self.input_file and os.path.exists(self.input_file)
        output_valid = self.output_dir and os.path.exists(self.output_dir)
        target_valid = self.target_combo.currentText() != ""
        analysis_valid = len(self.features_list.selectedItems()) > 0
        
        self.run_button.setEnabled(
            input_valid and output_valid and target_valid and analysis_valid
        )

    def validate_feature_selection(self):
        """Validate feature selection and return error message if invalid"""
        target_feature = self.target_combo.currentText()
        analysis_features = [item.text() for item in self.features_list.selectedItems()]
        
        if not target_feature:
            return "Please select a target feature"
            
        if not analysis_features:
            return "Please select at least one analysis feature"
            
        # Remove target feature from analysis features if present
        if target_feature in analysis_features:
            analysis_features.remove(target_feature)
            self.features_list.clearSelection()
            for i in range(self.features_list.count()):
                item = self.features_list.item(i)
                if item.text() != target_feature:
                    item.setSelected(True)
            return "Target feature cannot be included in analysis features"
        
        return None

    def run_analysis(self):
        # Validate inputs
        if not self.input_file_path.text():
            logging.error("Please select an input file")
            return
            
        if not self.output_dir_path.text():
            logging.error("Please select an output directory")
            return
        
        # Validate feature selection
        error_msg = self.validate_feature_selection()
        if error_msg:
            logging.error(error_msg)
            return
            
        target_feature = self.target_combo.currentText()
        analysis_features = [item.text() for item in self.features_list.selectedItems()]
        
        self.run_button.setEnabled(False)
        
        try:
            self.worker = AnalysisWorker(
                self.input_file,
                self.output_dir,
                target_feature,
                analysis_features,
                self.contamination_spin.value(),
                self.get_feature_options()
            )
            
            self.worker.progress.connect(lambda msg: logging.info(msg))
            self.worker.error.connect(lambda msg: logging.error(msg))
            self.worker.finished.connect(self.analysis_complete)
            
            logging.info("Starting analysis...")
            self.worker.start()
        except Exception as e:
            logging.error(f"Error starting analysis: {str(e)}")
            self.run_button.setEnabled(True)

def setup_logging():
    """Set up logging to both file and console"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    return log_file

if __name__ == '__main__':
    print("Initializing GUI...")
    log_file = setup_logging()
    logging.info(f"Application started. Log file: {log_file}")
    
    try:
        app = QApplication(sys.argv)
        print("Creating main window...")
        window = AnomalyDetectorGUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise
