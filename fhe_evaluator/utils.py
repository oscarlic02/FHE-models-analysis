"""
Utility classes and functions for the FHE Model Evaluator library.

This module provides helper classes for model creation, data processing,
and metrics calculation.
"""

import time
import warnings
from typing import Dict, List, Any, Tuple, Optional, Union
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler

from .exceptions import ModelNotSupportedError, DataProcessingError


class ModelCreator:
    """Factory class for creating model instances."""
    
    def __init__(self, config=None):
        """
        Initialize ModelCreator.
        
        Args:
            config: ModelConfig instance (optional, for future extensibility)
        """
        self.config = config
    
    # Available models mapping
    _sklearn_models = {
        'lr': LogisticRegression,
        'rf': RandomForestClassifier,
        'dt': DecisionTreeClassifier,
        'mlp': MLPClassifier
    }
    
    # Try to import XGBoost if available
    try:
        import xgboost as xgb
        _sklearn_models['xgb'] = xgb.XGBClassifier
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False
    
    # FHE model class names mapping
    _fhe_models = {
        'lr': 'LogisticRegression',
        'rf': 'RandomForestClassifier', 
        'dt': 'DecisionTreeClassifier',
        'mlp': 'MLPClassifier',
        'xgb': 'XGBClassifier'
    }
    
    @classmethod
    def get_sklearn_estimator(cls, model_type: str, random_state: int = 42, **kwargs) -> Any:
        """
        Create a scikit-learn model instance.
        
        Args:
            model_type: Type of model ('lr', 'rf', 'dt', 'mlp', 'xgb')
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the model
            
        Returns:
            Instantiated model
            
        Raises:
            ModelNotSupportedError: If model type is not supported
        """
        return cls.create_sklearn_model(model_type, random_state, **kwargs)
    
    @classmethod
    def create_sklearn_model(cls, model_type: str, random_state: int = 42, **kwargs) -> Any:
        """
        Create a scikit-learn model instance.
        
        Args:
            model_type: Type of model ('lr', 'rf', 'dt', 'mlp', 'xgb')
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the model
            
        Returns:
            Instantiated model
            
        Raises:
            ModelNotSupportedError: If model type is not supported
        """
        model_type = model_type.lower()
        
        if model_type not in cls._sklearn_models:
            available_models = list(cls._sklearn_models.keys())
            raise ModelNotSupportedError(
                f"Model type '{model_type}' not supported. "
                f"Available models: {available_models}"
            )
        
        model_class = cls._sklearn_models[model_type]
        return model_class(random_state=random_state, **kwargs)
    
    @classmethod
    def create_fhe_model(cls, model_type: str, n_bits: int = 4, 
                        random_state: int = 42, **kwargs) -> Any:
        """
        Create an FHE model instance.
        
        Args:
            model_type: Type of model ('lr', 'rf', 'dt', 'mlp', 'xgb')
            n_bits: Number of bits for FHE quantization
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the model
            
        Returns:
            Instantiated FHE model
            
        Raises:
            ModelNotSupportedError: If model type is not supported
        """
        model_type = model_type.lower()
        
        if model_type not in cls._fhe_models:
            available_models = list(cls._fhe_models.keys())
            raise ModelNotSupportedError(
                f"FHE model type '{model_type}' not supported. "
                f"Available models: {available_models}"
            )
        
        try:
            # Import concrete-ml module
            fhe_module = import_module('concrete.ml.sklearn')
            fhe_class_name = cls._fhe_models[model_type]
            fhe_class = getattr(fhe_module, fhe_class_name)
            
            return fhe_class(n_bits=n_bits, random_state=random_state, **kwargs)
            
        except ImportError as e:
            raise ModelNotSupportedError(
                f"concrete-ml not available. Please install it to use FHE models: {e}"
            )
        except AttributeError as e:
            raise ModelNotSupportedError(
                f"FHE model class '{fhe_class_name}' not found in concrete.ml.sklearn: {e}"
            )
    
    @classmethod
    def get_fhe_estimator(cls, model_type: str, n_bits: int = 4,
                         random_state: int = 42, **kwargs) -> Any:
        """
        Create an FHE model instance.
        
        Args:
            model_type: Type of model ('lr', 'rf', 'dt', 'mlp', 'xgb')
            n_bits: Number of bits for FHE quantization
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the model
            
        Returns:
            Instantiated FHE model
            
        Raises:
            ModelNotSupportedError: If model type is not supported
        """
        return cls.create_fhe_model(model_type, n_bits, random_state, **kwargs)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model types."""
        return list(cls._sklearn_models.keys())


class DataProcessor:
    """Class for data processing operations."""
    
    def __init__(self, config, target_column: str = 'target'):
        """
        Initialize data processor.
        
        Args:
            config: ModelConfig instance containing processing parameters
            target_column: Name of the target column in the dataset
        """
        self.config = config
        self.test_size = config.test_size
        self.scaling = config.scaling
        self.undersampling_ratio = config.undersampling_ratio
        self.random_state = config.random_state
        self.target_column = target_column
        
        self.scaler: Optional[StandardScaler] = None
        self.undersampler: Optional[RandomUnderSampler] = None
    
    def process_data(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Process the dataset for machine learning.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column (overrides instance setting if provided)
            
        Returns:
            Dictionary containing processed data arrays:
            - X: Full feature matrix
            - y: Full target vector  
            - X_train: Training features
            - X_test: Test features
            - y_train: Training targets
            - y_test: Test targets
            
        Raises:
            DataProcessingError: If data processing fails
        """
        try:
            # Use provided target column or fall back to instance setting
            target_col = target_column or self.target_column
            
            # Validate inputs
            if data is None or data.empty:
                raise DataProcessingError("Input data is empty or None")
            
            if target_col not in data.columns:
                raise DataProcessingError(
                    f"Target column '{target_col}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            
            # Extract features and target
            X = data.drop(target_col, axis=1).values
            y = data[target_col].values
            
            # Apply scaling if requested
            if self.scaling:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=y
            )
            
            # Setup undersampler
            if self.undersampling_ratio is not None:
                self.undersampler = RandomUnderSampler(
                    sampling_strategy=self.undersampling_ratio,
                    random_state=self.random_state
                )
            else:
                self.undersampler = RandomUnderSampler(
                    random_state=self.random_state
                )
            
            return {
                'X': X,
                'y': y,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'undersampler': self.undersampler
            }
            
        except Exception as e:
            raise DataProcessingError(f"Failed to process data: {str(e)}")
    
    def get_class_distribution(self, data: pd.DataFrame) -> pd.Series:
        """Get class distribution of target variable."""
        if self.target_column not in data.columns:
            raise DataProcessingError(f"Target column '{self.target_column}' not found")
        
        return data[self.target_column].value_counts(normalize=True) * 100


class MetricsCalculator:
    """Class for calculating evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        return MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted'),
            'precision': precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted'),
            'recall': recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'weighted')
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan
        
        return metrics


class LatencyMeasurer:
    """Utility class for measuring model inference latency."""
    
    def __init__(self, n_iterations: int = 100):
        """
        Initialize latency measurer.
        
        Args:
            n_iterations: Number of iterations for measurement
        """
        self.n_iterations = n_iterations
    
    def measure_latency(self, model: Any, X_test: np.ndarray, n_iterations: int = None) -> float:
        """
        Measure average inference latency.
        
        Args:
            model: Trained model
            X_test: Test data  
            n_iterations: Number of iterations for measurement (overrides default)
            
        Returns:
            Average latency in milliseconds
        """
        iterations = n_iterations or self.n_iterations
        inference_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            _ = model.predict(X_test)
            inference_times.append(time.time() - start_time)
        
        return np.mean(inference_times) * 1000  # Convert to milliseconds


def suppress_warnings():
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
