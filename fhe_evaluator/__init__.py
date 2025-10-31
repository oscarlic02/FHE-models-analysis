"""
FHE Model Evaluator Library

A framework for evaluating Fully Homomorphic Encryption (FHE) models
implemented using concrete-ml and comparing them against traditional scikit-learn models.

This library provides:
- Automated hyperparameter tuning for multiple model types
- Performance evaluation and comparison between FHE and plain models  
- Pareto optimality analysis for accuracy vs latency trade-offs
- Comprehensive visualization and reporting capabilities
- Support for various ML algorithms (Logistic Regression, Random Forest, Decision Tree, XGBoost, MLP)

Example usage:
    ```python
    from fhe_evaluator import FHEModelEvaluator
    
    evaluator = FHEModelEvaluator(
        data=df,
        target_column='target',
        model_types=['lr', 'rf', 'dt'],
        bit_widths=[2, 3, 4, 6, 8]
    )
    
    results = evaluator.run_full_pipeline()
    pareto_results = evaluator.find_pareto_optimal_solutions()
    ```
"""

from .core import FHEModelEvaluator
from .config import ModelConfig, DEFAULT_PARAM_GRIDS, DEFAULT_BIT_WIDTHS
from .utils import ModelCreator, DataProcessor, MetricsCalculator, LatencyMeasurer
from .exceptions import (
    FHEEvaluatorError,
    DataProcessingError,
    ModelEvaluationError,
    VisualizationError
)

__version__ = "1.0.0"
__author__ = "Oscar Licciardi"
__email__ = "oscar.licciardi@studenti.polito.it"

__all__ = [
    "FHEModelEvaluator",
    "ModelConfig",
    "ModelCreator",
    "DataProcessor", 
    "MetricsCalculator",
    "LatencyMeasurer",
    "DEFAULT_PARAM_GRIDS", 
    "DEFAULT_BIT_WIDTHS",
    "FHEEvaluatorError",
    "DataProcessingError",
    "ModelEvaluationError",
    "VisualizationError"
]
