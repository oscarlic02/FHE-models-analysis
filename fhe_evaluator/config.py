"""
Configuration module for FHE Model Evaluator.

Contains default configurations, parameter grids, and model settings.
"""

from typing import Dict, List, Any, Optional

DEFAULT_PARAM_GRIDS = {
    'lr': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['liblinear']
    },
    'rf': {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 3, 5]
    },
    'dt': {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgb': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
}

# Default bit widths for FHE evaluation
DEFAULT_BIT_WIDTHS = [2, 3, 4, 6, 8]

# Default configuration for the evaluator
DEFAULT_CONFIG = {
    'random_state': 42,
    'n_jobs': -1,
    'verbose': True,
    'test_size': 0.2,
    'undersampling_ratio': None,
    'scaling': True,
    'model_types': ['lr', 'rf', 'dt'],
    'param_grids': DEFAULT_PARAM_GRIDS,
    'bit_widths': DEFAULT_BIT_WIDTHS,
    'cv_folds': 5,
    'scoring': 'f1',
    'n_iterations': 100
}


class ModelConfig:
    """Configuration class for FHE model evaluation settings."""
    
    def __init__(self, 
                 model_types: Optional[List[str]] = None,
                 param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
                 bit_widths: Optional[List[int]] = None,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 test_size: float = 0.2,
                 undersampling_ratio: Optional[float] = None,
                 scaling: bool = True,
                 cv_folds: int = 5,
                 scoring: str = 'f1',
                 n_iterations: int = 100):
        """
        Initialize model configuration.
        
        Args:
            model_types: List of model types to evaluate
            param_grids: Custom parameter grids for hyperparameter tuning
            bit_widths: List of bit widths for FHE evaluation
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for computations
            verbose: Flag to enable verbose output
            test_size: Proportion of dataset for test split
            undersampling_ratio: Ratio for undersampling majority class
            scaling: Whether to apply StandardScaler to features
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            n_iterations: Number of iterations for latency measurement
        """
        self.model_types = model_types or DEFAULT_CONFIG['model_types']
        self.param_grids = param_grids or DEFAULT_PARAM_GRIDS
        self.bit_widths = bit_widths or DEFAULT_BIT_WIDTHS
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.test_size = test_size
        self.undersampling_ratio = undersampling_ratio
        self.scaling = scaling
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_iterations = n_iterations
    
    def get_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        """Get parameter grid for a specific model type."""
        if model_type not in self.param_grids:
            raise KeyError(f"No parameter grid defined for model type: {model_type}")
        return self.param_grids[model_type]
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.model_types:
            raise ValueError("At least one model type must be specified")
        
        if not self.bit_widths:
            raise ValueError("At least one bit width must be specified")
        
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
            
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
            
        if self.undersampling_ratio is not None and not (0 < self.undersampling_ratio <= 1):
            raise ValueError("undersampling_ratio must be between 0 and 1")


MODEL_MAPPINGS = {
    'lr': ('LogisticRegression', 'LogisticRegression'),
    'rf': ('RandomForest', 'RandomForestClassifier'),
    'dt': ('DecisionTree', 'DecisionTreeClassifier'),
    'mlp': ('MLP', 'MLPClassifier'),
    'xgb': ('XGBoost', 'XGBClassifier') 
}
