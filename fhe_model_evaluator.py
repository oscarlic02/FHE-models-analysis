import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

class FHEModelEvaluator:
    """
    A framework for evaluating Fully Homomorphic Encryption (FHE) models 
    implemented using concrete-ml and comparing them against traditional 
    scikit-learn models. This class provides methods for hyperparameter 
    tuning, model evaluation, and visualization of results.
    Attributes:
        random_state (int): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs for computations.
        verbose (bool): Flag to enable verbose output.
        results (dict): Cache for storing evaluation results.
        model_mappings (dict): Mapping of model types to their respective 
            scikit-learn and concrete-ml classes.
    Methods:
        grid_search(model_type, param_grid, x, y, undersampler=None, cv=5, scoring='f1'):
            Perform grid search to find optimal hyperparameters for a given model type.
        evaluate_models(model_types, x, y, best_params, bit_widths, undersampler=None, 
                        cv_splits=5, n_iterations=100):
        visualize_model_comparison():
            Create visualizations comparing all evaluated models.
    Private Methods:
        _get_estimator(model_type):
            Retrieve the appropriate scikit-learn estimator for a given model type.
        _get_fhe_estimator(model_type, **kwargs):
            Retrieve the appropriate FHE estimator for a given model type.
        _evaluate_model(model, x, y, undersampler, kf, n_iterations):
            Evaluate a model using cross-validation and calculate performance metrics.
        _evaluate_fhe_models(model_type, params, bit_widths, x, y, undersampler, kf, n_iterations):
            Evaluate FHE models with different bit widths and calculate performance metrics.
        _calculate_metrics(y_true, y_pred, y_pred_proba):
            Calculate classification metrics such as accuracy, F1 score, precision, recall, and ROC AUC.
        _measure_latency(model, X_test, n_iterations=100):
            Measure average inference latency in milliseconds.
        _compare_models(plain_results, cipher_results):
            Compare plain and FHE models to evaluate tradeoffs in accuracy and latency.
        _create_model_figures(plain_results, cipher_results, comparison):
            Create visualization figures for model comparison.
        _print_model_summary(results):
            Print a summary of model evaluation results.
        _print_bit_width_summary(results):
            Print a summary of the results for a given bit-width.
        _generate_summary(evaluation_results):
            Generate summary metrics across all evaluated models.
        _create_comparison_figures(models, plain_accs, fhe_accs, plain_lats, fhe_lats, overheads, bit_widths):
            Helper method to create comparison figures for accuracy and latency.
    """
    
    def __init__(self, data, target_column, random_state=42, n_jobs=-1, verbose=True, 
            test_size=0.2, undersampling_ratio=None, scaling=True,
            model_types=None, param_grids=None, bit_widths=None,
            cv_folds=5, scoring='f1', n_iterations=100):
        """
        Initialize the FHE model evaluator with configuration options.

        Parameters:
            data (DataFrame): Input dataset as pandas DataFrame (required).
            target_column (str): Name of the target column in the dataset (required).
            random_state (int): Random seed for reproducibility.
            n_jobs (int): Number of parallel jobs for computations.
            verbose (bool): Flag to enable verbose output.
            test_size (float): Proportion of the dataset to include in the test split.
            undersampling_ratio (float, optional): Ratio for undersampling the majority class.
            scaling (bool): Whether to apply StandardScaler to the features.
            model_types (list, optional): List of model types to evaluate (e.g., ['lr', 'rf', 'dt']).
            param_grids (dict, optional): Dictionary of parameter grids for each model type.
            bit_widths (list, optional): List of bit widths to evaluate for FHE models.
            cv_folds (int): Number of cross-validation folds.
            scoring (str): Scoring metric for model evaluation.
            n_iterations (int): Number of iterations for latency measurement.

        Attributes:
            data (DataFrame): The input dataset.
            X (array): The feature matrix.
            y (array): The target vector.
            X_train (array): Training features.
            X_test (array): Testing features.
            y_train (array): Training targets.
            y_test (array): Testing targets.
            scaler (StandardScaler): Scaler for feature normalization.
            undersampler (RandomUnderSampler): Undersampler for imbalanced data.
            best_params (dict): Best parameters found through grid search.
            evaluation_results (dict): Results of model evaluation.
        """
        # Validate required parameters
        if data is None:
            raise ValueError("Input data is required and cannot be None")
        if target_column is None:
            raise ValueError("Target column name is required and cannot be None")
            
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results = {}
        self.data_processed = False
        self.grid_search_complete = False
        self.evaluation_complete = False

        # Data configuration
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.undersampling_ratio = undersampling_ratio
        self.scaling = scaling

        # Model configuration
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_iterations = n_iterations

        # Define model type mappings for easier access
        self.model_mappings = {
            'lr': ('LogisticRegression', LogisticRegression, 'LogisticRegression'),
            'rf': ('RandomForest', RandomForestClassifier, 'RandomForestClassifier'),
            'dt': ('DecisionTree', DecisionTreeClassifier, 'DecisionTreeClassifier'),
            'xgb': ('XGBoost', xgb.XGBClassifier, 'XGBClassifier'),
            'mlp': ('MLP', MLPClassifier, 'MLPClassifier')
        }

        # Set default model types if not provided
        self.model_types = model_types or ['lr', 'rf', 'dt']

        # Set default parameter grids if not provided
        self.param_grids = param_grids or {
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

        self.bit_widths = bit_widths or [2, 3, 4, 6, 8]

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.undersampler = None
        self.best_params = {}
        self.evaluation_results = None

        self.process_data()

    def process_data(self):
        """
        Processes the input dataset by performing the following steps:
        1. Validates the presence of data and target column.
        2. Splits the dataset into features (X) and target (y).
        3. Scales the features using StandardScaler if scaling is enabled.
        4. Splits the data into training and testing sets.
        5. Creates an undersampler for handling class imbalance if an undersampling ratio is provided.
        Raises:
            ValueError: If `data` or `target_column` is not provided.
        Attributes:
            self.X (numpy.ndarray): Feature matrix after dropping the target column.
            self.y (numpy.ndarray): Target vector extracted from the dataset.
            self.scaler (StandardScaler): Scaler object used for feature scaling (if scaling is enabled).
            self.X_train (numpy.ndarray): Training feature matrix.
            self.X_test (numpy.ndarray): Testing feature matrix.
            self.y_train (numpy.ndarray): Training target vector.
            self.y_test (numpy.ndarray): Testing target vector.
            self.undersampler (RandomUnderSampler): Undersampler object for handling class imbalance.
            self.data_processed (bool): Flag indicating whether the data has been successfully processed.
        Returns:
            self: The instance of the class with processed data.
        """
        
        if self.data is None or self.target_column is None:
            raise ValueError("Data and target column must be provided")
        
        if self.verbose:
            print("Processing data...")
            print(f"Dataset shape: {self.data.shape}")
            print(f"Target distribution: {self.data[self.target_column].value_counts(normalize=True) * 100}")
        
        # Extract features and target
        self.X = self.data.drop(self.target_column, axis=1).values
        self.y = self.data[self.target_column].values
        
        # Scale features if requested
        if self.scaling:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, 
            stratify=self.y
        )
        
        # Create undersampler if ratio is provided
        if self.undersampling_ratio is not None:
            self.undersampler = RandomUnderSampler(
                sampling_strategy=self.undersampling_ratio, 
                random_state=self.random_state
            )
        else:
            self.undersampler = RandomUnderSampler(random_state=self.random_state)
        
        if self.verbose:
            print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        self.data_processed = True
        return self
    
    def run_full_pipeline(self):
        """
        Executes the full evaluation pipeline, including data processing, grid search for optimal 
        parameters, and model evaluation for both Fully Homomorphic Encryption (FHE) and plain text models.
        The pipeline performs the following steps:
        1. Processes the data if it has not already been processed.
        2. Conducts a grid search to find the best hyperparameters for each specified model type.
        3. Evaluates the performance of models on test data, comparing FHE and plain text implementations.
            dict: A dictionary containing the evaluation results, including a summary of the 
                  comparison between FHE and plain text models.        
        Returns:
            dict: Evaluation results
        """
        # Process data if not already done
        if not self.data_processed:
            self.process_data()
        
        # Run grid search
        if self.verbose:
            print("\nPerforming grid search for optimal parameters...")
        
        for model_type in self.model_types:
            if model_type not in self.param_grids:
                if self.verbose:
                    print(f"Skipping grid search for {model_type} - no param grid provided")
                continue
                
            if self.verbose:
                print(f"\nGrid search for {model_type.upper()}...")
                
            result = self.grid_search(
                model_type=model_type,
                param_grid=self.param_grids[model_type],
                x=self.X_train,
                y=self.y_train,
                undersampler=self.undersampler,
                cv=self.cv_folds,
                scoring=self.scoring
            )
            self.best_params[model_type] = result['best_params']
        
        self.grid_search_complete = True
        
        if self.verbose:
            print("\nEvaluating FHE vs plain text models...")
            
        self.evaluation_results = self.evaluate_models(
            model_types=self.model_types,
            x=self.X_test,
            y=self.y_test,
            best_params=self.best_params,
            bit_widths=self.bit_widths,
            undersampler=self.undersampler,
            cv_splits=self.cv_folds,
            n_iterations=self.n_iterations
        )
        
        self.evaluation_complete = True
        
        if self.verbose:
            print("\nFHE vs Plain Model Comparison Summary:")
            print(self.evaluation_results['summary']['comparison_table'])
        
        return self.evaluation_results
                
    
    def grid_search(self, model_type, param_grid, x, y, undersampler=None, cv=5, scoring='f1'):
        """
        Perform a grid search to find the optimal hyperparameters for a given model.
        Parameters:
            model_type (str): The type of model to evaluate. This should correspond to a valid model type 
                              that can be retrieved using the `_get_estimator` method.
            param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter 
                               settings to try as values.
            x (array-like): Feature matrix for training the model.
            y (array-like): Target vector for training the model.
            undersampler (imblearn.under_sampling.BaseUnderSampler, optional): An undersampling strategy 
                              to handle class imbalance. If None, a `RandomUnderSampler` with the 
                              instance's `random_state` will be used. Default is None.
            cv (int, optional): Number of cross-validation folds. Default is 5.
            scoring (str, optional): Scoring metric to evaluate the model performance. Default is 'f1'.
        Returns:
            dict: A dictionary containing the following keys:
                - 'best_params': The best hyperparameters found during the grid search.
                - 'best_score': The best score achieved with the best hyperparameters.
                - 'cv_results': Detailed cross-validation results as a dictionary.
                - 'grid_search': The fitted `GridSearchCV` object.
        Side Effects:
            Updates the `self.results` dictionary with the grid search results under the key 
            "{model_type}_grid_search".
        Notes:
            - The method prints progress and results if `self.verbose` is set to True.
            - The `n_jobs` parameter of `GridSearchCV` is set to `self.n_jobs` to control parallelism
        """
        if undersampler is None:
            undersampler = RandomUnderSampler(random_state=self.random_state)
        
        estimator = self._get_estimator(model_type)
        
        X_resampled, y_resampled = undersampler.fit_resample(x, y)
        
        if self.verbose:
            print(f"Performing grid search for {model_type.upper()} model...")
        
        grid_search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=scoring,
            verbose=1 if self.verbose else 0,
            n_jobs=self.n_jobs
        ).fit(X_resampled, y_resampled)
        
        if self.verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
        
        result = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'grid_search': grid_search
        }
        
        self.results[f"{model_type}_grid_search"] = result
        
        return result
    
    def _get_estimator(self, model_type):
        """
        Retrieve the appropriate scikit-learn estimator based on the model type.
        Args:
            model_type (str): The type of the model as a string. This should match
                one of the keys in `self.model_mappings`.
        Returns:
            sklearn.base.BaseEstimator: An instance of the scikit-learn estimator
                corresponding to the specified model type, initialized with the
                configured random state.
        Raises:
            ValueError: If the provided `model_type` is not supported or not found
                in `self.model_mappings`.
        """
        model_type = model_type.lower()
        if model_type not in self.model_mappings:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        _, estimator_class, _ = self.model_mappings[model_type]
        return estimator_class(random_state=self.random_state)
    
    def _get_fhe_estimator(self, model_type, **kwargs):
        """
        Retrieve the appropriate Fully Homomorphic Encryption (FHE) estimator class instance.
        This method dynamically imports and initializes an FHE estimator class based on the 
        provided model type. The mapping between model types and their corresponding FHE 
        classes is defined in `self.model_mappings`.
        Args:
            model_type (str): The type of the model for which the FHE estimator is required.
                              This should match a key in `self.model_mappings`.
            **kwargs: Additional keyword arguments to pass to the FHE estimator's constructor.
        Returns:
            object: An instance of the FHE estimator class corresponding to the given model type.
        Raises:
            ValueError: If the provided `model_type` is not supported or not found in 
                        `self.model_mappings`.
        Notes:
            - The `self.model_mappings` attribute is expected to be a dictionary where the keys 
              are model types (strings) and the values are tuples containing metadata, with the 
              third element being the name of the FHE class.
            - The FHE estimator classes are expected to be located in the `concrete.ml.sklearn` module.
        """
        model_type = model_type.lower()
        if model_type not in self.model_mappings:
            raise ValueError(f"Unsupported FHE model type: {model_type}")
        
        _, _, fhe_class_name = self.model_mappings[model_type]
        
        from importlib import import_module
        module = import_module('concrete.ml.sklearn')
        fhe_class = getattr(module, fhe_class_name)
        
        return fhe_class(random_state=self.random_state, **kwargs)
            
    def evaluate_models(self, model_types, x, y, best_params, bit_widths, 
                      undersampler=None, cv_splits=5, n_iterations=100):
        """
        Evaluate and compare plain models vs FHE (Fully Homomorphic Encryption) models 
                    across multiple model types.
                    Parameters:
                        model_types (list): A list of model types to evaluate (e.g., ['logistic', 'svm']).
                        x (array-like): Feature matrix for the dataset.
                        y (array-like): Target labels for the dataset.
                        best_params (dict): A dictionary containing the best hyperparameters for each model type.
                        bit_widths (list): A list of bit widths to evaluate for FHE models.
                        undersampler (object, optional): An undersampler instance for balancing the dataset. 
                                                         Defaults to RandomUnderSampler.
                        cv_splits (int, optional): Number of cross-validation splits. Defaults to 5.
                        n_iterations (int, optional): Number of iterations for evaluation. Defaults to 100.
                    Returns:
                        dict: A dictionary containing the evaluation results with the following keys:
                            - 'model_comparison': Detailed comparison of plain and FHE models for each model type.
                            - 'summary': A summary of the evaluation results.
                            - 'best_models': Information about the best-performing FHE models for each model type.
                    Notes:
                        - The method evaluates both plain and FHE models using cross-validation.
                        - It compares the performance of plain and FHE models in terms of accuracy, latency, 
                          and trade-offs.
                        - Figures and visualizations are generated for each model type to aid in analysis.       
        """
        # Setup defaults
        if undersampler is None:
            undersampler = RandomUnderSampler(random_state=self.random_state)
        
        kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            
        # Initialize results structure
        evaluation_results = {
            'model_comparison': {},
            'summary': {},
            'best_models': {}
        }
        
        # Evaluate each model type
        for model_type in model_types:
            if self.verbose:
                print(f"\n{'-'*70}")
                print(f"Evaluating {model_type.upper()} models")
                print(f"{'-'*70}")
            
            params = best_params.get(model_type, {})
            
            plain_model = self._get_estimator(model_type)
            plain_model.set_params(**params)
            plain_results = self._evaluate_model(plain_model, x, y, undersampler, kf, n_iterations)
            
            cipher_results = self._evaluate_fhe_models(
                model_type, params, bit_widths, x, y, undersampler, kf, n_iterations
            )
            
            comparison = self._compare_models(plain_results, cipher_results)
            
            figures = self._create_model_figures(plain_results, cipher_results, comparison)
            
            evaluation_results['model_comparison'][model_type] = {
                'plain_results': plain_results,
                'cipher_results': cipher_results,
                'comparison': comparison,
                'figures': figures
            }
            
            best_idx = comparison['best_tradeoff_idx']
            best_bit_width = bit_widths[best_idx]
            
            evaluation_results['best_models'][model_type] = {
                'bit_width': best_bit_width,
                'accuracy': cipher_results['accuracies'][best_idx],
                'latency': cipher_results['latencies'][best_idx],
                'overhead': comparison['latency_ratios'][best_idx],
                'plain_accuracy': plain_results['avg_accuracy'],
                'plain_latency': plain_results['avg_latency']
            }
        
        self._generate_summary(evaluation_results)
        
        self.results['evaluation'] = evaluation_results
        
        return evaluation_results
    
    def _evaluate_model(self, model, x, y, undersampler, kf, n_iterations):
        """
        This method performs cross-validation on the given model using the provided
        dataset, undersampler, and cross-validation strategy. It calculates various
        performance metrics for each fold and aggregates the results.
        Args:
            model (object): The machine learning model to evaluate. Must implement
                `fit`, `predict`, and optionally `predict_proba` methods.
            x (array-like): Feature matrix for the dataset.
            y (array-like): Target labels for the dataset.
            undersampler (object): An undersampling strategy object that implements
                the `fit_resample` method to handle class imbalance.
            kf (object): A cross-validation splitting strategy object (e.g., KFold).
            n_iterations (int): Number of iterations to measure latency.
        Returns:
            dict: A dictionary containing the evaluation results, including:
                - 'fold_results': A list of dictionaries with metrics for each fold.
                - 'metrics': A dictionary of lists for each metric across folds.
                - 'avg_<metric_name>': The average value of each metric across folds.
        Metrics:
            - accuracy: Classification accuracy.
            - f1: F1 score.
            - precision: Precision score.
            - recall: Recall score.
            - roc_auc: Area under the ROC curve.
            - latency: Inference latency in milliseconds.
            - training_time: Model training time in seconds.
        Notes:
            - If the model does not implement `predict_proba`, the predicted labels
              are used directly for metrics that require probabilities.
            - The method prints fold-wise results and a summary if `self.verbose` is True.
        """
        results = {
            'fold_results': [],
            'metrics': {
                'accuracy': [],
                'f1': [],
                'precision': [],
                'recall': [],
                'roc_auc': [],
                'latency': [],
                'training_time': []
            }
        }
        
        for fold_idx, (train_index, test_index) in enumerate(kf.split(x, y)):
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            
            X_train_resampled, Y_train_resampled = undersampler.fit_resample(X_train, Y_train)
            
            start_time = time.time()
            model.fit(X_train_resampled, Y_train_resampled)
            training_time = time.time() - start_time
            results['metrics']['training_time'].append(training_time)
            
            latency = self._measure_latency(model, X_test, n_iterations)
            results['metrics']['latency'].append(latency)
            
            Y_pred = model.predict(X_test)
            Y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else Y_pred
            
            metrics = self._calculate_metrics(Y_test, Y_pred, Y_pred_proba)
            for metric_name, value in metrics.items():
                results['metrics'][metric_name].append(value)
            
            results['fold_results'].append({
                'fold': fold_idx + 1,
                **metrics,
                'latency_ms': latency,
                'training_time_s': training_time,
                'test_samples': len(X_test)
            })
            
            if self.verbose:
                print(f"Fold {fold_idx + 1}: Accuracy={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1']:.4f}, Latency={latency:.4f}ms")
        
        for metric_name in results['metrics']:
            results[f'avg_{metric_name}'] = np.mean(results['metrics'][metric_name])
        
        if self.verbose:
            self._print_model_summary(results)
        
        return results
    
    def _evaluate_fhe_models(self, model_type, params, bit_widths, x, y, undersampler, kf, n_iterations):
        """
        Evaluate Fully Homomorphic Encryption (FHE) models with different bit widths.
        This method evaluates the performance of FHE models by iterating over a range of bit widths.
        For each bit width, it performs k-fold cross-validation, measures latency, and computes
        various performance metrics such as accuracy, F1-score, precision, recall, and ROC AUC.
        Args:
            model_type (str): The type of the model to be evaluated.
            params (dict): Parameters to configure the FHE model.
            bit_widths (list[int]): A list of bit widths to evaluate the model with.
            x (np.ndarray): Feature matrix for the dataset.
            y (np.ndarray): Target labels for the dataset.
            undersampler (object): An undersampling strategy to handle class imbalance.
            kf (object): A KFold cross-validator instance for splitting the data.
            n_iterations (int): Number of iterations for latency measurement.
        Returns:
            dict: A dictionary containing the evaluation results:
                - 'bit_widths': List of evaluated bit widths.
                - 'accuracies': Average accuracy for each bit width.
                - 'latencies': Average latency for each bit width.
                - 'detailed_results': Detailed results for each bit width, including metrics
                  for each fold, average metrics, and timing information.
        Notes:
            - The method uses k-fold cross-validation to evaluate the model's performance.
            - Latency is measured using the `_measure_latency` method.
            - Metrics are calculated using the `_calculate_metrics` method.
            - If `self.verbose` is True, progress and summaries are printed during execution.        """
        results = {
            'bit_widths': bit_widths,
            'accuracies': [],
            'latencies': [],
            'detailed_results': []
        }
        
        for bit_width in bit_widths:
            fhe_model = self._get_fhe_estimator(model_type, n_bits=bit_width, **params)
            
            bit_width_results = {
                'bit_width': bit_width,
                'fold_results': [],
                'metrics': {
                    'accuracy': [],
                    'f1': [],
                    'precision': [],
                    'recall': [],
                    'roc_auc': [],
                    'latency': [],
                    'training_time': []
                }
            }
            
            for fold_idx, (train_index, test_index) in enumerate(kf.split(x, y)):
                X_train, X_test = x[train_index], x[test_index]
                Y_train, Y_test = y[train_index], y[test_index]
                
                X_train_resampled, Y_train_resampled = undersampler.fit_resample(X_train, Y_train)
                
                start_time = time.time()
                fhe_model.fit(X_train_resampled, Y_train_resampled)
                training_time = time.time() - start_time
                bit_width_results['metrics']['training_time'].append(training_time)
                
                latency = self._measure_latency(fhe_model, X_test, n_iterations)
                bit_width_results['metrics']['latency'].append(latency)
                
                Y_pred = fhe_model.predict(X_test)
                Y_pred_proba = fhe_model.predict_proba(X_test)[:, 1] if hasattr(fhe_model, "predict_proba") else Y_pred
                
                metrics = self._calculate_metrics(Y_test, Y_pred, Y_pred_proba)
                for metric_name, value in metrics.items():
                    bit_width_results['metrics'][metric_name].append(value)
                
                bit_width_results['fold_results'].append({
                    'fold': fold_idx + 1,
                    **metrics,
                    'latency_ms': latency,
                    'training_time_s': training_time,
                    'test_samples': len(X_test)
                })
                
                if self.verbose:
                    print(f"Bit Width {bit_width}, Fold {fold_idx + 1}: "
                          f"Accuracy={metrics['accuracy']:.4f}, Latency={latency:.4f}ms")
            
            for metric_name in bit_width_results['metrics']:
                bit_width_results[f'avg_{metric_name}'] = np.mean(bit_width_results['metrics'][metric_name])
            
            results['accuracies'].append(bit_width_results['avg_accuracy'])
            results['latencies'].append(bit_width_results['avg_latency'])
            results['detailed_results'].append(bit_width_results)
            
            if self.verbose:
                self._print_bit_width_summary(bit_width_results)
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate classification metrics for a given set of true labels, predicted labels, 
        and predicted probabilities.
        Args:
            y_true (array-like): True class labels.
            y_pred (array-like): Predicted class labels.
            y_pred_proba (array-like): Predicted probabilities for the positive class.
        Returns:
            dict: A dictionary containing the following classification metrics:
                - 'accuracy': Accuracy of the predictions.
                - 'f1': F1 score of the predictions.
                - 'precision': Precision of the predictions.
                - 'recall': Recall of the predictions.
                - 'roc_auc': ROC AUC score of the predictions. If the ROC AUC score cannot 
                  be calculated, it will be set to NaN.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = np.nan
            
        return metrics
    
    def _measure_latency(self, model, X_test, n_iterations=100):
        """
        Measure the average inference latency of a given model.

        This method calculates the average time taken for the model to perform
        inference on the provided test data over a specified number of iterations.
        The result is returned in milliseconds.

        Args:
            model: The machine learning model to evaluate. It must have a `predict` method.
            X_test: The input data for testing the model's inference latency.
            n_iterations (int, optional): The number of iterations to perform the latency
                measurement. Default is 100.

        Returns:
            float: The average inference latency in milliseconds.
        """
        inference_times = []
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X_test)
            inference_times.append(time.time() - start_time)
        return np.mean(inference_times) * 1000  # Convert to ms
    
    def _compare_models(self, plain_results, cipher_results):
        """
        Compare the performance of plain and cipher models to evaluate tradeoffs.
        This method calculates the differences in accuracy, the ratios of latency,
        and a tradeoff score for each configuration of the cipher model compared
        to the plain model. It identifies the configuration with the best tradeoff
        based on the calculated scores.
        Args:
            plain_results (dict): A dictionary containing the results of the plain model.
                Expected keys:
                    - 'avg_accuracy' (float): The average accuracy of the plain model.
                    - 'avg_latency' (float): The average latency of the plain model.
            cipher_results (dict): A dictionary containing the results of the cipher model.
                Expected keys:
                    - 'bit_widths' (list of int): The bit widths used in the cipher model.
                    - 'accuracies' (list of float): The accuracies of the cipher model for each bit width.
                    - 'latencies' (list of float): The latencies of the cipher model for each bit width.
        Returns:
            dict: A dictionary containing the comparison results. Keys include:
                - 'bit_widths' (list of int): The bit widths used in the cipher model.
                - 'accuracy_diffs' (list of float): The differences in accuracy between the cipher
                  and plain models for each bit width.
                - 'latency_ratios' (list of float): The ratios of latency between the cipher and
                  plain models for each bit width.
                - 'tradeoff_scores' (list of float): The calculated tradeoff scores for each bit width.
                - 'best_tradeoff_idx' (int): The index of the configuration with the best tradeoff score.
                - 'best_bit_width' (int): The bit width corresponding to the best tradeoff score.
        """
        comparison = {
            'bit_widths': cipher_results['bit_widths'],
            'accuracy_diffs': [],
            'latency_ratios': [],
            'tradeoff_scores': []
        }
        
        # Calculate differences and ratios
        plain_accuracy = plain_results['avg_accuracy']
        plain_latency = plain_results['avg_latency']
        
        for i, cipher_accuracy in enumerate(cipher_results['accuracies']):
            cipher_latency = cipher_results['latencies'][i]
            
            # Calculate metrics
            accuracy_diff = cipher_accuracy - plain_accuracy
            latency_ratio = cipher_latency / plain_latency
            
            # Tradeoff score (lower is better)
            accuracy_weight = 1.0
            latency_weight = 0.5
            tradeoff_score = (latency_weight * latency_ratio) - (accuracy_weight * accuracy_diff)
            
            comparison['accuracy_diffs'].append(accuracy_diff)
            comparison['latency_ratios'].append(latency_ratio)
            comparison['tradeoff_scores'].append(tradeoff_score)
        
        # Find best tradeoff
        comparison['best_tradeoff_idx'] = np.argmin(comparison['tradeoff_scores'])
        comparison['best_bit_width'] = cipher_results['bit_widths'][comparison['best_tradeoff_idx']]
        
        return comparison
    
    def _create_model_figures(self, plain_results, cipher_results, comparison):
        """
        Create visualization figures for model comparison.
        This method generates three plots to visualize and compare the performance
        of a Fully Homomorphic Encryption (FHE) model against a plain model:
        - Accuracy vs Bit Width
        - Latency vs Bit Width (Log Scale)
        - Accuracy-Latency Tradeoff by Bit Width
        Args:
            plain_results (dict): A dictionary containing results for the plain model.
                Expected keys:
                    - 'avg_accuracy' (float): Average accuracy of the plain model.
                    - 'avg_latency' (float): Average latency of the plain model in milliseconds.
            cipher_results (dict): A dictionary containing results for the FHE model.
                Expected keys:
                    - 'bit_widths' (list of int): List of bit widths used in the FHE model.
                    - 'accuracies' (list of float): Corresponding accuracies for each bit width.
                    - 'latencies' (list of float): Corresponding latencies (in milliseconds) for each bit width.
            comparison (dict): A dictionary containing tradeoff analysis results.
                Expected keys:
                    - 'tradeoff_scores' (list of float): Tradeoff scores for each bit width.
                    - 'best_tradeoff_idx' (int): Index of the bit width with the best tradeoff score.
        Returns:
            dict: A dictionary containing the generated matplotlib figures.
                Keys:
                    - 'accuracy_fig': Figure for Accuracy vs Bit Width.
                    - 'latency_fig': Figure for Latency vs Bit Width (Log Scale).
                    - 'tradeoff_fig': Figure for Accuracy-Latency Tradeoff by Bit Width.
        """
        bit_widths = cipher_results['bit_widths']
        
        # Create accuracy comparison plot
        acc_fig, acc_ax = plt.subplots(figsize=(10, 6))
        acc_ax.plot(bit_widths, cipher_results['accuracies'], 'o-', label='FHE Model')
        acc_ax.axhline(y=plain_results['avg_accuracy'], color='r', linestyle='--', 
                     label='Plain Model')
        acc_ax.set_xlabel('Bit Width')
        acc_ax.set_ylabel('Accuracy')
        acc_ax.set_title('Accuracy vs Bit Width')
        acc_ax.grid(True, alpha=0.3)
        acc_ax.legend()
        
        # Create latency comparison plot (log scale)
        lat_fig, lat_ax = plt.subplots(figsize=(10, 6))
        lat_ax.plot(bit_widths, cipher_results['latencies'], 'o-', label='FHE Model')
        lat_ax.axhline(y=plain_results['avg_latency'], color='r', linestyle='--', 
                     label='Plain Model')
        lat_ax.set_xlabel('Bit Width')
        lat_ax.set_ylabel('Latency (ms)')
        lat_ax.set_title('Latency vs Bit Width (Log Scale)')
        lat_ax.set_yscale('log')
        lat_ax.grid(True, alpha=0.3)
        lat_ax.legend()
        
        # Create tradeoff plot
        trade_fig, trade_ax = plt.subplots(figsize=(10, 6))
        trade_ax.plot(bit_widths, comparison['tradeoff_scores'], 'o-')
        best_idx = comparison['best_tradeoff_idx']
        trade_ax.plot(bit_widths[best_idx], comparison['tradeoff_scores'][best_idx], 'ro', 
                    markersize=10, label=f'Best Tradeoff: {bit_widths[best_idx]} bits')
        trade_ax.set_xlabel('Bit Width')
        trade_ax.set_ylabel('Tradeoff Score (lower is better)')
        trade_ax.set_title('Accuracy-Latency Tradeoff by Bit Width')
        trade_ax.grid(True, alpha=0.3)
        trade_ax.legend()
        
        return {
            'accuracy_fig': acc_fig,
            'latency_fig': lat_fig, 
            'tradeoff_fig': trade_fig
        }
    
    def _print_model_summary(self, results):
        """
        Print a summary of model evaluation results.

        Args:
            results (dict): A dictionary containing the evaluation metrics of the model. 
                            Expected keys include:
                            - 'avg_accuracy' (float): The average accuracy of the model.
                            - 'avg_f1' (float): The average F1 score of the model.
                            - 'avg_precision' (float): The average precision of the model.
                            - 'avg_recall' (float): The average recall of the model.
                            - 'avg_roc_auc' (float): The average ROC AUC score of the model.
                            - 'avg_latency' (float): The average inference latency in milliseconds.
                            - 'avg_training_time' (float): The average training time in seconds.

        Returns:
            None
        """
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print(f"Average Accuracy: {results['avg_accuracy']:.4f}")
        print(f"Average F1 Score: {results['avg_f1']:.4f}")
        print(f"Average Precision: {results['avg_precision']:.4f}")
        print(f"Average Recall: {results['avg_recall']:.4f}")
        print(f"Average ROC AUC: {results['avg_roc_auc']:.4f}")
        print(f"Average Inference Latency: {results['avg_latency']:.4f} ms")
        print(f"Average Training Time: {results['avg_training_time']:.4f} s")
        print("="*50 + "\n")
    
    def _print_bit_width_summary(self, results):
        """
        Print a summary of evaluation results for a specific bit-width.

        Args:
            results (dict): A dictionary containing evaluation metrics for a given bit-width.
                Expected keys in the dictionary:
                    - 'bit_width' (int): The bit-width value.
                    - 'avg_accuracy' (float): The average accuracy of the model.
                    - 'avg_f1' (float): The average F1 score of the model.
                    - 'avg_latency' (float): The average latency in milliseconds.

        Returns:
            None
        """
        print("\n" + "="*50)
        print(f"Bit-Width: {results['bit_width']}")
        print(f"Average Accuracy: {results['avg_accuracy']:.4f}")
        print(f"Average F1 Score: {results['avg_f1']:.4f}")
        print(f"Average Latency: {results['avg_latency']:.4f} ms")
        print("="*50 + "\n")
    
    def _generate_summary(self, evaluation_results):
        """
        Generate a summary of evaluation metrics across all models.
        This method processes the evaluation results to identify the best models
        based on accuracy, latency, and overhead. It also creates a comparison
        table summarizing key metrics for each model.
        Args:
            evaluation_results (dict): A dictionary containing evaluation results
                with the following structure:
                {
                    'best_models': {
                        <model_name>: {
                            'plain_accuracy': float,
                            'accuracy': float,
                            'plain_latency': float,
                            'latency': float,
                            'overhead': float,
                            'bit_width': int
                        },
                        ...
        Updates:
            evaluation_results['summary'] (dict): A dictionary containing the summary
                of evaluation metrics with the following keys:
                - 'best_accuracy_model': The model with the highest FHE accuracy.
                - 'best_latency_model': The model with the lowest FHE latency.
                - 'best_overhead_model': The model with the lowest latency overhead.
                - 'comparison_table': A pandas DataFrame summarizing metrics for all models.
        Prints:
            If `self.verbose` is True, prints a detailed summary of the best models
            and the comparison table.
        """
        model_types = list(evaluation_results['best_models'].keys())
        best_models = evaluation_results['best_models']
        
        summary = {
            'best_accuracy_model': max(model_types, key=lambda m: best_models[m]['accuracy']),
            'best_latency_model': min(model_types, key=lambda m: best_models[m]['latency']),
            'best_overhead_model': min(model_types, key=lambda m: best_models[m]['overhead']),
            'comparison_table': pd.DataFrame({
                model: {
                    'Plain Accuracy': best_models[model]['plain_accuracy'],
                    'FHE Accuracy': best_models[model]['accuracy'],
                    'Plain Latency (ms)': best_models[model]['plain_latency'],
                    'FHE Latency (ms)': best_models[model]['latency'],
                    'Latency Overhead': best_models[model]['overhead'],
                    'Best Bit Width': best_models[model]['bit_width']
                } for model in model_types
            })
        }
        
        evaluation_results['summary'] = summary
        
        if self.verbose:
            print("\n" + "="*80)
            print("OVERALL FHE MODEL COMPARISON SUMMARY")
            print("="*80)
            print(f"\nBest model by accuracy: {summary['best_accuracy_model'].upper()} " + 
                  f"({best_models[summary['best_accuracy_model']]['accuracy']:.4f})")
            print(f"Best model by latency: {summary['best_latency_model'].upper()} " + 
                  f"({best_models[summary['best_latency_model']]['latency']:.4f} ms)")
            print(f"Best model by overhead: {summary['best_overhead_model'].upper()} " + 
                  f"({best_models[summary['best_overhead_model']]['overhead']:.2f}x)")
            
            print("\nModel Comparison Table:")
            print(summary['comparison_table'])
            print("="*80)
    
    def visualize_model_comparison(self):
        """
        Create visualizations comparing the performance of all evaluated models.
        This method generates visualizations that compare the accuracy, latency, 
        and other metrics of models evaluated in the system. It requires that 
        the `evaluate_models` method has been run beforehand to populate the 
        `self.results` dictionary with evaluation data.
        Raises:
            ValueError: If no evaluation results are found in `self.results`.
        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]: 
                A tuple containing two matplotlib figures:
                - The first figure compares the plain and FHE accuracies of the models.
                - The second figure compares the plain and FHE latencies, along with 
                  other metrics such as overhead and bit width.
        """
        if 'evaluation' not in self.results:
            raise ValueError("No evaluation results found. Run evaluate_models first.")
        
        evaluation = self.results['evaluation']
        best_models = evaluation['best_models']
        model_types = list(best_models.keys())
        
        models = [model_type.upper() for model_type in model_types]
        plain_accuracies = [best_models[model]['plain_accuracy'] for model in model_types]
        fhe_accuracies = [best_models[model]['accuracy'] for model in model_types]
        plain_latencies = [best_models[model]['plain_latency'] for model in model_types]
        fhe_latencies = [best_models[model]['latency'] for model in model_types]
        overheads = [best_models[model]['overhead'] for model in model_types]
        bit_widths = [best_models[model]['bit_width'] for model in model_types]
        
        acc_fig, latency_fig = self._create_comparison_figures(
            models, plain_accuracies, fhe_accuracies, 
            plain_latencies, fhe_latencies, 
            overheads, bit_widths
        )
        
        return acc_fig, latency_fig
    
    
    
    def _create_comparison_figures(self, models, plain_accs, fhe_accs, 
                                plain_lats, fhe_lats, overheads, bit_widths):
        """
        Helper method to create comparison figures for model accuracy and latency.
        This method generates two bar plots:
        1. A comparison of the accuracy between plain and FHE (Fully Homomorphic Encryption) models.
        2. A comparison of the latency between plain and FHE models, with latency displayed on a logarithmic scale.
        Parameters:
            models (list of str): A list of model names/types.
            plain_accs (list of float): A list of accuracy values for the plain models.
            fhe_accs (list of float): A list of accuracy values for the FHE models.
            plain_lats (list of float): A list of latency values (in milliseconds) for the plain models.
            fhe_lats (list of float): A list of latency values (in milliseconds) for the FHE models.
            overheads (list of float): A list of latency overhead factors for the FHE models compared to the plain models.
            bit_widths (list of int): A list of bit-width values for the FHE models.
        Returns:
            tuple: A tuple containing two matplotlib figure objects:
                - acc_fig: The figure for accuracy comparison.
                - lat_fig: The figure for latency comparison. """
        acc_fig, ax1 = plt.subplots(figsize=(12, 8))
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, plain_accs, width, label='Plain Model')
        ax1.bar(x + width/2, fhe_accs, width, label='FHE Model')
        
        # Add bit width annotations
        for i, (acc, bw) in enumerate(zip(fhe_accs, bit_widths)):
            ax1.annotate(f"{bw} bits", 
                         xy=(x[i] + width/2, acc), 
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Plain vs FHE Model Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        lat_fig, ax2 = plt.subplots(figsize=(12, 8))
        
        ax2.bar(x - width/2, plain_lats, width, label='Plain Model')
        ax2.bar(x + width/2, fhe_lats, width, label='FHE Model')
        
        # Add overhead annotations
        for i, (lat, ovr) in enumerate(zip(fhe_lats, overheads)):
            ax2.annotate(f"{ovr:.1f}x", 
                         xy=(x[i] + width/2, lat), 
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Plain vs FHE Model Latency Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return acc_fig, lat_fig
    

    def generate_report(self, report_file='fhe_evaluation_report.txt', output_folder='results', save_figures=True):
        """      
        This method generates a detailed report summarizing the evaluation of Fully Homomorphic Encryption (FHE) models 
        compared to plain models. It includes visualizations of accuracy, latency, and tradeoff metrics, and saves the 
        results to specified files and folders.
            report_file (str): The name of the report file to save. Default is 'fhe_evaluation_report.txt'.
            output_folder (str): The folder where the report and figures will be saved. Default is 'results'.
            save_figures (bool): Whether to save the generated figures to disk. Default is True.
            tuple: A tuple containing:
                - acc_fig (matplotlib.figure.Figure): The figure for overall accuracy comparison.
                - latency_fig (matplotlib.figure.Figure): The figure for overall latency comparison.
                - model_figures (dict): A dictionary containing paths to model-specific figures, including:
                    - 'accuracy': Path to the accuracy figure for each model type.
                    - 'latency': Path to the latency figure for each model type.
                    - 'tradeoff': Path to the tradeoff figure for each model type.
                    - 'bit_width_impact': Path to the bit-width impact figure for each model type.
        Raises:
            ValueError: If the evaluation has not been completed before calling this method.
        Notes:
            - The method assumes that the evaluation has been completed and results are stored in the instance variables.
            - Figures are saved in the specified output folder with descriptive filenames.
            - The report includes dataset information, model parameters, evaluation results, and generated figure details.
        """
        if not self.evaluation_complete:
            raise ValueError("Must run evaluation before generating report")
        
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        report_path = os.path.join(output_folder, report_file)
        
        acc_fig, latency_fig = self.visualize_model_comparison()
        
        model_figures = {}
        
        if save_figures:
            acc_fig_path = os.path.join(output_folder, 'overall_accuracy_comparison.png')
            latency_fig_path = os.path.join(output_folder, 'overall_latency_comparison.png')
            acc_fig.savefig(acc_fig_path)
            latency_fig.savefig(latency_fig_path)
            
            for model_type in self.model_types:
                if 'model_comparison' in self.evaluation_results and model_type in self.evaluation_results['model_comparison']:
                    model_data = self.evaluation_results['model_comparison'][model_type]
                    
                    # accuracy 
                    if 'figures' in model_data and 'accuracy_fig' in model_data['figures']:
                        acc_path = os.path.join(output_folder, f'{model_type}_accuracy.png')
                        model_data['figures']['accuracy_fig'].savefig(acc_path)
                        
                    # latency
                    if 'figures' in model_data and 'latency_fig' in model_data['figures']:
                        lat_path = os.path.join(output_folder, f'{model_type}_latency.png')
                        model_data['figures']['latency_fig'].savefig(lat_path)
                        
                    # tradeoff
                    if 'figures' in model_data and 'tradeoff_fig' in model_data['figures']:
                        trade_path = os.path.join(output_folder, f'{model_type}_tradeoff.png')
                        model_data['figures']['tradeoff_fig'].savefig(trade_path)
                    
                    bit_width_impact = model_data['comparison']
                    accuracies = model_data['cipher_results']['accuracies']
                    latencies = model_data['cipher_results']['latencies']
                    best_idx = bit_width_impact['best_tradeoff_idx']
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(self.bit_widths, accuracies, 'o-', label='Accuracy')
                    plt.axhline(
                        y=model_data['plain_results']['avg_accuracy'],
                        color='r', linestyle='--', label='Plain Model Accuracy'
                    )
                    plt.scatter(
                        self.bit_widths[best_idx],
                        accuracies[best_idx],
                        color='red', s=100, zorder=5, label='Best Tradeoff'
                    )
                    plt.xlabel('Bit Width')
                    plt.ylabel('Accuracy')
                    plt.title(f'{model_type.upper()} FHE Accuracy by Bit Width')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    impact_path = os.path.join(output_folder, f'{model_type}_bit_width_impact.png')
                    plt.savefig(impact_path)
                    plt.close()
                    
                    model_figures[model_type] = {
                        'accuracy': acc_path if 'accuracy_fig' in model_data['figures'] else None,
                        'latency': lat_path if 'latency_fig' in model_data['figures'] else None,
                        'tradeoff': trade_path if 'tradeoff_fig' in model_data['figures'] else None,
                        'bit_width_impact': impact_path
                    }
            
            if self.verbose:
                print(f"Saved visualization figures to folder: '{output_folder}'")
        
        if self.verbose:
            print(f"\nWriting summary report to {report_path}...")
            
        with open(report_path, 'w') as f:
            f.write("FHE vs Plain Model Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            
            if self.data is not None:
                f.write(f"Dataset Shape: {self.data.shape}\n")
                if self.target_column:
                    target_counts = self.data[self.target_column].value_counts()
                    f.write(f"Target Distribution: {target_counts}\n")
                    f.write(f"Target Ratios: {target_counts / len(self.data) * 100:.4f}%\n")
            f.write("\n")
            
            f.write("Model Parameters:\n")
            for model_type, params in self.best_params.items():
                f.write(f"{model_type.upper()}: {params}\n")
            f.write("\n")
            
            f.write("FHE Evaluation Results:\n")
            f.write(f"Best model by accuracy: "
                    f"{self.evaluation_results['summary']['best_accuracy_model'].upper()}\n")
            f.write(f"Best model by latency: "
                    f"{self.evaluation_results['summary']['best_latency_model'].upper()}\n")
            f.write(f"Best model by overhead: "
                    f"{self.evaluation_results['summary']['best_overhead_model'].upper()}\n\n")
            
            f.write("Model Comparison Table:\n")
            f.write(self.evaluation_results['summary']['comparison_table'].to_string())
            f.write("\n\n")
            
            if save_figures:
                f.write("Generated Figures:\n")
                f.write(f"Overall accuracy comparison: {os.path.basename(acc_fig_path)}\n")
                f.write(f"Overall latency comparison: {os.path.basename(latency_fig_path)}\n\n")
                
                f.write("Model-specific figures:\n")
                for model_type, paths in model_figures.items():
                    f.write(f"{model_type.upper()}:\n")
                    for fig_type, path in paths.items():
                        if path:
                            f.write(f"  {fig_type}: {os.path.basename(path)}\n")
                    f.write("\n")
            
            f.write("FHE Bit Width Impact:\n")
            for model_type in self.model_types:
                bit_width_impact = self.evaluation_results['model_comparison'][model_type]['comparison']
                f.write(f"{model_type.upper()} - Best bit width: {bit_width_impact['best_bit_width']}\n")
                accuracies = self.evaluation_results['model_comparison'][model_type]['cipher_results']['accuracies']
                latencies = self.evaluation_results['model_comparison'][model_type]['cipher_results']['latencies']
                
                for i, bw in enumerate(self.bit_widths):
                    f.write(f"  {bw} bits: Accuracy={accuracies[i]:.4f}, "
                            f"Latency={latencies[i]:.2f}ms, "
                            f"Overhead={bit_width_impact['latency_ratios'][i]:.2f}x\n")
                f.write("\n")
        
        if self.verbose:
            print(f"Evaluation complete! Results saved to '{report_path}'")
        
        return acc_fig, latency_fig, model_figures
    
