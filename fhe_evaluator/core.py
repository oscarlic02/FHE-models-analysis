"""
FHE Model Evaluator Core Module

This module contains the main FHEModelEvaluator class, which provides a comprehensive
framework for evaluating and comparing Fully Homomorphic Encryption (FHE) models
against traditional machine learning models.
"""

import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from .config import ModelConfig, DEFAULT_PARAM_GRIDS, DEFAULT_BIT_WIDTHS
from .utils import ModelCreator, DataProcessor, MetricsCalculator, LatencyMeasurer
from .exceptions import (
    FHEEvaluatorError, 
    DataProcessingError, 
    ModelEvaluationError, 
    VisualizationError
)


class FHEModelEvaluator:
    """
    A comprehensive framework for evaluating Fully Homomorphic Encryption (FHE) models
    and comparing them against traditional machine learning models.
    
    This class provides methods for hyperparameter tuning, model evaluation, visualization,
    and Pareto optimality analysis of the accuracy-latency trade-offs between FHE and
    plain text models.
    
    Attributes:
        config (ModelConfig): Configuration object containing all model and evaluation settings
        data_processor (DataProcessor): Handles data preprocessing and splitting
        model_creator (ModelCreator): Creates model instances for evaluation
        metrics_calculator (MetricsCalculator): Calculates performance metrics
        latency_measurer (LatencyMeasurer): Measures model inference latency
        results (dict): Stores all evaluation results and cached data
        
    Example:
        >>> import pandas as pd
        >>> from fhe_evaluator import FHEModelEvaluator
        >>> 
        >>> # Load your dataset
        >>> data = pd.read_csv('creditcard.csv')
        >>> 
        >>> # Initialize evaluator
        >>> evaluator = FHEModelEvaluator(
        ...     data=data,
        ...     target_column='Class',
        ...     model_types=['lr', 'rf', 'dt'],
        ...     bit_widths=[2, 4, 6, 8]
        ... )
        >>> 
        >>> # Run full evaluation pipeline
        >>> results = evaluator.run_full_pipeline()
        >>> 
        >>> # Generate Pareto analysis
        >>> pareto_results = evaluator.find_pareto_optimal_solutions()
        >>> 
        >>> # Get recommendations
        >>> recommendations = evaluator.get_pareto_recommendations('balanced')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_types: Optional[List[str]] = None,
        param_grids: Optional[Dict[str, Dict]] = None,
        bit_widths: Optional[List[int]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
        test_size: float = 0.2,
        undersampling_ratio: Optional[float] = None,
        scaling: bool = True,
        cv_folds: int = 5,
        scoring: str = 'f1',
        n_iterations: int = 100
    ):
        """
        Initialize the FHE Model Evaluator.
        
        Args:
            data: Input dataset as pandas DataFrame
            target_column: Name of the target column in the dataset
            model_types: List of model types to evaluate (e.g., ['lr', 'rf', 'dt'])
            param_grids: Dictionary of parameter grids for each model type
            bit_widths: List of bit widths to evaluate for FHE models
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for computations
            verbose: Flag to enable verbose output
            test_size: Proportion of the dataset to include in the test split
            undersampling_ratio: Ratio for undersampling the majority class
            scaling: Whether to apply StandardScaler to the features
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            n_iterations: Number of iterations for latency measurement
            
        Raises:
            DataProcessingError: If data or target_column is invalid
            FHEEvaluatorError: If configuration parameters are invalid
        """
        # Validate required parameters
        if data is None or data.empty:
            raise DataProcessingError("Input data is required and cannot be None or empty")
        if target_column is None or target_column not in data.columns:
            raise DataProcessingError(f"Target column '{target_column}' not found in data")
        
        # Initialize configuration
        self.config = ModelConfig(
            model_types=model_types or ['lr', 'rf', 'dt'],
            param_grids=param_grids or DEFAULT_PARAM_GRIDS,
            bit_widths=bit_widths or DEFAULT_BIT_WIDTHS,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            test_size=test_size,
            undersampling_ratio=undersampling_ratio,
            scaling=scaling,
            cv_folds=cv_folds,
            scoring=scoring,
            n_iterations=n_iterations
        )
        
        # Initialize utility classes
        self.data_processor = DataProcessor(self.config)
        self.model_creator = ModelCreator(self.config)
        self.metrics_calculator = MetricsCalculator()
        self.latency_measurer = LatencyMeasurer(self.config.n_iterations)
        
        # Initialize data attributes
        self.data = data
        self.target_column = target_column
        
        # Initialize state tracking
        self.results = {}
        self._data_processed = False
        self._grid_search_complete = False
        self._evaluation_complete = False
        
        # Process data on initialization
        self._process_data()
    
    def _process_data(self) -> None:
        """Process the input dataset and prepare it for evaluation."""
        try:
            if self.config.verbose:
                print("Processing data...")
                print(f"Dataset shape: {self.data.shape}")
                print(f"Target distribution: {self.data[self.target_column].value_counts(normalize=True) * 100}")
            
            # Use data processor to handle all data preprocessing
            self.processed_data = self.data_processor.process_data(self.data, self.target_column)
            
            if self.config.verbose:
                print(f"Training set: {self.processed_data['X_train'].shape}, "
                      f"Test set: {self.processed_data['X_test'].shape}")
            
            self._data_processed = True
            
        except Exception as e:
            raise DataProcessingError(f"Failed to process data: {str(e)}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete evaluation pipeline.
        
        This method runs:
        1. Data processing (if not already done)
        2. Grid search for optimal hyperparameters
        3. Model evaluation comparing FHE and plain text models
        
        Returns:
            Dictionary containing complete evaluation results
            
        Raises:
            FHEEvaluatorError: If pipeline execution fails
        """
        try:
            # Ensure data is processed
            if not self._data_processed:
                self._process_data()
            
            # Run grid search
            if self.config.verbose:
                print("\nPerforming grid search for optimal parameters...")
            
            self._run_grid_search()
            
            # Run model evaluation
            if self.config.verbose:
                print("\nEvaluating FHE vs plain text models...")
            
            evaluation_results = self._evaluate_models()
            
            if self.config.verbose:
                print("\nFHE vs Plain Model Comparison Summary:")
                if 'summary' in evaluation_results and 'comparison_table' in evaluation_results['summary']:
                    print(evaluation_results['summary']['comparison_table'])
            
            self._evaluation_complete = True
            return evaluation_results
            
        except Exception as e:
            raise FHEEvaluatorError(f"Pipeline execution failed: {str(e)}")
    
    def _run_grid_search(self) -> None:
        """Run grid search for all configured model types."""
        self.best_params = {}
        
        for model_type in self.config.model_types:
            if model_type not in self.config.param_grids:
                if self.config.verbose:
                    print(f"Skipping grid search for {model_type} - no param grid provided")
                continue
            
            if self.config.verbose:
                print(f"\nGrid search for {model_type.upper()}...")
            
            try:
                result = self.grid_search(
                    model_type=model_type,
                    param_grid=self.config.param_grids[model_type],
                    x=self.processed_data['X_train'],
                    y=self.processed_data['y_train'],
                    undersampler=self.processed_data['undersampler'],
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring
                )
                self.best_params[model_type] = result['best_params']
                
            except Exception as e:
                if self.config.verbose:
                    print(f"Grid search failed for {model_type}: {str(e)}")
                self.best_params[model_type] = {}
        
        self._grid_search_complete = True
    
    def grid_search(
        self,
        model_type: str,
        param_grid: Dict[str, List],
        x: np.ndarray,
        y: np.ndarray,
        undersampler: Optional[RandomUnderSampler] = None,
        cv: int = 5,
        scoring: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Perform grid search to find optimal hyperparameters for a model.
        
        Args:
            model_type: Type of model to evaluate
            param_grid: Dictionary of parameter settings to try
            x: Feature matrix for training
            y: Target vector for training
            undersampler: Undersampling strategy for class imbalance
            cv: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            
        Returns:
            Dictionary containing best parameters and grid search results
            
        Raises:
            ModelEvaluationError: If grid search fails
        """
        try:
            if undersampler is None:
                undersampler = RandomUnderSampler(random_state=self.config.random_state)
            
            # Get model estimator
            estimator = self.model_creator.get_sklearn_estimator(model_type)
            
            # Apply undersampling
            X_resampled, y_resampled = undersampler.fit_resample(x, y)
            
            if self.config.verbose:
                print(f"Performing grid search for {model_type.upper()} model...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator,
                param_grid,
                cv=cv,
                scoring=scoring,
                verbose=1 if self.config.verbose else 0,
                n_jobs=self.config.n_jobs
            )
            
            grid_search.fit(X_resampled, y_resampled)
            
            if self.config.verbose:
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best score: {grid_search.best_score_:.4f}")
            
            result = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'grid_search': grid_search
            }
            
            # Store results
            self.results[f"{model_type}_grid_search"] = result
            
            return result
            
        except Exception as e:
            raise ModelEvaluationError(f"Grid search failed for {model_type}: {str(e)}")
    
    def _evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate and compare plain vs FHE models across all configured model types.
        
        Returns:
            Dictionary containing complete evaluation results
        """
        if not self._grid_search_complete:
            raise FHEEvaluatorError("Must run grid search before model evaluation")
        
        # Setup cross-validation
        kf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Initialize results structure
        evaluation_results = {
            'model_comparison': {},
            'summary': {},
            'best_models': {}
        }
        
        # Evaluate each model type
        for model_type in self.config.model_types:
            if self.config.verbose:
                print(f"\n{'-'*70}")
                print(f"Evaluating {model_type.upper()} models")
                print(f"{'-'*70}")
            
            try:
                params = self.best_params.get(model_type, {})
                
                # Evaluate plain model
                plain_results = self._evaluate_plain_model(model_type, params, kf)
                
                # Evaluate FHE models
                fhe_results = self._evaluate_fhe_models(model_type, params, kf)
                
                # Compare models
                comparison = self._compare_models(plain_results, fhe_results)
                
                # Create visualizations
                figures = self._create_model_figures(plain_results, fhe_results, comparison)
                
                # Store results for this model type
                evaluation_results['model_comparison'][model_type] = {
                    'plain_results': plain_results,
                    'fhe_results': fhe_results,
                    'comparison': comparison,
                    'figures': figures
                }
                
                # Track best model configuration
                best_idx = comparison['best_tradeoff_idx']
                best_bit_width = self.config.bit_widths[best_idx]
                
                evaluation_results['best_models'][model_type] = {
                    'bit_width': best_bit_width,
                    'accuracy': fhe_results['accuracies'][best_idx],
                    'latency': fhe_results['latencies'][best_idx],
                    'overhead': comparison['latency_ratios'][best_idx],
                    'plain_accuracy': plain_results['avg_accuracy'],
                    'plain_latency': plain_results['avg_latency']
                }
                
            except Exception as e:
                if self.config.verbose:
                    print(f"Error evaluating {model_type}: {str(e)}")
                continue
        
        # Generate summary
        self._generate_evaluation_summary(evaluation_results)
        
        # Store results
        self.results['evaluation'] = evaluation_results
        self.evaluation_results = evaluation_results
        
        return evaluation_results
    
    def _evaluate_plain_model(
        self,
        model_type: str,
        params: Dict[str, Any],
        kf: StratifiedKFold
    ) -> Dict[str, Any]:
        """Evaluate a plain (non-FHE) model using cross-validation."""
        try:
            model = self.model_creator.get_sklearn_estimator(model_type)
            model.set_params(**params)
            
            return self._evaluate_model_cv(
                model, 
                self.processed_data['X_test'],
                self.processed_data['y_test'],
                self.processed_data['undersampler'],
                kf
            )
            
        except Exception as e:
            raise ModelEvaluationError(f"Plain model evaluation failed for {model_type}: {str(e)}")
    
    def _evaluate_fhe_models(
        self,
        model_type: str,
        params: Dict[str, Any],
        kf: StratifiedKFold
    ) -> Dict[str, Any]:
        """Evaluate FHE models with different bit widths."""
        try:
            results = {
                'bit_widths': self.config.bit_widths,
                'accuracies': [],
                'latencies': [],
                'detailed_results': []
            }
            
            for bit_width in self.config.bit_widths:
                if self.config.verbose:
                    print(f"Evaluating {model_type.upper()} with {bit_width} bits...")
                
                fhe_model = self.model_creator.get_fhe_estimator(
                    model_type, 
                    n_bits=bit_width, 
                    **params
                )
                
                bit_width_results = self._evaluate_model_cv(
                    fhe_model,
                    self.processed_data['X_test'],
                    self.processed_data['y_test'],
                    self.processed_data['undersampler'],
                    kf
                )
                
                bit_width_results['bit_width'] = bit_width
                
                results['accuracies'].append(bit_width_results['avg_accuracy'])
                results['latencies'].append(bit_width_results['avg_latency'])
                results['detailed_results'].append(bit_width_results)
                
                if self.config.verbose:
                    print(f"Bit Width {bit_width}: Accuracy={bit_width_results['avg_accuracy']:.4f}, "
                          f"Latency={bit_width_results['avg_latency']:.4f}ms")
            
            return results
            
        except Exception as e:
            raise ModelEvaluationError(f"FHE model evaluation failed for {model_type}: {str(e)}")
    
    def _evaluate_model_cv(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        undersampler: RandomUnderSampler,
        kf: StratifiedKFold
    ) -> Dict[str, Any]:
        """Evaluate a model using cross-validation."""
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
            
            # Apply undersampling
            X_train_resampled, Y_train_resampled = undersampler.fit_resample(X_train, Y_train)
            
            # Train model and measure training time
            start_time = time.time()
            model.fit(X_train_resampled, Y_train_resampled)
            training_time = time.time() - start_time
            
            # Measure inference latency
            latency = self.latency_measurer.measure_latency(model, X_test)
            
            # Make predictions
            Y_pred = model.predict(X_test)
            Y_pred_proba = (
                model.predict_proba(X_test)[:, 1] 
                if hasattr(model, "predict_proba") 
                else Y_pred
            )
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(Y_test, Y_pred, Y_pred_proba)
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'latency_ms': latency,
                'training_time_s': training_time,
                'test_samples': len(X_test),
                **metrics
            }
            
            results['fold_results'].append(fold_result)
            
            # Store metrics for averaging
            for metric_name, value in metrics.items():
                results['metrics'][metric_name].append(value)
            
            results['metrics']['latency'].append(latency)
            results['metrics']['training_time'].append(training_time)
            
            if self.config.verbose:
                print(f"Fold {fold_idx + 1}: Accuracy={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1']:.4f}, Latency={latency:.4f}ms")
        
        # Calculate averages
        for metric_name in results['metrics']:
            results[f'avg_{metric_name}'] = np.mean(results['metrics'][metric_name])
        
        if self.config.verbose:
            self._print_model_summary(results)
        
        return results
    
    def _compare_models(
        self,
        plain_results: Dict[str, Any],
        fhe_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare plain and FHE models to evaluate trade-offs."""
        comparison = {
            'bit_widths': fhe_results['bit_widths'],
            'accuracy_diffs': [],
            'latency_ratios': [],
            'tradeoff_scores': []
        }
        
        plain_accuracy = plain_results['avg_accuracy']
        plain_latency = plain_results['avg_latency']
        
        for i, fhe_accuracy in enumerate(fhe_results['accuracies']):
            fhe_latency = fhe_results['latencies'][i]
            
            # Calculate differences and ratios
            accuracy_diff = fhe_accuracy - plain_accuracy
            latency_ratio = fhe_latency / plain_latency if plain_latency > 0 else float('inf')
            
            # Calculate trade-off score (lower is better)
            accuracy_weight = 1.0
            latency_weight = 0.5
            tradeoff_score = (latency_weight * latency_ratio) - (accuracy_weight * accuracy_diff)
            
            comparison['accuracy_diffs'].append(accuracy_diff)
            comparison['latency_ratios'].append(latency_ratio)
            comparison['tradeoff_scores'].append(tradeoff_score)
        
        # Find best trade-off
        comparison['best_tradeoff_idx'] = np.argmin(comparison['tradeoff_scores'])
        comparison['best_bit_width'] = fhe_results['bit_widths'][comparison['best_tradeoff_idx']]
        
        return comparison
    
    def _create_model_figures(
        self,
        plain_results: Dict[str, Any],
        fhe_results: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> Dict[str, plt.Figure]:
        """Create visualization figures for model comparison."""
        try:
            bit_widths = fhe_results['bit_widths']
            
            # Create accuracy comparison plot
            acc_fig, acc_ax = plt.subplots(figsize=(10, 6))
            acc_ax.plot(bit_widths, fhe_results['accuracies'], 'o-', label='FHE Model')
            acc_ax.axhline(y=plain_results['avg_accuracy'], color='r', linestyle='--', 
                         label='Plain Model')
            acc_ax.set_xlabel('Bit Width')
            acc_ax.set_ylabel('Accuracy')
            acc_ax.set_title('Accuracy vs Bit Width')
            acc_ax.grid(True, alpha=0.3)
            acc_ax.legend()
            
            # Create latency comparison plot
            lat_fig, lat_ax = plt.subplots(figsize=(10, 6))
            lat_ax.plot(bit_widths, fhe_results['latencies'], 'o-', label='FHE Model')
            lat_ax.axhline(y=plain_results['avg_latency'], color='r', linestyle='--', 
                         label='Plain Model')
            lat_ax.set_xlabel('Bit Width')
            lat_ax.set_ylabel('Latency (ms)')
            lat_ax.set_title('Latency vs Bit Width (Log Scale)')
            lat_ax.set_yscale('log')
            lat_ax.grid(True, alpha=0.3)
            lat_ax.legend()
            
            # Create trade-off plot
            trade_fig, trade_ax = plt.subplots(figsize=(10, 6))
            trade_ax.plot(bit_widths, comparison['tradeoff_scores'], 'o-')
            best_idx = comparison['best_tradeoff_idx']
            trade_ax.plot(bit_widths[best_idx], comparison['tradeoff_scores'][best_idx], 'ro', 
                        markersize=10, label=f'Best Trade-off: {bit_widths[best_idx]} bits')
            trade_ax.set_xlabel('Bit Width')
            trade_ax.set_ylabel('Trade-off Score (lower is better)')
            trade_ax.set_title('Accuracy-Latency Trade-off by Bit Width')
            trade_ax.grid(True, alpha=0.3)
            trade_ax.legend()
            
            return {
                'accuracy_fig': acc_fig,
                'latency_fig': lat_fig,
                'tradeoff_fig': trade_fig
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to create model figures: {str(e)}")
    
    def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> None:
        """Generate summary of evaluation results across all models."""
        model_types = list(evaluation_results['best_models'].keys())
        if not model_types:
            return
        
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
        
        if self.config.verbose:
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
    
    def _print_model_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of model evaluation results."""
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
    
    def visualize_model_comparison(self) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create visualizations comparing performance of all evaluated models.
        
        Returns:
            Tuple of matplotlib figures: (accuracy_comparison, latency_comparison)
            
        Raises:
            VisualizationError: If visualization creation fails
        """
        if not self._evaluation_complete:
            raise FHEEvaluatorError("Must run evaluation before creating visualizations")
        
        try:
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
            
            return self._create_comparison_figures(
                models, plain_accuracies, fhe_accuracies,
                plain_latencies, fhe_latencies,
                overheads, bit_widths
            )
            
        except Exception as e:
            raise VisualizationError(f"Failed to create comparison visualizations: {str(e)}")
    
    def _create_comparison_figures(
        self,
        models: List[str],
        plain_accs: List[float],
        fhe_accs: List[float],
        plain_lats: List[float],
        fhe_lats: List[float],
        overheads: List[float],
        bit_widths: List[int]
    ) -> Tuple[plt.Figure, plt.Figure]:
        """Create comparison figures for accuracy and latency."""
        # Accuracy comparison
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
        
        # Latency comparison
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
    
    # Pareto analysis methods will be added in a future update
    # to keep the core class focused on primary evaluation functionality
    
    def find_pareto_optimal_solutions(
        self,
        accuracy_weight: float = 1.0,
        latency_weight: float = 1.0,
        invert_latency: bool = True
    ) -> Dict[str, Any]:
        """
        Find Pareto-optimal solutions for accuracy-latency trade-offs.
        
        Note: This is a placeholder for Pareto analysis functionality.
        The full implementation would be added in a separate module or
        as an extension to keep the core class focused.
        
        Args:
            accuracy_weight: Weight for accuracy in optimization
            latency_weight: Weight for latency in optimization
            invert_latency: If True, use 1/latency for optimization
            
        Returns:
            Dictionary containing Pareto analysis results
            
        Raises:
            NotImplementedError: This functionality is not yet implemented
        """
        raise NotImplementedError(
            "Pareto analysis functionality will be implemented in a future version. "
            "For now, use the comparison results from run_full_pipeline() to analyze trade-offs."
        )
    
    def get_pareto_recommendations(self, use_case_priority: str = 'balanced') -> Dict[str, Any]:
        """
        Get recommendations based on Pareto analysis.
        
        Note: This is a placeholder for recommendation functionality.
        
        Args:
            use_case_priority: Priority for recommendations ('accuracy', 'latency', 'balanced', 'conservative')
            
        Returns:
            Dictionary containing recommendations
            
        Raises:
            NotImplementedError: This functionality is not yet implemented
        """
        raise NotImplementedError(
            "Pareto recommendations functionality will be implemented in a future version."
        )
    
    def save_results(self, output_dir: str = "results") -> Dict[str, str]:
        """
        Save evaluation results and figures to disk.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping result types to file paths
            
        Raises:
            FHEEvaluatorError: If saving fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_files = {}
            
            # Save evaluation summary
            if self._evaluation_complete and 'evaluation' in self.results:
                summary_path = os.path.join(output_dir, 'fhe_evaluation_report.txt')
                self._save_text_report(summary_path)
                saved_files['report'] = summary_path
            
            # Save figures
            if self._evaluation_complete:
                try:
                    acc_fig, lat_fig = self.visualize_model_comparison()
                    
                    acc_path = os.path.join(output_dir, 'overall_accuracy_comparison.png')
                    lat_path = os.path.join(output_dir, 'overall_latency_comparison.png')
                    
                    acc_fig.savefig(acc_path, dpi=300, bbox_inches='tight')
                    lat_fig.savefig(lat_path, dpi=300, bbox_inches='tight')
                    
                    saved_files['accuracy_comparison'] = acc_path
                    saved_files['latency_comparison'] = lat_path
                    
                    plt.close(acc_fig)
                    plt.close(lat_fig)
                    
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Could not save comparison figures: {e}")
            
            if self.config.verbose:
                print(f"Results saved to: {output_dir}")
                for result_type, path in saved_files.items():
                    print(f"  {result_type}: {path}")
            
            return saved_files
            
        except Exception as e:
            raise FHEEvaluatorError(f"Failed to save results: {str(e)}")
    
    def _save_text_report(self, filepath: str) -> None:
        """Save a text report of evaluation results."""
        with open(filepath, 'w') as f:
            f.write("FHE MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Dataset shape: {self.data.shape}\n")
            f.write(f"Target column: {self.target_column}\n")
            f.write(f"Features: {self.data.shape[1] - 1}\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 15 + "\n")
            f.write(f"Model types: {', '.join(self.config.model_types)}\n")
            f.write(f"Bit widths: {', '.join(map(str, self.config.bit_widths))}\n")
            f.write(f"CV folds: {self.config.cv_folds}\n")
            f.write(f"Scoring: {self.config.scoring}\n\n")
            
            # Results summary
            if 'evaluation' in self.results:
                evaluation = self.results['evaluation']
                if 'summary' in evaluation:
                    summary = evaluation['summary']
                    f.write("RESULTS SUMMARY\n")
                    f.write("-" * 15 + "\n")
                    
                    # Check if keys exist before writing
                    if 'best_accuracy_model' in summary:
                        f.write(f"Best accuracy model: {summary['best_accuracy_model']}\n")
                    else:
                        f.write("Best accuracy model: Not available\n")
                        
                    if 'best_latency_model' in summary:
                        f.write(f"Best latency model: {summary['best_latency_model']}\n")
                    else:
                        f.write("Best latency model: Not available\n")
                        
                    if 'best_overhead_model' in summary:
                        f.write(f"Best overhead model: {summary['best_overhead_model']}\n")
                    else:
                        f.write("Best overhead model: Not available\n")
                    
                    f.write("\n")
                    
                    if 'comparison_table' in summary:
                        f.write("DETAILED COMPARISON\n")
                        f.write("-" * 20 + "\n")
                        f.write(str(summary['comparison_table']))
                        f.write("\n\n")
