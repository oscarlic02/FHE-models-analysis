# FHE Model Evaluator Library

A comprehensive Python library for evaluating and comparing Fully Homomorphic Encryption (FHE) models against traditional machine learning models. This library provides automated hyperparameter tuning, performance evaluation, and visualization capabilities for FHE implementations using concrete-ml.

## Academic Context

This project is part of the **Undergraduate Research Opportunity Programme** at the **Polytechnic of Turin**, under the supervision of **Prof. Pelusi**. The research aims to contribute to the academic community by exploring the practical applications of Fully Homomorphic Encryption in machine learning and data analysis.

## Overview

The **FHE-models-analysis** project is dedicated to the exploration and evaluation of Fully Homomorphic Encryption (FHE) models. FHE enables computations on encrypted data, ensuring privacy and security throughout the process. This repository has been refactored to follow standard Python library structure and best practices, improving readability and usability for external use.

### Integration with Concrete-ML

The project leverages the **Concrete-ML** library, a state-of-the-art framework for machine learning on encrypted data. Concrete-ML simplifies the application of FHE by providing tools to train, compile, and execute machine learning models that operate directly on encrypted inputs.

## Features

- **Automated Model Evaluation**: Compare FHE vs plain text models across multiple algorithms
- **Hyperparameter Optimization**: Grid search for optimal model parameters
- **Performance Metrics**: Comprehensive accuracy, latency, and trade-off analysis
- **Visualization**: Generate detailed plots and comparison charts
- **Modular Design**: Clean, extensible architecture with separate utility modules
- **Error Handling**: Robust error handling with custom exception classes
- **Export Capabilities**: Save results and figures to disk

## Supported Models

- **Logistic Regression** (`lr`)
- **Random Forest** (`rf`)
- **Decision Tree** (`dt`)
- **XGBoost** (`xgb`) - if available
- **Multi-layer Perceptron** (`mlp`)

## Installation

### Prerequisites

```bash
# Install concrete-ml for FHE functionality
pip install concrete-ml

# Install scientific computing dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Install additional ML libraries
pip install xgboost imbalanced-learn

# For Jupyter notebook support
pip install jupyter ipykernel
```

### Library Setup

1. Clone or download the repository
2. Ensure the `fhe_evaluator/` package is in your Python path
3. Run the test script to verify installation:

```bash
python test_fhe_evaluator.py
```

## Quick Start

### Basic Usage

```python
from fhe_evaluator import FHEModelEvaluator
import pandas as pd

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Initialize the evaluator
evaluator = FHEModelEvaluator(
    data=data,
    target_column='target',
    model_types=['lr', 'rf', 'dt'],
    bit_widths=[2, 4, 6, 8],
    cv_folds=5
)

# Run complete evaluation
results = evaluator.run_full_pipeline()

# Generate visualizations
acc_fig, lat_fig = evaluator.visualize_model_comparison()

# Save results
evaluator.save_results("output_directory")
```

### Credit Card Fraud Detection Example

```python
# Load credit card dataset
df = pd.read_csv('creditcard.csv')

# Configure for imbalanced classification
evaluator = FHEModelEvaluator(
    data=df,
    target_column='Class',
    model_types=['lr', 'rf', 'dt'],
    bit_widths=[2, 3, 4, 6, 8],
    undersampling_ratio=0.1,  # Handle class imbalance
    scaling=True,             # Scale features
    scoring='f1',            # F1 score for imbalanced data
    cv_folds=5
)

# Run evaluation
results = evaluator.run_full_pipeline()
```

## Library Architecture

### Core Components

```
fhe_evaluator/
├── __init__.py          # Package initialization and exports
├── core.py              # Main FHEModelEvaluator class
├── config.py            # Configuration and parameter management
├── utils.py             # Utility classes for model creation, data processing, etc.
└── exceptions.py        # Custom exception classes
```

### Key Classes

- **`FHEModelEvaluator`**: Main evaluation class with complete pipeline
- **`ModelConfig`**: Configuration management for evaluation parameters
- **`ModelCreator`**: Factory for creating sklearn and FHE model instances
- **`DataProcessor`**: Handles data preprocessing and splitting
- **`MetricsCalculator`**: Computes performance metrics
- **`LatencyMeasurer`**: Measures model inference latency

## Examples

### 1. Credit Card Fraud Detection

Run the complete credit card fraud detection example:

```bash
python credit_card_fhe_evaluation.py
```

Or for a quicker test:

```bash
python credit_card_fhe_evaluation.py --quick
```

### 2. Jupyter Notebook

Open and run the `CreditCard.ipynb` notebook for an interactive experience.

### 3. Custom Dataset

```python
# Example with your own dataset
import pandas as pd
from fhe_evaluator import FHEModelEvaluator

# Load your data
data = pd.read_csv('my_dataset.csv')

# Basic evaluation
evaluator = FHEModelEvaluator(
    data=data,
    target_column='my_target_column',
    model_types=['lr', 'rf'],
    bit_widths=[2, 4, 6]
)

results = evaluator.run_full_pipeline()
```

## Testing

Run the test suite to verify library functionality:

```bash
python test_fhe_evaluator.py
```

This will test:

- Import functionality
- Configuration management
- Utility classes
- Basic evaluation with synthetic data
- Error handling

## Results and Analysis

### Output Structure

The evaluation produces comprehensive results including:

```python
results = {
    'model_comparison': {
        'lr': {
            'plain_results': {...},
            'fhe_results': {...},
            'comparison': {...},
            'figures': {...}
        },
        # ... other models
    },
    'summary': {
        'best_accuracy_model': 'rf',
        'best_latency_model': 'lr', 
        'best_overhead_model': 'dt',
        'comparison_table': DataFrame(...)
    },
    'best_models': {
        'lr': {
            'bit_width': 4,
            'accuracy': 0.95,
            'latency': 12.3,
            'overhead': 5.2,
            # ...
        }
        # ... other models
    }
}
```

### Saved Outputs

When using `evaluator.save_results()`:

```
results/
├── fhe_evaluation_report.txt           # Text summary report
├── overall_accuracy_comparison.png     # Model accuracy comparison
├── overall_latency_comparison.png      # Model latency comparison
├── lr/                                 # Per-model visualizations
│   ├── lr_accuracy.png
│   ├── lr_latency.png
│   └── lr_tradeoff.png
├── rf/
│   └── ...
└── dt/
    └── ...
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure concrete-ml is installed
   pip install concrete-ml
   
   # Check Python path
   export PYTHONPATH=$PYTHONPATH:/path/to/fhe-evaluator
   ```

2. **Memory Issues**

   ```python
   # Use undersampling for large datasets
   evaluator = FHEModelEvaluator(
       data=data,
       target_column='target',
       undersampling_ratio=0.1  # Use 10% of data
   )
   ```

3. **Slow Execution**

   ```python
   # Reduce computational load
   evaluator = FHEModelEvaluator(
       data=data,
       target_column='target',
       model_types=['lr'],    # Single model
       bit_widths=[2, 4],     # Fewer bit widths
       cv_folds=3,            # Fewer folds
       n_iterations=20        # Fewer iterations
   )
   ```

## Contributing

The library is designed to be extensible:

1. **Adding New Models**: Extend `ModelCreator` class
2. **New Metrics**: Add to `MetricsCalculator` class  
3. **Custom Visualizations**: Extend visualization methods
4. **New Configurations**: Add to `ModelConfig` class

## Future Enhancements

- Pareto optimality analysis for trade-off optimization
- Support for regression tasks
- Advanced visualization options
- Model deployment utilities
- Automated report generation
- Integration with MLflow for experiment tracking

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
