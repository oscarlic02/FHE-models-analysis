#!/usr/bin/env python3
"""
Credit Card Fraud Detection - FHE Model Evaluation Example

This script demonstrates how to use the FHE Evaluator library to evaluate
and compare Fully Homomorphic Encryption models for credit card fraud detection.

The script:
1. Loads the credit card fraud detection dataset
2. Evaluates multiple ML models (Logistic Regression, Random Forest, Decision Tree)
3. Compares FHE vs plain text implementations across different bit widths
4. Generates comprehensive analysis and visualizations
5. Saves results and figures to disk

Dataset: Credit Card Fraud Detection
Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
Features: 30 (Time, V1-V28 anonymized features, Amount)
Target: Class (0: Normal, 1: Fraud)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the FHE evaluator library
try:
    from fhe_evaluator import FHEModelEvaluator, ModelConfig
    print("✓ FHE Evaluator library imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FHE Evaluator library: {e}")
    print("Make sure you're running this script from the correct directory.")
    sys.exit(1)


def load_credit_card_data(file_path: str = "creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card fraud detection dataset.
    
    Args:
        file_path: Path to the creditcard.csv file
        
    Returns:
        DataFrame containing the credit card data
        
    Raises:
        FileNotFoundError: If the dataset file is not found
    """
    if not os.path.exists(file_path):
        print(f"✗ Dataset file not found: {file_path}")
        print("\nPlease download the Credit Card Fraud Detection dataset from:")
        print("https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("And place the 'creditcard.csv' file in the same directory as this script.")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Samples: {df.shape[0]:,}")
    
    # Display class distribution
    class_dist = df['Class'].value_counts()
    fraud_rate = class_dist[1] / len(df) * 100
    print(f"  Normal transactions: {class_dist[0]:,} ({100-fraud_rate:.2f}%)")
    print(f"  Fraudulent transactions: {class_dist[1]:,} ({fraud_rate:.2f}%)")
    
    return df


def run_fhe_evaluation(data: pd.DataFrame, quick_run: bool = False) -> dict:
    """
    Run the FHE model evaluation pipeline.
    
    Args:
        data: Credit card dataset
        quick_run: If True, use reduced parameters for faster execution
        
    Returns:
        Dictionary containing evaluation results
    """
    print("\n" + "="*70)
    print("INITIALIZING FHE MODEL EVALUATOR")
    print("="*70)
    
    # Configure evaluation parameters
    if quick_run:
        print("Running in quick mode (reduced parameters for faster execution)")
        model_types = ['lr', 'dt']  # Fewer models
        bit_widths = [2, 4, 6]      # Fewer bit widths
        cv_folds = 3                # Fewer CV folds
        n_iterations = 20           # Fewer latency measurements
        undersampling_ratio = 0.05  # More aggressive undersampling
    else:
        print("Running in full mode (comprehensive evaluation)")
        model_types = ['lr', 'rf', 'dt']
        bit_widths = [2, 3, 4, 6, 8]
        cv_folds = 5
        n_iterations = 100
        undersampling_ratio = 0.1
    
    # Initialize the evaluator
    evaluator = FHEModelEvaluator(
        data=data,
        target_column='Class',
        model_types=model_types,
        bit_widths=bit_widths,
        random_state=42,
        n_jobs=-1,
        verbose=True,
        test_size=0.2,
        undersampling_ratio=undersampling_ratio,
        scaling=True,
        cv_folds=cv_folds,
        scoring='f1',  # F1 score is good for imbalanced data
        n_iterations=n_iterations
    )
    
    print(f"\nConfiguration:")
    print(f"  Model types: {', '.join([m.upper() for m in model_types])}")
    print(f"  Bit widths: {bit_widths}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Undersampling ratio: {undersampling_ratio}")
    print(f"  Latency measurement iterations: {n_iterations}")
    
    # Run the evaluation pipeline
    print("\n" + "="*70)
    print("RUNNING EVALUATION PIPELINE")
    print("="*70)
    
    try:
        results = evaluator.run_full_pipeline()
        print("\n✓ Evaluation pipeline completed successfully!")
        return evaluator, results
    
    except Exception as e:
        print(f"\n✗ Evaluation pipeline failed: {e}")
        # Return a basic results structure with empty dictionaries to prevent KeyErrors
        empty_results = {
            'summary': {},
            'best_models': {},
            'error_message': str(e)
        }
        return evaluator, empty_results


def analyze_results(evaluator: FHEModelEvaluator, results: dict) -> None:
    """
    Analyze and display the evaluation results.
    
    Args:
        evaluator: The FHE evaluator instance
        results: Evaluation results dictionary
    """
    print("\n" + "="*70)
    print("ANALYSIS OF RESULTS")
    print("="*70)
    
    if 'summary' not in results:
        print("No summary results available.")
        return
    
    summary = results['summary']
    
    if 'best_models' not in results:
        print("No best models data available.")
        return
        
    best_models = results['best_models']
    
    # Display best models
    print("\nBest Models by Metric:")
    print("-" * 25)
    
    if 'best_accuracy_model' in summary:
        print(f"Best Accuracy: {summary['best_accuracy_model'].upper()}")
        if summary['best_accuracy_model'] in best_models:
            model_info = best_models[summary['best_accuracy_model']]
            print(f"  FHE Accuracy: {model_info['accuracy']:.4f}")
            print(f"  Plain Accuracy: {model_info['plain_accuracy']:.4f}")
            print(f"  Best Bit Width: {model_info['bit_width']}")
    else:
        print("Best Accuracy: Not available")
    
    if 'best_latency_model' in summary:
        print(f"\nBest Latency: {summary['best_latency_model'].upper()}")
        if summary['best_latency_model'] in best_models:
            model_info = best_models[summary['best_latency_model']]
            print(f"  FHE Latency: {model_info['latency']:.2f} ms")
            print(f"  Plain Latency: {model_info['plain_latency']:.2f} ms")
            print(f"  Overhead: {model_info['overhead']:.1f}x")
    else:
        print("\nBest Latency: Not available")
    
    if 'best_overhead_model' in summary:
        print(f"\nBest Overhead: {summary['best_overhead_model'].upper()}")
        if summary['best_overhead_model'] in best_models:
            model_info = best_models[summary['best_overhead_model']]
            print(f"  Latency Overhead: {model_info['overhead']:.1f}x")
            print(f"  FHE Accuracy: {model_info['accuracy']:.4f}")
    else:
        print("\nBest Overhead: Not available")
    
    # Display comparison table
    if 'comparison_table' in summary:
        print("\nDetailed Comparison Table:")
        print("-" * 30)
        print(summary['comparison_table'].round(4))
    
    # Model-specific insights
    print("\nModel-Specific Insights:")
    print("-" * 25)
    
    if not best_models:
        print("No model-specific insights available.")
        return
    
    for model_type, model_data in best_models.items():
        accuracy_diff = model_data['accuracy'] - model_data['plain_accuracy']
        print(f"\n{model_type.upper()}:")
        print(f"  Accuracy change: {accuracy_diff:+.4f}")
        print(f"  Latency overhead: {model_data['overhead']:.1f}x")
        print(f"  Optimal bit width: {model_data['bit_width']}")
        
        if accuracy_diff > -0.01:  # Less than 1% accuracy loss
            print(f"  → Good accuracy preservation")
        elif accuracy_diff > -0.05:  # Less than 5% accuracy loss
            print(f"  → Moderate accuracy loss")
        else:
            print(f"  → Significant accuracy loss")


def save_results_and_figures(evaluator: FHEModelEvaluator, output_dir: str = "results") -> None:
    """
    Save evaluation results and generate visualizations.
    
    Args:
        evaluator: The FHE evaluator instance
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("SAVING RESULTS AND GENERATING VISUALIZATIONS")
    print("="*70)
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")
        
        # Save results
        saved_files = evaluator.save_results(output_dir)
        
        # Generate and save model comparison figures
        try:
            acc_fig, lat_fig = evaluator.visualize_model_comparison()
            
            acc_path = os.path.join(output_dir, "overall_accuracy_comparison.png")
            lat_path = os.path.join(output_dir, "overall_latency_comparison.png")
            
            acc_fig.savefig(acc_path, dpi=300, bbox_inches='tight')
            lat_fig.savefig(lat_path, dpi=300, bbox_inches='tight')
            
            saved_files.update({
                'accuracy_comparison': acc_path,
                'latency_comparison': lat_path
            })
            
            print("✓ Comparison visualizations saved")
            
            # Close figures to free memory
            plt.close(acc_fig)
            plt.close(lat_fig)
            
        except Exception as e:
            print(f"Warning: Could not create comparison visualizations: {e}")
        
        # Save individual model figures
        if 'evaluation' in evaluator.results:
            model_comparison = evaluator.results['evaluation']['model_comparison']
            
            for model_type, model_results in model_comparison.items():
                if 'figures' in model_results:
                    model_dir = os.path.join(output_dir, model_type)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    figures = model_results['figures']
                    for fig_name, fig in figures.items():
                        if fig is not None:
                            fig_path = os.path.join(model_dir, f"{model_type}_{fig_name.replace('_fig', '')}.png")
                            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                            plt.close(fig)
                    
                    print(f"✓ {model_type.upper()} visualizations saved to {model_type}/")
        
        print(f"\n✓ All results saved successfully!")
        print("\nSaved files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
            
    except Exception as e:
        print(f"✗ Error saving results: {e}")

def main():
    """Main execution function."""
    print("FHE MODEL EVALUATION FOR CREDIT CARD FRAUD DETECTION")
    print("=" * 55)
    print("This script evaluates FHE models for credit card fraud detection.")
    print("It compares accuracy and latency trade-offs across different configurations.\n")
    
    # Check for quick run option
    quick_run = '--quick' in sys.argv or '-q' in sys.argv
    if quick_run:
        print("⚡ Quick run mode enabled (use --quick or -q)")
    else:
        print("🔬 Full evaluation mode (use --quick for faster execution)")
    
    try:
        # Step 1: Load dataset
        print("\n" + "="*50)
        print("STEP 1: LOADING DATASET")
        print("="*50)
        
        data = load_credit_card_data("creditcard.csv")
        
        # Step 2: Run evaluation
        print("\n" + "="*50)
        print("STEP 2: FHE MODEL EVALUATION")
        print("="*50)
        
        evaluator, results = run_fhe_evaluation(data, quick_run=quick_run)
        
        # Step 3: Analyze results
        print("\n" + "="*50)
        print("STEP 3: RESULTS ANALYSIS")
        print("="*50)
        
        analyze_results(evaluator, results)
        
        # Step 4: Save results
        print("\n" + "="*50)
        print("STEP 4: SAVING RESULTS")
        print("="*50)
        
        save_results_and_figures(evaluator, "results")
                
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Check the 'results/' directory for detailed outputs.")
        
    except FileNotFoundError:
        print("\n✗ Dataset not found. Please download the credit card dataset.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Evaluation failed with error: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
