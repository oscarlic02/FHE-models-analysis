## Overview

The **FHE-models-analysis** project is dedicated to the exploration and evaluation of Fully Homomorphic Encryption (FHE) models. FHE enables computations on encrypted data, ensuring privacy and security throughout the process. This project is supervised by **Prof. Pelusi** as part of the **Undergraduate Research Opportunity Programme** at the **Polytechnic of Turin**, emphasizing academic rigor and innovation in the field of cryptography and secure computation.

This repository includes a Python class `fhe_model_evaluator` and a Jupyter Notebook `CreditCard.ipynb`, both of which are designed to facilitate the analysis and benchmarking of FHE models. The class provides a structured approach to evaluate FHE models, while the notebook documents the findings and visualizations, supporting academic research and reproducibility.

### Integration with Concrete-ML

The project leverages the **Concrete-ML** library, a state-of-the-art framework for machine learning on encrypted data. Concrete-ML simplifies the application of FHE by providing tools to train, compile, and execute machine learning models that operate directly on encrypted inputs. Key benefits of using Concrete-ML in this project include:

- **Seamless FHE Integration**:
  - Concrete-ML bridges the gap between machine learning and FHE, enabling straightforward implementation of privacy-preserving models.
- **Optimized Performance**:
  - The library is designed to minimize computational overhead, ensuring efficient execution of encrypted operations.
- **Broad Compatibility**:
  - Support for a variety of machine learning algorithms, including linear models, tree-based models, and neural networks.
- **Ease of Use**:
  - High-level APIs and detailed documentation make it accessible to researchers and developers.

## Features

### **`fhe_model_evaluator` Class**

The `fhe_model_evaluator` class is a cornerstone of this project, offering a comprehensive toolkit for FHE model analysis. Key features include:

- **Initialization and Configuration**:
  - A constructor that allows users to initialize the evaluator with a model, dataset, and encryption scheme.
  - Flexible configuration options to adapt to various FHE schemes and datasets.
- **Dataset Loading**:
  - Methods to load and preprocess datasets, ensuring compatibility with FHE operations.
  - Support for real-world datasets commonly used in academic research.
- **FHE Scheme Application**:
  - Integration with popular FHE libraries to apply encryption schemes seamlessly.
  - Support for multiple encryption schemes to compare their performance and suitability.
- **Benchmarking and Metrics**:
  - Tools to evaluate the performance of FHE models using custom metrics.
  - Detailed benchmarking reports to assess computational overhead, accuracy, and scalability.
- **Visualization**:
  - Built-in methods to generate graphical representations of performance metrics.
  - Visualizations designed to aid in academic presentations and publications.

### **Jupyter Notebook**

The `CreditCard.ipynb` notebook serves as a comprehensive guide and documentation of the research process. It includes:

- **Step-by-Step Analysis**:
  - A structured walkthrough of the evaluation process, from dataset preparation to result interpretation.
  - Clear explanations of each step, making it accessible to researchers and students.
- **Detailed Model Evaluation**:
  - In-depth analysis of FHE models, highlighting their strengths and limitations.
  - Comparative studies of different encryption schemes and their impact on model performance.
- **Graphical Insights**:
  - High-quality visualizations of performance metrics, including accuracy, latency, and computational cost.
  - Graphs and charts tailored for academic research and presentations.
- **Reproducibility**:
  - Code snippets and explanations to ensure that the research can be reproduced and extended by others.

## Academic Context

This project is part of the **Undergraduate Research Opportunity Programme** at the **Polytechnic of Turin**, under the supervision of **Prof. Pelusi**. The research aims to contribute to the academic community by:

- Exploring the practical applications of Fully Homomorphic Encryption in machine learning and data analysis.
- Providing a robust framework for evaluating FHE models, fostering further research in the field.
- Promoting the adoption of privacy-preserving technologies in real-world scenarios.

The findings and methodologies documented in this repository are intended to support collaborative research efforts.

## Getting Started

1. Clone the repository:
        ```bash
        git clone https://github.com/your-username/FHE-models-analysis.git
        ```
2. Navigate to the project directory:
        ```bash
        cd FHE-models-analysis
3. Explore the `fhe_model_evaluator` class:
        - Import the class in your Python script:
            ```python
            from fhe_model_evaluator import FHEModelEvaluator
            ```
        - Initialize the evaluator:
            ```python
            evaluator = FHEModelEvaluator(model, dataset, encryption_scheme)
            ```
        - Use the provided methods to analyze and benchmark your FHE model.
5. Open the `CreditCard.ipynb` file in Jupyter Notebook or JupyterLab to follow the documented analysis:
        ```bash
        jupyter notebook CreditCard.ipynb
        ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
