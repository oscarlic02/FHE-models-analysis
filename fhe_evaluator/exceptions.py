"""
Exception classes for the FHE Model Evaluator library.
"""

class FHEEvaluatorError(Exception):
    """Base exception for FHE Model Evaluator."""
    pass


class ModelNotSupportedError(FHEEvaluatorError):
    """Raised when an unsupported model type is requested."""
    pass


class DataProcessingError(FHEEvaluatorError):
    """Raised when data processing fails."""
    pass


class ParameterValidationError(FHEEvaluatorError):
    """Raised when invalid parameters are provided."""
    pass


class EvaluationError(FHEEvaluatorError):
    """Raised when model evaluation fails."""
    pass


class ParetoAnalysisError(FHEEvaluatorError):
    """Raised when Pareto analysis fails."""
    pass


class ModelEvaluationError(FHEEvaluatorError):
    """Raised when model evaluation fails."""
    pass


class VisualizationError(FHEEvaluatorError):
    """Raised when visualization creation fails."""
    pass
