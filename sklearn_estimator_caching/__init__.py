"""
Code for wrapping sklearn estimators to make them automatically cache the
results of fit/predict/transform/fit_predict/fit_transform to disk.

Use EstimatorCachingWrapperGetter and see its docstring for details
"""

from .estimator_caching_wrapper_getter import EstimatorCachingWrapperGetter

__all__ = ("EstimatorCachingWrapperGetter",)
