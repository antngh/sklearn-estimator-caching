"""
A utility class to help creating an sklearn estimator wrapped in the
EstimatorCachingWrapper class.

You want to instantiate this once in your script, for a given experiment or pipeline,
and then use that wrapper to wrap your slow running estimators. See the
EstimatorCachingWrapperGetter docstring for more information.
"""

import hashlib
from pathlib import Path

import structlog
from sklearn.base import BaseEstimator

from .cacheable_estimator import EstimatorCachingWrapper

logger = structlog.get_logger()


class EstimatorCachingWrapperGetter:
    """
    A utility class to help creating an sklearn estimator wrapped in the
    EstimatorCachingWrapper class.

    This is a useful utility for when you have an estimator with a slow
    fit/predict/transform method and you want to cache the results to disk.
    For example you might be building an sklearn pipeline with a slow early stage,
    now as you add steps or modify code, only the slow code that needs to run will
    be run.

    ```python
    estimator_cacher = EstimatorCacherGetter(
        project = "overarching_project_name",
        experiment = "experiment_name",
        base_dir="/path/to/sklearn/caching",
        label="some_variation_of_experiment",
        other_identifier="some_other_identifier",
    )
    sklearn_estimator_with_caching = estimator_cacher(sklearn_estimator_instance)
    another_sklearn_estimator_with_caching = estimator_cacher(
        another_sklearn_estimator_instance
    )
    ```
    You can now go ahead and use your estimators like normal, but the
    fit/transform/predict (and fit_predict/fit_transform) methods will only run the
    first time (saving to disk), subsequent runs (called with identical setup and
    identical input data) will load the cached data from disk.

    Attributes
    ----------
    project: str
        The name of the overarching project
    experiment: str
        The name of the experiment within the project
    base_dir: Path
        The base directory to save the cached estimators/data to
    """

    def __init__(
        self,
        project: str,
        experiment: str,
        base_dir: Path | str,
        **kwargs,
    ):
        """
        Set up.

        Validate inputs and set attribute

        Parameters
        ----------
        project: str
            The name of the overarching project
        experiment: str
            The name of the experiment within the project
        base_dir: Path | str,
            The base directory to save the cached estimators/data to
        kwargs:
            Any other kwargs to define a given experiment run. For example, you might
            want to add a label or some other identifier to mark this run as a different
            one within experiment
            Use these instead of the kwargs in the __call__ method if you want all
            cached estimators to have the same kwargs. If you only have one estimator
            you can use kwargs here or in __call__ interchangeably.
            The kwargs passed to the __init__ method and the __call__ method should not
            have any overlapping keys.
        """
        self.validate_dict(kwargs)

        self.project = project
        self.experiment = experiment
        self.base_dir = Path(base_dir)

        self._init_kwargs = kwargs

        logger.debug(
            f"Created {type(self).__name__} with {self.project=}, "
            f"{self.experiment=}, {self.base_dir=}, "
        )

    @staticmethod
    def validate_dict(dict_: dict):
        """
        Validate that a dictionary only contains string values.

        Parameters
        ----------
        dict_: dict
            The dictionary to validate
        """
        key_type_dict = {k: type(v) for k, v in dict_.items() if not isinstance(v, str)}
        if len(key_type_dict):
            msg = f"Only string values are allowed, have {key_type_dict=}"
            raise TypeError(msg)

    @staticmethod
    def get_dictionary_hash(dict_: dict) -> str:
        """
        Get a hash for a dictionary.

        Parameters
        ----------
        dict_: dict
            The dictionary to hash

        Returns
        -------
        str
            The hash of the dictionary
        """
        return hashlib.sha256(str(sorted(dict_.items())).encode()).hexdigest()

    def __call__(self, estimator: BaseEstimator, **kwargs) -> EstimatorCachingWrapper:
        """
        Wrap an estimator in the EstimatorCachingWrapper class.

        Parameters
        ----------
        estimator: BaseEstimator
            The sklearn estimator to wrap

        Returns
        -------
        EstimatorCachingWrapper
            The wrapped estimator.
            This can be used like a normal sklearn estimator but with appropriate
            caching to prevent re-running slow code.
        kwargs:
            Any other kwargs to define a given experiment run. For example, you might
            want to add a label or some other identifier to mark this run as a different
            one within experiment.
            This is the same as the kwargs in the __init__ but will only apply to this
            estimator. You can use the kwargs here instead, but if you have multiple
            estimators with the same kwargs then you can de-duplicate by using the
            kwargs in the __init__.
            The kwargs passed to the __init__ method and the __call__ method should not
            have any overlapping keys.
        """
        estimator_name = type(estimator).__name__
        logger.debug(
            f"Creating estimator wrapper for {estimator_name}",
        )
        self.validate_dict(kwargs)

        if any(
            overlap := set(self._init_kwargs.keys()).intersection(set(kwargs.keys()))
        ):
            msg = (
                f"The kwargs passed to the __init__ method and the __call__ method "
                f"should not have any overlapping keys, have {overlap=}"
            )
            raise ValueError(msg)

        return EstimatorCachingWrapper(
            estimator=estimator,
            experiment_save_path=self.base_dir
            / self.project
            / self.experiment
            / estimator_name
            / self.get_dictionary_hash({**self._init_kwargs, **kwargs}),
        )
