"""
Code for wrapping sklearn estimators to make them automatically cache the results of
fit/predict/transform/fit_predict/fit_transform to disk.

Use EstimatorCachingWrapperGetter and see its docstring for details
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import structlog
from sklearn.base import BaseEstimator, clone

from .cache_utils import (
    get_metadata,
    load_fitted_estimator_from_file,
    load_processed_data_from_file,
    save_fitted_estimator_to_file,
    save_processed_data_to_file,
    validate_func_name,
)
from .config import DataType

logger = structlog.get_logger()


class EstimatorCachingWrapper:
    """
    A wrapper for sklearn estimators that caches the results of fit/predict/transform
    to disk.

    You probably don't want to use this directly, use EstimatorCachingWrapperGetter
    instead.

    Attributes
    ----------
    estimator: BaseEstimator
        The sklearn estimator to wrap
    experiment_save_path: Path
        The path to save the cached data to.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        experiment_save_path: Path,
        **_kwargs,
    ):
        """
        Set up the EstimatorCachingWrapper

        Either clones the input estimator or makes a new one from _kwargs, the latter is
        for internal logic, don't use it.

        Parameters
        ----------
        estimator: BaseEstimator
            The sklearn estimator to wrap
        experiment_save_path: Path
            The path to save the cached data to.
        _kwargs:
            DO NOT USE THIS.
            This is required for the case when sklearn decides to clone the estimator
            behind the scenes.
        """
        # DON'T USE KWARGS. This is needed for when sklearn decides to clone it behind
        # the scenes
        self.estimator: BaseEstimator = (
            clone(estimator) if not _kwargs else type(estimator)(**_kwargs)
        )

        self.experiment_save_path = experiment_save_path

        logger.debug(
            f"Created estimator wrapper that wraps a {self._estimator_classname} "
            f"instance, using  {self.experiment_save_path=}"
        )

    @property
    def _estimator_class(self) -> BaseEstimator:
        """
        Get the class of the estimator

        Returns
        -------
        BaseEstimator
            The class of the estimator
        """
        return type(self.estimator)

    @property
    def _estimator_classname(self) -> str:
        """
        Get the classname of the estimator

        Returns
        -------
        str
            The classname of the estimator
        """
        return self._estimator_class.__name__

    def __sklearn_clone__(self) -> EstimatorCachingWrapper:
        """
        Clone itself. Needed so that when the estimator is cloned it returns
        EstimatorCachingWrapper instead of the clone of the estimator itself.

        Without this, any clone will lose all the caching logic (and will be a normal
        sklearn estimator).

        Returns
        -------
        EstimatorCachingWrapper
            A clone of itself.
        """
        logger.info(
            "Cloning estimator wrapper",
        )
        return type(self)(
            estimator=self.estimator,  # this will be cloned on init
            experiment_save_path=self.experiment_save_path,
        )

    def __getattr__(self, attr: str) -> Any:
        """
        This method is what allows composition to work.

        __getattr__ is called only when an explicit attribute can't be found (contrast
        to __getattribute__), so if EstimatorCachingWrapper doesn't have a method, it
        will then look for it in the estimator itself.

        This means that the EstimatorCachingWrapper instance looks and behaves like the
        estimator itself, but with the extra methods and caching logic added.

        Parameters
        ----------
        attr: str
            The attribute to get from the estimator

        Returns
        -------
        Any
            The attribute (property or method) from the estimator
        """
        # Delegate attribute access to the original estimator.
        # This allows this wrapper to be called internally by sklearn as if it
        # was the estimator itself
        return getattr(self.estimator, attr)

    def __setattr__(self, attr: str, value: Any):
        """
        Set an attribute to the correct class instance.

        If the attribute is a property of the wrapper and not the estimator then set it
        as an attribute of the wrapper. Otherwise if the estimator has that property,
        or neither the wrapper nor the estimator have the property, then set it as an
        attribute of the estimator.

        Parameters
        ----------
        attr: str
            The attribute to set
        value: Any
            The value to set the attribute to
        """
        # hasattr called on the wrapper will be true if either the wrapper or the
        # estimator has the attribute
        if hasattr(self, attr) and not hasattr(self.estimator, attr):
            object.__setattr__(self, attr, value)
        else:
            setattr(self.estimator, attr, value)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for the estimator and the wrapper

        Parameters
        ----------
        deep: bool
            If True, will return the parameters for this estimator and contained
            sub-objects that are estimators.

        Returns
        -------
        dict[str, Any]
            Estimator parameter names mapped to their values plus the parameters of the
            wrapper (which includes the estimator itself).
        """
        logger.debug(
            f"Getting the params of the estimator wrapper and the estimator with "
            f"{deep=}"
        )
        return {
            "estimator": self.estimator,
            "experiment_save_path": self.experiment_save_path,
            **self.estimator.get_params(deep=deep),
        }

    def set_params(self, **params) -> EstimatorCachingWrapper:
        """
        Set the parameters for this estimator and the wrapper.

        Parameters
        ----------
        params:
            The parameters to set. Will also work if providing the parameters of the
            wrapper i.e. the estimator and the experiment save path.
            Do NOT pass in an estimator of a different type to the original.
        """
        for attr, type_ in [
            (
                "estimator",
                self._estimator_class,
            ),
            (
                "experiment_save_path",
                Path,
            ),
        ]:
            input_ = params.pop(attr, getattr(self, attr))
            if not isinstance(input_, type_):
                msg = f"{attr} must be an instance of {type_}, have {type(input_)}"
                raise TypeError(msg)
            setattr(self, attr, input_)

        self.estimator = self.estimator.set_params(**params)
        return self

    def __repr__(self) -> str:
        """
        Overwrite the repr of the estimator to include the wrapper.

        Returns
        -------
        str
            The string representation of the estimator and the wrapper
        """
        return f"{type(self).__name__}({repr(self.estimator)})"

    @property
    def _repr_html_(self) -> str:
        """
        Overwrite the _repr_html_ of the estimator to pass in the wrapped estimator

        Returns
        -------
        str
            The html representation of the estimator and the wrapper
        """
        # fget gets the property getter method
        return getattr(self._estimator_class, "_repr_html_").fget(self)

    def _repr_html_inner(self) -> str:
        """
        Overwrite the _repr_html_inner of the estimator to pass in the wrapped estimator

        Returns
        -------
        str
            The html representation of the estimator and the wrapper
        """
        return getattr(self._estimator_class, "_repr_html_inner")(self)

    def _repr_mimebundle_(self, **kwargs) -> dict[str, str]:
        """
        Overwrite the _repr_mimebundle_ of the estimator to pass in the wrapped
        estimator

        Parameters
        ----------
        kwargs:
            Any kwargs to pass to the estimator _repr_mimebundle_ method

        Returns
        -------
        dict[str, str]
            The mimebundle representation of the estimator and the wrapper
        """
        return getattr(self._estimator_class, "_repr_mimebundle_")(self, **kwargs)

    def __getstate__(self) -> dict:
        """
        Get the state of the estimator and the wrapper.

        This is used for pickling (which in turn is needed for parallel processing).

        Returns
        -------
        dict
            The state of the estimator and the wrapper to get
        """
        state = self.estimator.__getstate__()
        state.update(
            {
                "estimator": self.estimator,
                "experiment_save_path": self.experiment_save_path,
            }
        )
        return state

    def __setstate__(self, state: dict):
        """
        Set the state of the estimator and the wrapper.

        This is used for un-pickling (which in turn is needed for parallel processing).

        Parameters
        ----------
        state: dict
            The state of the estimator and the wrapper to set
        """
        self.estimator = state.pop("estimator")
        self.experiment_save_path = state.pop("experiment_save_path")
        self.estimator.__setstate__(state)

    def _validate_estimator_method(self, method: str):
        """
        Check that the sklearn estimator has a given method.

        Parameters
        ----------
        method: str
            The method to check for

        Raises
        ------
        NotImplementedError
            If the method doesn't exist
        """
        logger.debug(
            f"Checking that the estimator has the method {method}",
        )
        if not hasattr(self.estimator, method):
            msg = f"The estimator {self._estimator_classname} has no {method} method"
            raise NotImplementedError(msg)

    def _predict_transform_runner(
        self,
        process_data_func_name: Literal["predict", "transform"],
        process_data_func_kwargs: dict,
    ) -> DataType:
        """
        A utility function to run the predict/transform methods.

        If the data has been cached to disk from a run of the same method with the same
        args/kwargs (and class/instance setup) then load and return it, otherwise run
        the estimators predict/transform method and save the data.

        Parameters
        ----------
        process_data_func_name: Literal["predict", "transform"]
            The function that was called i.e. predict/transform
        process_data_func_kwargs: dict
            The args/kwargs that were passed to the function call

        Returns
        -------
        DataType
            The data output from the predict/transform method
        """
        self._validate_estimator_method(process_data_func_name)

        output_data: DataType
        fitted_estimator_process_metadata = get_metadata(
            estimator=self.estimator, **process_data_func_kwargs
        )
        try:
            output_data = load_processed_data_from_file(
                experiment_save_path=self.experiment_save_path,
                process_data_func_name=process_data_func_name,
                fitted_estimator_process_metadata=fitted_estimator_process_metadata,
            )
        except FileNotFoundError:
            pass
        else:
            logger.warning(
                (
                    f"Loaded cached data from file, so not running "
                    f"{self._estimator_classname}.{process_data_func_name}, "
                    f"{process_data_func_name=}"
                )
            )
            return output_data

        logger.warning(
            f"Unable to load cached data from file, so running "
            f"{self._estimator_classname}.{process_data_func_name}, "
            f"{process_data_func_name=}"
        )
        output_data = getattr(self.estimator, process_data_func_name)(
            **process_data_func_kwargs
        )
        save_processed_data_to_file(
            experiment_save_path=self.experiment_save_path,
            process_data_func_name=process_data_func_name,
            save_data=output_data,
            fitted_estimator_process_metadata=fitted_estimator_process_metadata,
        )

        return output_data

    def _fit_and_process_runner(
        self,
        process_data_func_name: Literal["predict", "transform"],
        X: DataType,
        y: DataType | None = None,
        **fit_params,
    ) -> DataType:
        """
        A utility function to run the fit_predict/fit_transform methods.

        The logic is the same as _predict_transform_runner but with extra logic to call
        the fit_predict/fit_transform methods if they exist, otherwise simply call fit
        and then predict/transform.

        Note that predict params is not an option because sklearn fit_predict doesn't
        accept them. Separately call fit and predict if you want to use them.

        Parameters
        ----------
        process_data_func_name: Literal["predict", "transform"]
            The function that was called (predict/transform) along with fit (i.e.
            "predict" for "fit_predict")
        X: DataType
            The input data to fit the estimator and predict/transform with
        y: DataType | None
            The optional target data to fit the estimator with
        fit_params:
            Any other kwargs to pass to the estimator fit method

        Returns
        -------
        DataType
            The data output from the fit_predict/fit_transform method (which will be
            identical to the output of the predict/transform method)
        """
        validate_func_name(estimator_func_name=process_data_func_name)

        self._validate_estimator_method("fit")
        self._validate_estimator_method(process_data_func_name)

        fit_and_process_func_name = f"fit_{process_data_func_name}"
        # if no fit_predict or fit_transform method then call fit first and then the
        # corresponding method
        if not hasattr(self.estimator, fit_and_process_func_name):
            logger.info(
                f"{self._estimator_classname} doesn't have {fit_and_process_func_name},"
                f" so calling fit and then {process_data_func_name}, "
                f"{process_data_func_name=}"
            )

            # Call EstimatorCachingWrapper fit method, saving/loading the estimator as
            # needed
            self.fit(X=X, y=y, **fit_params)
            # call the EstimatorCachingWrapper predict/transform method (saving/loading
            # the data as needed)
            return getattr(self, process_data_func_name)(X=X)

        # We handle the following case separately because fit_x can often be faster than
        # falling fit and x separately depending on the sklearn estimator, in any case
        # it should never be slower and we want to allow to use those underlying methods
        unfitted_estimator_fit_method_metadata = get_metadata(
            # clone to get an unfitted copy in case that it had already been fit
            estimator=clone(self.estimator),
            X=X,
            y=y,
            **fit_params,
        )
        try:
            # load the previously fitted estimator
            loaded_estimator = load_fitted_estimator_from_file(
                experiment_save_path=self.experiment_save_path,
                unfitted_estimator_metadata=unfitted_estimator_fit_method_metadata,
            )
        except FileNotFoundError:
            logger.warning(
                f"Failed to load fitted estimator from file, so running "
                f"{self._estimator_classname}.{fit_and_process_func_name}"
            )

            # If the estimator doesn't exist then we don't care if the data does, re-run
            # the processing anyway.
            # Call the estimators fit_predict/fit_transform method, save the object and
            # data
            data = getattr(self.estimator, fit_and_process_func_name)(
                X=X, y=y, **fit_params
            )
            save_fitted_estimator_to_file(
                estimator=self.estimator,
                experiment_save_path=self.experiment_save_path,
                unfitted_estimator_metadata=unfitted_estimator_fit_method_metadata,
            )  # this saves the fit estimator.

            fitted_estimator_process_metadata = get_metadata(
                estimator=self.estimator, X=X
            )
            save_processed_data_to_file(
                experiment_save_path=self.experiment_save_path,
                save_data=data,
                process_data_func_name=process_data_func_name,
                fitted_estimator_process_metadata=fitted_estimator_process_metadata,
            )
            return data

        logger.warning(
            f"Loaded fitted estimator from file, so not running "
            f"{self._estimator_classname}.{fit_and_process_func_name}, "
            f"{process_data_func_name=}"
        )

        self.estimator = loaded_estimator
        fitted_estimator_process_metadata = get_metadata(estimator=self.estimator, X=X)
        try:
            data = load_processed_data_from_file(
                experiment_save_path=self.experiment_save_path,
                process_data_func_name=process_data_func_name,
                fitted_estimator_process_metadata=fitted_estimator_process_metadata,
            )
        except FileNotFoundError:
            # if the estimator exists but the data doesn't then call the
            # EstimatorCachingWrapper predict/transform method (saving the data)
            data = getattr(self, process_data_func_name)(X=X)

            logger.warning(
                f"Failed to load data from file, so running "
                f"{self._estimator_classname}.{process_data_func_name}, "
                f"{process_data_func_name=}"
            )
        else:
            logger.warning(
                f"Loaded data from file, so not running "
                f"{self._estimator_classname}.{process_data_func_name}, "
                f"{process_data_func_name=}"
            )
        return data

    def fit(
        self, X: DataType, y: DataType | None = None, **fit_params
    ) -> EstimatorCachingWrapper:
        """
        Find a previously fit estimator (with the same args/kwargs as this fit function
        call and class/instance setup) if it exists. If it doesn't exist then fit the
        estimator and save it to disk.

        Parameters
        ----------
        X: DataType
            The input data to fit the estimator with
        y: DataType | None
            The optional target data to fit the estimator with
        fit_params:
            Any other kwargs to pass to the estimator fit method
        """
        self._validate_estimator_method("fit")

        fit_kwargs = {"X": X, "y": y, **fit_params}

        unfitted_estimator_metadata = get_metadata(
            # clone to get an unfitted copy in case that it had already been fit
            estimator=clone(self.estimator),
            **fit_kwargs,
        )
        try:
            self.estimator = load_fitted_estimator_from_file(
                experiment_save_path=self.experiment_save_path,
                unfitted_estimator_metadata=unfitted_estimator_metadata,
            )
        except FileNotFoundError:
            logger.warning(
                f"Failed to load fitted estimator from file, so running "
                f"{self._estimator_classname}.fit"
            )

            self.estimator = self.estimator.fit(**fit_kwargs)
            save_fitted_estimator_to_file(
                estimator=self.estimator,
                experiment_save_path=self.experiment_save_path,
                unfitted_estimator_metadata=unfitted_estimator_metadata,
            )
        else:
            logger.warning(
                f"Loaded fitted estimator from file, so not running "
                f"{self._estimator_classname}.fit"
            )

        return self

    def transform(self, X: DataType) -> DataType:
        """
        Run any caching logic for the transform method, calling the estimators transform
        method if needed.

        Parameters
        ----------
        X: DataType
            The input data to transform

        Returns
        -------
        DataType
            The transformed data
        """
        logger.debug(f"Calling {self.transform.__name__}")
        return self._predict_transform_runner(
            process_data_func_name="transform",  # Redefining as it expects a literal
            process_data_func_kwargs={"X": X},
        )

    def predict(self, X: DataType, **predict_params) -> DataType:
        """
        Run any caching logic for the predict method, calling the estimators predict
        method if needed.

        Parameters
        ----------
        X: DataType
            The input data to predict with
        predict_params:
            Any other kwargs to pass to the estimator predict method

        Returns
        -------
        DataType
            The models predictions
        """
        logger.debug(
            f"Calling {self.predict.__name__}",
        )
        return self._predict_transform_runner(
            process_data_func_name="predict",  # Redefining as it expects a literal
            process_data_func_kwargs={"X": X, **predict_params},
        )

    def fit_transform(
        self, X: DataType, y: DataType | None = None, **fit_params
    ) -> DataType:
        """
        A utility function to run the fit_transform method

        Will apply any caching logic and if needed call the sklearn estimators
        fit_transform method (or fit and then transform method) depending on what has
        already been cached.

        Parameters
        ----------
        X: DataType
            The input data to fit the estimator and transform with
        y: DataType | None
            The optional target data to fit the estimator with
        fit_params:
            Any other kwargs to pass to the estimator fit method

        Returns
        -------
        DataType
            The data output from the fit_transform method (which will be identical to
            the output of the transform method of the already fit estimator).
        """
        logger.debug(
            f"Calling {self.fit_transform.__name__}",
        )
        return self._fit_and_process_runner(
            process_data_func_name="transform",
            X=X,
            y=y,
            **fit_params,
        )

    def fit_predict(
        self, X: DataType, y: DataType | None = None, **fit_params
    ) -> DataType:
        """
        A utility function to run the fit_predict method

        Will apply any caching logic and if needed call the sklearn estimators
        fit_predict method (or fit and then predict method) depending on what has
        already been cached.

        Parameters
        ----------
        X: DataType
            The input data to fit the estimator and predict with
        y: DataType | None
            The optional target data to fit the estimator with
        fit_params:
            Any other kwargs to pass to the estimator fit method

        Returns
        -------
        DataType
            The data output from the fit_predict method (which will be identical to the
            output of the predict method of the already fit estimator).
        """
        logger.debug(
            f"Calling {self.fit_predict.__name__}",
        )
        return self._fit_and_process_runner(
            process_data_func_name="predict",
            X=X,
            y=y,
            **fit_params,
        )
