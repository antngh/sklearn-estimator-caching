import tempfile

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_estimator_caching import EstimatorCachingWrapperGetter


class RandomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_of_calls_to_fit = 0
        self.num_of_calls_to_transform = 0

    def fit(self, X, y=None):
        self.num_of_calls_to_fit += 1
        return self

    def transform(self, X):
        self.num_of_calls_to_transform += 1
        random_data = np.random.rand(*X.shape)
        return random_data


@pytest.fixture
def example_x_matrix():
    return np.array([[1, 2, 3], [1, 1, 1], [2, 1, 3]])


@pytest.fixture
def example_classification_data():
    X, y = make_classification(n_samples=20, n_features=5, random_state=1)
    return X, y


def test_estimator_caching_inside_sklearn_pipeline_fit_transform(example_x_matrix):
    """
    Checks the behaviour of cacheable estimators used in sklearn pipeline.
    Verifies that calling `fit_transform` on pipeline object with the same data twice
    will result with using cached values. When `fit_transform` is called with different
    data estimator should be rerun.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        wrapper = EstimatorCachingWrapperGetter(
            "project",
            "experiment",
            temp_dir,
        )
        cached_random_transformer = wrapper(RandomTransformer())
        cached_std_scaler = wrapper(StandardScaler())

        pipeline = Pipeline(
            [
                ("random_transformer", cached_random_transformer),
                ("scaler", cached_std_scaler),
            ]
        )

        assert (
            cached_random_transformer.num_of_calls_to_transform
            == cached_random_transformer.num_of_calls_to_fit
            == 0
        )
        fit_transform_result_1 = pipeline.fit_transform(example_x_matrix)

        assert (
            cached_random_transformer.num_of_calls_to_transform
            == cached_random_transformer.num_of_calls_to_fit
            == 1
        )
        fit_transform_result_2 = pipeline.fit_transform(example_x_matrix)

        assert (
            cached_random_transformer.num_of_calls_to_transform
            == cached_random_transformer.num_of_calls_to_fit
            == 1
        ), (
            "Fit/transform shouldn't be triggered when running pipeline "
            "with the same data more than once"
        )

        assert np.array_equal(
            fit_transform_result_1, fit_transform_result_2
        ), "Pipeline fit results should be the same"

        different_input_data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        fit_transform_result_different_data = pipeline.fit_transform(
            different_input_data
        )
        assert not np.array_equal(
            fit_transform_result_1, fit_transform_result_different_data
        ), "Pipeline output with different input data should produce different results"


def test_estimator_caching_inside_sklearn_pipeline_predict(example_classification_data):
    """
    Checks the behaviour of cacheable estimators used in sklearn pipeline.
    Verifies that calling `predict` on pipeline object with the same data twice
    will result with using cached values. When `predict` is called with different data
    estimator should be rerun.
    """
    X, y = example_classification_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        wrapper = EstimatorCachingWrapperGetter(
            "project",
            "experiment",
            temp_dir,
        )
        cached_scaler = wrapper(StandardScaler())
        cached_classifier = wrapper(DummyClassifier(strategy="uniform"))

        pipeline = Pipeline(
            [("scaler", cached_scaler), ("classifier", cached_classifier)]
        )

        pipeline.fit(X_train, y_train)

        predictions_1 = pipeline.predict(X_test)

        pipeline.fit(X_train, y_train)
        predictions_2 = pipeline.predict(X_test)

        assert np.array_equal(
            predictions_1, predictions_2
        ), "Pipeline predictions should be the same when using same input data"

        X_new, y_new = make_classification(n_samples=20, n_features=5, random_state=2)
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_new, y_new, test_size=0.3, random_state=2
        )

        pipeline.fit(X_train_new, y_train_new)
        predictions_new = pipeline.predict(X_test_new)

        assert not np.array_equal(
            predictions_1, predictions_new
        ), "Pipeline predictions with different data should be different"
