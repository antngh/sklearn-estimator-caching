from pathlib import Path

import pytest

from sklearn_estimator_caching.cacheable_estimator import EstimatorCachingWrapper
from sklearn_estimator_caching.estimator_caching_wrapper_getter import (
    EstimatorCachingWrapperGetter,
)
from tests.conftest import DummyEstimator


@pytest.fixture
def estimator_cacher():
    return EstimatorCachingWrapperGetter(
        project="project",
        experiment="experiment",
        base_dir="/base_dir",
        something_="something",
        else_="else",
    )


def test_EstimatorCacherGetter_init(estimator_cacher):
    assert estimator_cacher.project == "project"
    assert estimator_cacher.experiment == "experiment"
    assert estimator_cacher.base_dir == Path("/base_dir")
    assert estimator_cacher._init_kwargs == {
        "something_": "something",
        "else_": "else",
    }


def test_EstimatorCacherGetter_errors_on_non_string_kwargs(estimator_cacher):
    with pytest.raises(TypeError):
        EstimatorCachingWrapperGetter(
            project="project",
            experiment="experiment",
            base_dir="/base_dir",
            something_=123,  # should be a string
            else_="else",
        )


def test_EstimatorCacherGetter_call_errors_on_non_string_kwargs(estimator_cacher):
    with pytest.raises(TypeError):
        estimator_cacher(DummyEstimator(a=1, b=2), new_kwarg_=123)  # should be a string


def test_EstimatorCacherGetter_call_errors_on_overlapping_kwargs_with_init(
    estimator_cacher,
):
    with pytest.raises(ValueError):
        estimator_cacher(
            DummyEstimator(a=1, b=2),
            something_="something",  # this kwarg is already in the init kwargs
        )


def test_EstimatorCacherGetter_call_return_object(
    estimator_cacher,
):
    """
    Test that the EstimatorCacherGetter returns an EstimatorCachingWrapper object.

    It should not error on call with a new kwarg that isn't in the init kwargs

    It should properly create a EstimatorCachingWrapper instance where the dummy
    estimator is stored as an attribute, and the save path should reflect it.
    """
    estimator_cacher(
        DummyEstimator(a=1, b=2),
        new_kwarg_="something_new",  # this kwarg is not in the init kwargs
    )
    wrapped_estimator: EstimatorCachingWrapper = estimator_cacher(
        DummyEstimator(a=1, b=2)
    )
    assert isinstance(wrapped_estimator, EstimatorCachingWrapper)
    assert isinstance(wrapped_estimator.estimator, DummyEstimator)
    assert wrapped_estimator.estimator.a == 1
    assert wrapped_estimator.estimator.b == 2
    assert str(wrapped_estimator.experiment_save_path).startswith(
        "/base_dir/project/experiment/DummyEstimator/"
    )


def test_EstimatorCacherGetter_impact_of_kwargs_on_call_return_object():
    estimator = DummyEstimator(a=1, b=2)
    setup = dict(
        project="project",
        experiment="experiment",
        base_dir="/base_dir",
    )

    def _get_experiment_path(init_kwargs, call_kwargs):
        return EstimatorCachingWrapperGetter(
            **setup,
            **init_kwargs,
        )(estimator, **call_kwargs).experiment_save_path

    p1 = _get_experiment_path(dict(something_="something", else_="else"), {})
    p2 = _get_experiment_path(dict(something_="something"), dict(else_="else"))
    p3 = _get_experiment_path({}, dict(something_="something", else_="else"))

    assert p1 == p2 == p3

    p1_different = _get_experiment_path(
        dict(something_="something_different", else_="else"), {}
    )
    p3_different = _get_experiment_path(
        {}, dict(something_="something_different", else_="else")
    )

    assert p1_different == p3_different
    assert p1 != p1_different
