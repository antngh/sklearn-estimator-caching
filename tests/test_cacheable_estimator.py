import itertools
from pathlib import Path

import pytest
from data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator import (
    EstimatorCachingWrapper,
)
from sklearn.base import clone

from tests.utils.sklearn_pipelines.caching.conftest import DummyEstimator


class EstimatorCachingWrapperChild(EstimatorCachingWrapper):
    def some_other_method(self):
        return 999


@pytest.fixture(scope="function")
def raw_estimator_wrapped_estimator():
    raw_estimator = DummyEstimator(a=1, b=2)
    return raw_estimator, EstimatorCachingWrapper(
        estimator=raw_estimator,
        experiment_save_path=Path("experiment_save_path"),
    )


def test_EstimatorCachingWrapper_init(raw_estimator_wrapped_estimator):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    assert isinstance(wrapped_estimator.estimator, DummyEstimator)
    assert str(wrapped_estimator.experiment_save_path) == "experiment_save_path"
    assert wrapped_estimator.estimator.a == 1
    assert wrapped_estimator.estimator.b == 2
    assert wrapped_estimator.estimator is not raw_estimator
    assert dir(wrapped_estimator.estimator) == dir(raw_estimator)

    # assert that passing kwargs triggers creation of a new estimator
    # this needs to work for sklearn internally cloning to work properly
    wrapped_estimator2 = EstimatorCachingWrapper(
        estimator=raw_estimator,
        experiment_save_path="experiment_save_path",
        a=3,
        b=4,
    )
    assert isinstance(wrapped_estimator2.estimator, DummyEstimator)
    assert wrapped_estimator2.estimator is not raw_estimator
    assert wrapped_estimator2.estimator is not wrapped_estimator.estimator
    assert wrapped_estimator2.estimator.a == 3
    assert wrapped_estimator2.estimator.b == 4


def test_EstimatorCachingWrapper_cloning(raw_estimator_wrapped_estimator):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    cloned_wrapped_estimator = wrapped_estimator.__sklearn_clone__()
    cloned_wrapped_estimator2 = clone(wrapped_estimator)

    wrapped_estimators = [
        wrapped_estimator,
        cloned_wrapped_estimator,
        cloned_wrapped_estimator2,
    ]
    for wrapped_estimator_ in wrapped_estimators:
        assert isinstance(wrapped_estimator_, EstimatorCachingWrapper)
        assert isinstance(wrapped_estimator_.estimator, DummyEstimator)
        assert wrapped_estimator_.estimator is not raw_estimator
        assert wrapped_estimator_.estimator.a == 1
        assert wrapped_estimator_.estimator.b == 2

    for wrapped_estimator_1, wrapped_estimator_2 in itertools.combinations(
        wrapped_estimators, 2
    ):
        assert wrapped_estimator_1 is not wrapped_estimator_2
        assert wrapped_estimator_1.estimator is not wrapped_estimator_2.estimator
        assert wrapped_estimator_1.estimator.a == wrapped_estimator_2.estimator.a
        assert wrapped_estimator_1.estimator.b == wrapped_estimator_2.estimator.b
        assert dir(wrapped_estimator_1) == dir(wrapped_estimator_2)
        assert dir(wrapped_estimator_1.estimator) == dir(wrapped_estimator_2.estimator)


def test__EstimatorCachingWrapper_mimicks_estimator():
    # i.e. test getattr works to get the underlying estimator attributes if it doesn't exist in the wrapper
    raw_estimator = DummyEstimator(a=1, b=2)
    wrapped_estimator = EstimatorCachingWrapperChild(
        estimator=raw_estimator,
        experiment_save_path=Path("experiment_save_path"),
    )
    # check that if the wrapper doesn't have it, get the estimator attributes
    assert wrapped_estimator.a == wrapped_estimator.estimator.a == 1
    assert wrapped_estimator.b == wrapped_estimator.estimator.b == 2
    assert (
        wrapped_estimator.some_method()
        == wrapped_estimator.estimator.some_method()
        == 137
    )

    # check that the wrapper's method is called if it exists, otherwise child
    assert wrapped_estimator.some_other_method() == 999
    assert wrapped_estimator.estimator.some_other_method() == -10


# def test_test_EstimatorCachingWrapper_getattr_and_setattr(
#     raw_estimator_wrapped_estimator,
# ):
#     assert False
#     raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
#     wrapped_estimator.some_attr = 123
#     assert wrapped_estimator.some_attr == 123
#     assert wrapped_estimator.estimator.some_attr == 123
#     assert wrapped_estimator.some_attr == wrapped_estimator.estimator.some_attr

#     wrapped_estimator.some_attr = 456
#     assert wrapped_estimator.some_attr == 456
#     assert wrapped_estimator.estimator.some_attr == 456
#     assert wrapped_estimator.some_attr == wrapped_estimator.estimator.some_attr

#     del wrapped_estimator.some_attr
#     with pytest.raises(AttributeError):
#         _ = wrapped_estimator.some_attr
#     with pytest.raises(AttributeError):
#         _ = wrapped_estimator.estimator.some_attr

#     with pytest.raises(AttributeError):
#         _ = wrapped_estimator.some_non_existent_attr
#     with pytest.raises(AttributeError):
#         _ = wrapped_estimator.estimator.some_non_existent_attr


def test_EstimatorCachingWrapper_get_params(raw_estimator_wrapped_estimator):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    wrapper_params = wrapped_estimator.get_params(deep=True)
    assert {
        k: v for k, v in wrapper_params.items() if k not in ["estimator", "logger"]
    } == {
        "experiment_save_path": wrapped_estimator.experiment_save_path,
        "a": 1,
        "b": 2,
    }
    assert wrapper_params["estimator"] is not raw_estimator
    assert isinstance(wrapper_params["estimator"], DummyEstimator)
    assert dir(wrapper_params["estimator"]) == dir(raw_estimator)


def test_EstimatorCachingWrapper_check_estimator_method_exists(
    raw_estimator_wrapped_estimator,
):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    wrapped_estimator._validate_estimator_method(
        "some_method"
    )  # a method of the estimator
    with pytest.raises(NotImplementedError):
        wrapped_estimator._validate_estimator_method("some_non_existing_method")


def test_EstimatorCachingWrapper_predict_transform_runner_validates_input_process_data_func(
    raw_estimator_wrapped_estimator, mocker
):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    with pytest.raises(NotImplementedError):
        wrapped_estimator._predict_transform_runner(
            process_data_func_name="some_non_existent_inference_func",
            process_data_func_kwargs={},
        )


@pytest.mark.parametrize(
    "process_data_func, kwargs, metadata",
    [
        *[
            (process_data_func, kwargs, metadata)
            for process_data_func in ["transform", "predict"]
            for kwargs, metadata in [
                (
                    {},
                    '{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {}}',
                ),
                (
                    {"c": 3},
                    '{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"c": 3}}',
                ),
            ]
        ]
    ],
)
def test_EstimatorCachingWrapper_predict_transform_runner(
    raw_estimator_wrapped_estimator, mocker, process_data_func, kwargs, metadata
):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    mocked_load_process_data_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_process_data_from_file",
        return_value="loaded_data",
    )

    setattr(
        wrapped_estimator.estimator, process_data_func, lambda: "estimator_func_output"
    )

    assert (
        wrapped_estimator._predict_transform_runner(
            process_data_func_name=process_data_func, process_data_func_kwargs=kwargs
        )
        == "loaded_data"
    )
    mocked_load_process_data_from_file.assert_called_once_with(
        experiment_save_path=Path("experiment_save_path"),
        process_data_func_name=process_data_func,
        fitted_estimator_process_method_metadata=metadata,
    )

    def bad_file_load(*args, **kwargs):
        raise FileNotFoundError

    mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_process_data_from_file",
        side_effect=bad_file_load,
    )
    estimator_func = mocker.patch.object(
        wrapped_estimator.estimator,
        process_data_func,
        return_value="estimator_func_output_mock",
    )
    save_patch = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.save_process_data_to_file",
    )
    assert (
        wrapped_estimator._predict_transform_runner(
            process_data_func_name=process_data_func, process_data_func_kwargs=kwargs
        )
        == "estimator_func_output_mock"
    )
    estimator_func.assert_called_once_with(**kwargs)
    print(save_patch.call_args_list)
    save_patch.assert_called_once_with(
        experiment_save_path=Path("experiment_save_path"),
        process_data_func_name=process_data_func,
        save_data="estimator_func_output_mock",
        fitted_estimator_process_method_metadata=metadata,
    )


def test_EstimatorCachingWrapper_transform(raw_estimator_wrapped_estimator, mocker):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    mock_predict_transform_runner = mocker.patch.object(
        wrapped_estimator, "_predict_transform_runner", return_value="output"
    )
    wrapped_estimator.transform(X="input_data")
    mock_predict_transform_runner.assert_called_once_with(
        process_data_func_name="transform", process_data_func_kwargs={"X": "input_data"}
    )


def test_EstimatorCachingWrapper_predict(raw_estimator_wrapped_estimator, mocker):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    mock_predict_transform_runner = mocker.patch.object(
        wrapped_estimator, "_predict_transform_runner", return_value="output"
    )
    wrapped_estimator.predict(X="input_data", some="kwargs")
    mock_predict_transform_runner.assert_called_once_with(
        process_data_func_name="predict",
        process_data_func_kwargs={"X": "input_data", "some": "kwargs"},
    )


def test_EstimatorCachingWrapper_fit(raw_estimator_wrapped_estimator, mocker):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    setattr(wrapped_estimator.estimator, "fit", lambda: "some_method_output")

    mock_load_fitted_estimator_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_fitted_estimator_from_file",
        return_value="the_loaded_estimator",
    )
    wrapped_estimator_ = wrapped_estimator.fit(
        X="input_data",
        y="target_data",
        some="kwargs",
    )
    assert wrapped_estimator_ is wrapped_estimator
    assert wrapped_estimator.estimator == "the_loaded_estimator"
    mock_load_fitted_estimator_from_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        unfitted_estimator_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data", "some": "kwargs", "y": "target_data"}}',
    )

    wrapped_estimator.estimator = raw_estimator
    setattr(wrapped_estimator.estimator, "fit", lambda: "some_method_output")

    def raise_fnf_error(*args, **kwargs):
        raise FileNotFoundError

    mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_fitted_estimator_from_file",
        side_effect=raise_fnf_error,
    )
    mock_estimator_fit = mocker.patch.object(
        wrapped_estimator.estimator, "fit", return_value="the_fit_estimator"
    )
    mock_save_fitted_estimator_to_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.save_fitted_estimator_to_file",
    )

    wrapped_estimator_ = wrapped_estimator.fit(
        X="input_data", y="target_data", some="kwargs"
    )
    assert wrapped_estimator_ is wrapped_estimator
    assert wrapped_estimator.estimator == "the_fit_estimator"
    mock_load_fitted_estimator_from_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        unfitted_estimator_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data", "some": "kwargs", "y": "target_data"}}',
    )
    mock_estimator_fit.assert_called_once_with(
        X="input_data", y="target_data", some="kwargs"
    )
    mock_save_fitted_estimator_to_file.assert_called_once_with(
        estimator=wrapped_estimator.estimator,
        experiment_save_path=wrapped_estimator.experiment_save_path,
        unfitted_estimator_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data", "some": "kwargs", "y": "target_data"}}',
    )


def test_EstimatorCachingWrapper_fit_predict(raw_estimator_wrapped_estimator, mocker):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    mock_fit_and_process_runner = mocker.patch.object(
        wrapped_estimator, "_fit_and_process_runner", return_value="output"
    )
    assert (
        wrapped_estimator.fit_predict(X="input_data", y="target_data", some="kwargs")
        == "output"
    )
    mock_fit_and_process_runner.assert_called_once_with(
        process_data_func_name="predict", X="input_data", y="target_data", some="kwargs"
    )


def test_EstimatorCachingWrapper_fit_transform(raw_estimator_wrapped_estimator, mocker):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    mock_fit_and_process_runner = mocker.patch.object(
        wrapped_estimator, "_fit_and_process_runner", return_value="output"
    )
    assert (
        wrapped_estimator.fit_transform(X="input_data", y="target_data", some="kwargs")
        == "output"
    )
    mock_fit_and_process_runner.assert_called_once_with(
        process_data_func_name="transform",
        X="input_data",
        y="target_data",
        some="kwargs",
    )


@pytest.mark.parametrize("process_func", ["predict", "transform"])
def test_EstimatorCachingWrapper_fit_and_process_runner_when_estimator_only_has_fit_and_separate_func_methods(
    raw_estimator_wrapped_estimator, mocker, process_func
):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator

    setattr(wrapped_estimator.estimator, "fit", lambda: wrapped_estimator.estimator)

    # in fit_and_process_runner, we always fit, so process_func refers to the inference function
    with pytest.raises(ValueError):
        wrapped_estimator._fit_and_process_runner(
            process_data_func_name="fit", X="input_data"
        )

    setattr(wrapped_estimator.estimator, process_func, lambda: "some_inference_output")
    mock_fit = mocker.patch.object(
        wrapped_estimator, "fit", return_value=wrapped_estimator
    )
    mock_estimator_func = mocker.patch.object(
        wrapped_estimator, process_func, return_value="some_inference_output"
    )
    assert (
        getattr(wrapped_estimator, f"fit_{process_func}")(
            X="input_data", y="target_data", some="kwargs"
        )
        == "some_inference_output"
    )
    mock_fit.assert_called_once_with(X="input_data", y="target_data", some="kwargs")
    mock_estimator_func.assert_called_once_with(X="input_data")


@pytest.mark.parametrize("process_func", ["predict", "transform"])
def test_EstimatorCachingWrapper_fit_and_process_runner_when_has_fit_func_methods_and_estimator_was_already_saved_on_previous_fit_or_fit_func(
    raw_estimator_wrapped_estimator, mocker, process_func
):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    setattr(wrapped_estimator.estimator, "fit", lambda: wrapped_estimator.estimator)

    setattr(wrapped_estimator.estimator, process_func, lambda: "some_inference_output")
    setattr(
        wrapped_estimator.estimator,
        f"fit_{process_func}",
        lambda: "some_inference_output",
    )

    mock_load_fitted_estimator_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_fitted_estimator_from_file",
        return_value=wrapped_estimator.estimator,
    )
    mock_load_process_data_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_process_data_from_file",
        return_value="loaded_data",
    )
    assert (
        getattr(wrapped_estimator, f"fit_{process_func}")(
            X="input_data", y="target_data", some="kwargs"
        )
        == "loaded_data"
    )
    mock_load_fitted_estimator_from_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        unfitted_estimator_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data", "some": "kwargs", "y": "target_data"}}',
    )
    mock_load_process_data_from_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        process_data_func_name=process_func,
        fitted_estimator_process_method_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data"}}',
    )

    def raise_fnf_error(*args, **kwargs):
        raise FileNotFoundError

    mock_load_process_data_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_process_data_from_file",
        side_effect=raise_fnf_error,
    )
    mock_estimator_func = mocker.patch.object(
        wrapped_estimator, process_func, return_value="some_inference_output"
    )
    assert (
        getattr(wrapped_estimator, f"fit_{process_func}")(
            X="input_data", y="target_data", some="kwargs"
        )
        == "some_inference_output"
    )
    mock_load_process_data_from_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        process_data_func_name=process_func,
        fitted_estimator_process_method_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data"}}',
    )
    mock_estimator_func.assert_called_once_with(X="input_data")


@pytest.mark.parametrize("process_func", ["predict", "transform"])
def test_EstimatorCachingWrapper_fit_and_process_runner_when_has_fit_func_methods_and_is_first_run(
    raw_estimator_wrapped_estimator, mocker, process_func
):
    raw_estimator, wrapped_estimator = raw_estimator_wrapped_estimator
    setattr(wrapped_estimator.estimator, "fit", lambda: wrapped_estimator.estimator)

    setattr(wrapped_estimator.estimator, process_func, lambda: "some_inference_output")
    setattr(
        wrapped_estimator.estimator,
        f"fit_{process_func}",
        lambda: "some_inference_output",
    )

    def raise_fnf_error(*args, **kwargs):
        raise FileNotFoundError

    mock_load_fitted_estimator_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_fitted_estimator_from_file",
        side_effect=raise_fnf_error,
    )
    mock_estimator_fit_and_func = mocker.patch.object(
        wrapped_estimator.estimator,
        f"fit_{process_func}",
        return_value="some_fit_and_inference_output",
    )
    mock_save_fitted_estimator_to_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.save_fitted_estimator_to_file",
    )
    mock_save_process_data_to_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.save_process_data_to_file",
    )
    mock_load_process_data_from_file = mocker.patch(
        "data_science_common.utils.sklearn_pipelines.caching.cacheable_estimator.load_process_data_from_file",
        return_value="loaded_data",
    )
    assert (
        getattr(wrapped_estimator, f"fit_{process_func}")(
            X="input_data", y="target_data", some="kwargs"
        )
        == "some_fit_and_inference_output"
    )
    mock_load_fitted_estimator_from_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        unfitted_estimator_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data", "some": "kwargs", "y": "target_data"}}',
    )
    mock_estimator_fit_and_func.assert_called_once_with(
        X="input_data", y="target_data", some="kwargs"
    )
    mock_save_fitted_estimator_to_file.assert_called_once_with(
        estimator=wrapped_estimator.estimator,
        experiment_save_path=wrapped_estimator.experiment_save_path,
        unfitted_estimator_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data", "some": "kwargs", "y": "target_data"}}',
    )
    mock_save_process_data_to_file.assert_called_once_with(
        experiment_save_path=wrapped_estimator.experiment_save_path,
        save_data="some_fit_and_inference_output",
        process_data_func_name=process_func,
        fitted_estimator_process_method_metadata='{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"X": "input_data"}}',
    )
    mock_load_process_data_from_file.assert_not_called()
