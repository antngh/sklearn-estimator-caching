import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from sklearn_estimator_caching.cache_utils import (
    DataTypePacking,
    _get_object_save_path,
    get_metadata,
    load_fitted_estimator_from_file,
    load_processed_data_from_file,
    save_fitted_estimator_to_file,
    save_processed_data_to_file,
    validate_func_name,
)
from tests.conftest import DummyEstimator

example_data_type_equality_check = [
    (
        pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pl.DataFrame,
        pl_testing.assert_frame_equal,
    ),
    (
        pl.Series([1, 2, 3]),
        pl.Series,
        pl_testing.assert_series_equal,
    ),
    (
        np.array([1, 2, 3]),
        np.ndarray,
        np.testing.assert_equal,
    ),
    (
        np.array([[1, 2], [3, 4]]),
        np.ndarray,
        np.testing.assert_equal,
    ),
    (
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        np.ndarray,
        np.testing.assert_equal,
    ),
    (
        np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
            ]
        ),
        np.ndarray,
        np.testing.assert_equal,
    ),
    (
        pd.DataFrame({"a": [1, 2]}, index=[2, 3]),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.DataFrame({"a": [1, 2]}, index=[2, 3]).astype("string"),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.DataFrame({None: [1, 2]}, index=[2, 3]),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[2, 3]),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).set_index(["a", "b"]),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.DataFrame({1: [1, 2], 2: [3, 4]}, index=[2, 3]),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.DataFrame({1.111: [1, 2], 2.222: [3, 4]}, index=[2, 3]),
        pd.DataFrame,
        pd.testing.assert_frame_equal,
    ),
    (
        pd.Series([1, 2, 3], index=[8, 9, 10], name=None),
        pd.Series,
        pd.testing.assert_series_equal,
    ),
    (
        pd.Series([1, 2, 3], index=[8, 9, 10], name="some name"),
        pd.Series,
        pd.testing.assert_series_equal,
    ),
    (
        pd.Series([1, 2, 3], index=[8, 9, 10], name=1),
        pd.Series,
        pd.testing.assert_series_equal,
    ),
    (
        pd.Series([1, 2, 3], index=[8, 9, 10], name=1.111),
        pd.Series,
        pd.testing.assert_series_equal,
    ),
    (
        pd.Series(
            [1, 2, 3],
            index=[
                pd.Timestamp("2023"),
                pd.Timestamp("2027"),
                pd.Timestamp("2024"),
            ],
        ),
        pd.Series,
        pd.testing.assert_series_equal,
    ),
]


@pytest.fixture
def temp_directory(tmp_path):
    directory = Path(tmp_path)
    return directory


@pytest.mark.parametrize(
    "obj, type_, name",
    [
        (pd.DataFrame(), pd.DataFrame, "data_pandas_df"),
        (pd.Series(dtype="float64"), pd.Series, "data_pandas_series"),
        (pl.DataFrame(), pl.DataFrame, "data_polars_df"),
        (pl.Series(), pl.Series, "data_polars_series"),
        (np.array([1, 2, 3]), np.ndarray, "data_numpy_array"),
    ],
)
def test_DataTypeFileMapping_get_from_object(obj, type_, name):
    """
    Test that the get_from_object method correctly returns the correct mapping entry
    """
    res_type_, res_name, res_packer, res_unpacker = DataTypePacking.get_from_object(obj)
    assert isinstance(obj, res_type_)
    assert res_name == name
    assert res_packer is not None
    assert res_unpacker is not None


def test_DataTypeFileMapping_get_from_object_raises_error_with_bad_object():
    with pytest.raises(TypeError):
        DataTypePacking.get_from_object("a non DataType object")


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame({"a": ["1", 2], "b": [3, 4]}),
        pd.DataFrame(
            {"a": ["1", "2"], "b": ["3", "4"]}
        ),  # is object by default, needs to be cast to "string" not str
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}, dtype="object"),
    ],
)
def test_DataTypeFileMapping_packing_dataframe_raises_sensible_error_on_object_dtype(
    obj,
):
    """
    Don't permit object dtype because this can contain mixed entries.
    Strings should be the string type
    """
    _, _, packer, _ = DataTypePacking.get_from_object(obj)
    with pytest.raises(NotImplementedError):
        packer(obj)


@pytest.mark.parametrize(
    "obj, type, equality_checker", example_data_type_equality_check
)
def test_DataTypeFileMapping_packing_unpacking(obj, type, equality_checker):
    """
    Test that the packing and unpacking of the objects works correctly

    expect any pandas df or series, polars df or series, or numpy array is unchanged
    after packing (converting to polars df for saving) and unpacking (converting back to
    original type from the saved polars df).
    """
    _, _, packer, unpacker = DataTypePacking.get_from_object(obj)
    packed = packer(obj)
    assert isinstance(packed, pl.DataFrame)
    unpacked = unpacker(packed)
    assert isinstance(unpacked, type)
    equality_checker(obj, unpacked)


def test_get_metadata_return_value():
    assert (
        get_metadata(DummyEstimator(a=1, b=2), c=3)
        == """{"estimator_name": "DummyEstimator", "estimator_params": {"a": 1, "b": 2}, "_get_metadata_kwargs": {"c": 3}}"""
    )


@pytest.mark.parametrize(
    "estimator1, estimator2",
    [
        (x, y)
        for x, y in itertools.combinations(
            [
                KMeans(n_clusters=3),
                KMeans(n_clusters=6),
                RandomForestClassifier(n_estimators=100),
                RandomForestClassifier(n_estimators=200),
                LinearRegression(),
                GaussianNB(),
                SVC(),
                DecisionTreeClassifier(max_depth=5),
                DecisionTreeClassifier(max_depth=9),
                GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
                KNeighborsClassifier(n_neighbors=3),
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    max_iter=300,
                    activation="relu",
                    solver="adam",
                    random_state=1,
                ),
                LogisticRegression(max_iter=100),
                SVR(kernel="rbf", C=1.0, epsilon=0.1),
                Ridge(alpha=1.0),
                AdaBoostClassifier(n_estimators=100),
                PCA(n_components=2),
                DBSCAN(eps=0.5, min_samples=5),
                MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500),
                MLPRegressor(hidden_layer_sizes=(2, 2), max_iter=2),
            ],
            2,
        )
        if x != y
    ],
)
def test_get_metadata_returns_different_results_for_different_estimators(
    estimator1, estimator2
):
    """
    make sure that different estimators, or the same estimator with different params,
    gives different metadata.
    """
    assert get_metadata(estimator1) != get_metadata(
        estimator2
    ), "metadata for different estimators should differ"


@pytest.mark.parametrize(
    "obj1, obj2",
    [
        (obj1, obj2)
        for objs in [
            [
                pd.DataFrame({"0": [1, 2]}),
                pd.Series([1, 2], index=[0, 1]),
                np.array([1, 2]),
                pl.DataFrame({"0": [1, 2]}),
                pl.Series([1, 2]),
            ],
            [
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
            ],
            [
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[2, 3]),
                pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[3, 4]),
            ],
            [
                pd.Series([1, 2, 3], index=[8, 9, 10]),
                pd.Series([1, 2, 3], index=[1, 9, 10]),
            ],
            [
                pd.Series([1, 2, 3], index=[8, 9, 10], name="some name"),
                pd.Series([1, 2, 3], index=[8, 9, 10], name="some other name"),
            ],
        ]
        for obj1, obj2 in itertools.product(objs, objs)
        if obj1 is not obj2
    ],
)
def test_get_metadata_changes_for_different_represenations_of_the_same_data(obj1, obj2):
    """
    check that the metadata changes for different representations of the same data
    """
    estimator = KMeans(n_clusters=3)
    assert get_metadata(estimator, data=obj1) != get_metadata(
        estimator, data=obj2
    ), "metadata for data of different types should differ"


def test_validate_func_name():
    validate_func_name("transform", allowed_funcs=("transform", "predict"))
    validate_func_name("predict", allowed_funcs=("transform", "predict"))
    validate_func_name("fit", allowed_funcs=("fit", "transform", "predict"))

    with pytest.raises(ValueError):
        validate_func_name("fit", allowed_funcs=("transform", "predict"))

    with pytest.raises(ValueError):
        validate_func_name("something_else")


def _assert_return_path_format(path, estimator_func):
    experiment_and_function_dir = f"/some_experiment_save_path/{estimator_func}/"
    assert str(path).startswith(
        experiment_and_function_dir
    ), "should begin with experiment save path and estimator func as dirs"
    assert (
        path.stem not in experiment_and_function_dir
    ), "should have a unique name for the directory based on metadata or kwargs"


@pytest.mark.parametrize("estimator_func", ["fit", "transform", "predict"])
def test_get_object_save_path_return_value(estimator_func):
    save_path = _get_object_save_path(
        experiment_save_path=Path("/some_experiment_save_path"),
        estimator_func=estimator_func,
        metadata="some_metadata",
    )
    _assert_return_path_format(save_path, estimator_func)

    save_path2 = _get_object_save_path(
        experiment_save_path=Path("/some_experiment_save_path"),
        estimator_func=estimator_func,
        metadata="some_different_metadata",
    )
    _assert_return_path_format(save_path2, estimator_func)
    assert (
        save_path != save_path2
    ), "different metadata so it should give a different path"


def test_get_object_save_path_raises_error_on_bad_func():
    with pytest.raises(ValueError):
        _get_object_save_path(
            experiment_save_path=Path("/some_experiment_save_path"),
            estimator_func="bad_func",
            metadata="some_metadata",
        )


def test_saving_and_loading_estimator(temp_directory):
    experiment_save_path = temp_directory
    input_estimator = KMeans(n_clusters=3)
    input_estimator.cluster_centers_ = "some_data_from_fit"
    save_fitted_estimator_to_file(
        estimator=input_estimator,
        experiment_save_path=experiment_save_path,
        unfitted_estimator_metadata="unique_metadata",
    )
    fit_save_dir = experiment_save_path / "fit"
    assert len(list(fit_save_dir.iterdir())) == 1
    fit_save_dir_entries = list(fit_save_dir.iterdir())
    assert len(fit_save_dir_entries) == 1
    metadata_labeled_dir = fit_save_dir_entries[0]
    metadata_dir_entries = list(metadata_labeled_dir.iterdir())
    assert len(metadata_dir_entries) == 2

    saved_metadata_file = metadata_labeled_dir / "metadata.json"
    assert saved_metadata_file.exists()
    assert saved_metadata_file.read_text() == "unique_metadata"

    loaded_estimator = load_fitted_estimator_from_file(
        experiment_save_path=experiment_save_path,
        unfitted_estimator_metadata="unique_metadata",
    )
    assert isinstance(loaded_estimator, type(input_estimator))
    assert loaded_estimator.get_params(deep=True) == input_estimator.get_params(
        deep=True
    )
    assert loaded_estimator.cluster_centers_ == "some_data_from_fit"


def _saving_and_loading_data_general_checking(
    temp_directory: Path,
    process_data_func: str,
    input_data: pd.DataFrame | pd.Series,
    equality_checker: Callable,
    expected_file_count: int,
):
    experiment_save_path = temp_directory
    save_processed_data_to_file(
        experiment_save_path,
        input_data,
        process_data_func_name=process_data_func,
        fitted_estimator_process_metadata="unique_metadata",
    )
    process_data_save_dir = experiment_save_path / process_data_func
    print(f"\n\n{list(experiment_save_path.iterdir())=}\n{process_data_func=}\n")
    process_data_save_entries = list(process_data_save_dir.iterdir())
    assert len(process_data_save_entries) == 1
    metadata_labeled_dir = process_data_save_entries[0]
    metadata_dir_entries = list(metadata_labeled_dir.iterdir())
    assert len(metadata_dir_entries) == expected_file_count, (
        f"expect {expected_file_count} files in the metadata labeled dir. We have"
        f" {metadata_dir_entries}"
    )

    loaded_data = load_processed_data_from_file(
        experiment_save_path=experiment_save_path,
        process_data_func_name=process_data_func,
        fitted_estimator_process_metadata="unique_metadata",
    )
    assert isinstance(loaded_data, type(input_data))
    equality_checker(input_data, loaded_data)

    saved_metadata_file = metadata_labeled_dir / "metadata.json"
    assert saved_metadata_file.exists()
    assert saved_metadata_file.read_text() == "unique_metadata"

    return metadata_labeled_dir, loaded_data


@pytest.mark.parametrize(
    "process_data_func, input_data, equality_checker",
    [
        *[
            (process_func, data, checker)
            for process_func in ["transform", "predict"]
            for data, _, checker in example_data_type_equality_check
            if not isinstance(data, (pd.DataFrame, pd.Series))
        ]
    ],
)
def test_saving_and_loading_data_non_pandas(
    temp_directory,
    process_data_func,
    input_data,
    equality_checker,
):
    _saving_and_loading_data_general_checking(
        temp_directory,
        process_data_func,
        input_data,
        equality_checker,
        expected_file_count=2,
    )


@pytest.mark.parametrize(
    "process_data_func, input_data, equality_checker",
    [
        *[
            (process_func, data, checker)
            for process_func in ["transform", "predict"]
            for data, _, checker in example_data_type_equality_check
            if isinstance(data, (pd.DataFrame, pd.Series))
        ]
    ],
)
def test_saving_and_loading_data_pandas(
    temp_directory,
    process_data_func,
    input_data,
    equality_checker,
):
    input_data.attrs = {"unique": "metadata"}
    metadata_labeled_dir, loaded_data = _saving_and_loading_data_general_checking(
        temp_directory,
        process_data_func,
        input_data,
        equality_checker,
        expected_file_count=3,
    )
    assert loaded_data.attrs == input_data.attrs

    saved_attrs_file = metadata_labeled_dir / "data_attrs.json"
    assert saved_attrs_file.exists()
    assert saved_attrs_file.read_text() == '{"unique": "metadata"}'
