"""
Utility functions for everything related to data processing and caching for
the sklearn caching module.

Includes:
- DataTypePacking: A class to pack and unpack dataframes, series and numpy
arrays to and from polars. Needed in order to have a unified approach for
converting data to string, and saving data to file.
- get_metadata: Get a metadata dictionary for this wrapped estimator,
including any data about the estimator itself (the class, parameters etc.) and
including any extra information like the args/kwargs passed to the fit
function. Uses _EstimatorCachingWrapperMetadataEncoder so that we can
efficiently json dump huge data objects of different types.
- validate_func_name: Validate that a given function name is one of the given
allowed functions.
- save_fitted_estimator_to_file: Save a fitted estimator to disk
- load_fitted_estimator_from_file: load a fitted estimator from disk
- save_processed_data_to_file: Save data to disk for a given inference method's
(predict or transform) output
- load_processed_data_from_file: Load data from disk for a given inference
method's (predict or transform) output
"""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
import numpy as np
import pandas as pd
import polars as pl
import structlog
from pandas.core.indexes.frozen import FrozenList
from sklearn.base import BaseEstimator

from .config import BLOCK_SIZE, DataType

logger = structlog.get_logger()


class DataTypePacking:
    """
    A class to pack and unpack dataframes, series and numpy arrays to and from polars
    Needed in order to have a unified approach for converting data to string, and saving
    data to file. Otherwise we would need to implement it repeatedly for each data type.

    To add support for a new data type you just need to add entries here and in
    config.DataType.
    The packer and unpacker must be capable of storing common variations of the data (
    for example data frame column name types, index name and values, and type, series
    name etc.) so that after packing and unpacking, you have recovered the input data.

    Note: that pandas metadata attrs object is not handled here, but is handled
    separately when saving/loading the data and when json dumping the object.

    Attributes
    ----------
    mapping: List[
        tuple[
            type,
            str,
            Callable[[DataType], pl.DataFrame],
            Callable[[pl.DataFrame], DataType],
    ]
        A list of tuples, each containing:
        - Data type. must be compatible with the entries in DataType
        - Name. A label describing the data type
        - packer: A function that converts the data type to a polars dataframe
        - unpacker: A function that converts a polars dataframe to the data type
    types: tuple[type, ...]
        A tuple of the data types that are supported.
        Equivalent to [t for t, *_ in mapping]
    names: tuple[str, ...]
        A tuple of the names of the data types that are supported.
        Equivalent to [n for _, n, *_ in mapping]
    packers: tuple[Callable[[DataType], pl.DataFrame], ...]
        A tuple of the packer functions for the data types that are supported.
        Equivalent to [p for _, _, p, *_ in mapping]
    unpackers: tuple[Callable[[pl.DataFrame], DataType], ...]
        A tuple of the unpacker functions for the data types that are supported.
        Equivalent to [u for _, _, _, u in mapping]
    name_unpacking_mapping: dict[str, Callable[[pl.DataFrame], DataType]]
        A dictionary mapping the names of the data types to their unpacker functions.
    """

    mapping = [  # (type, name, packer, unpacker)
        (
            pd.DataFrame,
            "data_pandas_df",
            lambda pd_df: DataTypePacking._pandas_dataframe_packer(pd_df.copy()),
            lambda pl_df: DataTypePacking._pandas_dataframe_unpacker(pl_df),
        ),
        (
            pd.Series,
            "data_pandas_series",
            lambda pd_series: DataTypePacking._pandas_dataframe_packer(
                pd_series.copy()
                .to_frame()
                .rename(columns={0: None} if pd_series.name is None else {})
                # That renaming is needed because to_frame renames None to 0
            ),
            lambda pl_df: DataTypePacking._pandas_dataframe_unpacker(pl_df).iloc[:, 0],
        ),
        (
            pl.DataFrame,
            "data_polars_df",
            lambda pl_df: pl_df,
            lambda pl_df: pl_df,
        ),
        (
            pl.Series,
            "data_polars_series",
            lambda pl_series: pl_series.to_frame(),
            lambda pl_df: pl_df[pl_df.columns[0]],
        ),
        (
            np.ndarray,
            "data_numpy_array",
            lambda np_array: DataTypePacking._numpy_array_packer(np_array.copy()),
            lambda pl_df: DataTypePacking._numpy_array_unpacker(pl_df),
        ),
    ]
    types, names, packers, unpackers = zip(*mapping)
    name_unpacking_mapping = dict(zip(names, unpackers))

    @staticmethod
    def _pandas_dataframe_packer(pd_df: pd.DataFrame) -> pl.DataFrame:
        """
        Convert a pandas dataframe to a polars dataframe, in a way that stores the
        information about the original data, column names, and index and index names.

        Should not lose any information, such that calling _pandas_dataframe_unpacker
        should recover the original pandas dataframe.

        Dtypes must not be of type object, because these can include mixed types. Cast
        them to string type first if all entries in the columnn are strings.

        Note that the attrs property is not handled here. If it exists it is ignored.

        Parameters
        ----------
        pd_df: pd.DataFrame
            The pandas dataframe to convert to a polars dataframe

        Returns
        -------
        pl.DataFrame
            The polars dataframe representation of the pandas dataframe
            It will not necessarily look the same as the input pandas dataframe (will
            have different col names etc.)
        """
        if not (object_cols_df := pd_df.select_dtypes(include=["object"])).empty:
            raise NotImplementedError(
                f"object dtype columns are not supported in the input pandas dataframe "
                f"because these can include mixed types. Cast them to string type "
                f"first (.astype('string') not .astype(str)). Have the following "
                f"'object' dtype columns: {object_cols_df.columns}"
            )

        column_mapping = {}
        for pandas_col in pd_df.columns:
            if isinstance(pandas_col, str):
                polars_col = pandas_col
            elif pandas_col is None:
                polars_col = "__None_no_col_name"
            elif isinstance(pandas_col, (float, int)):
                polars_col = f"__dtype_{type(pandas_col).__name__}_col_{pandas_col}"
            else:
                raise ValueError(f"Column name {pandas_col} is not a valid type")
            column_mapping[pandas_col] = polars_col

        pd_df = pd_df.rename(columns=column_mapping)

        # pandas implements index names as FrozenList
        pd_df.index.names = FrozenList(
            [
                "__None_no_index" if i_name is None else i_name
                for i_name in pd_df.index.names
            ]
        )
        return pl.from_pandas(
            pd_df.reset_index().rename(
                columns={i: f"_index__{i}" for i in pd_df.index.names}
            )
        )

    @staticmethod
    def _pandas_dataframe_unpacker(pl_df: pl.DataFrame) -> pd.DataFrame:
        """
        Convert a polars dataframe back to a pandas dataframe, in a way that restores
        the original data, column names, and index and index names. The input should
        be the output of _pandas_dataframe_packer.

        Parameters
        ----------
        pl_df: pl.DataFrame
            The polars dataframe to convert to a pandas dataframe, should be the output
            of _pandas_dataframe_packer

        Returns
        -------
        pd.DataFrame
            The recovered pandas dataframe
        """
        df = pl_df.to_pandas()
        df_columns: list[str] = list(df.columns)
        index_names = [
            col.removeprefix("_index__") for col in df_columns if "_index__" in col
        ]
        df = df.rename(columns={f"_index__{i}": i for i in index_names})
        df = df.set_index(index_names)
        df.index.names = FrozenList(
            [None if i == "__None_no_index" else i for i in df.index.names]
        )

        column_mapping = {}
        for polars_col in df.columns:
            if polars_col == "__None_no_col_name":
                pandas_col = None
            elif polars_col.startswith("__dtype_"):
                col_str = polars_col.split("_col_")[-1]
                dtype = polars_col.split("_col_")[0].split("__dtype_")[-1]
                if dtype not in ["int", "float"]:
                    raise ValueError(
                        f"Only str, None, int and float column types allowed, got "
                        f"{dtype}"
                    )
                pandas_col = int(col_str) if dtype == "int" else float(col_str)
            else:
                # is string
                pandas_col = polars_col
            column_mapping[polars_col] = pandas_col

        str_cols_df = df.select_dtypes(include=["object"])
        df[str_cols_df.columns] = str_cols_df.astype("string")
        return df.rename(columns=column_mapping)

    @staticmethod
    def _numpy_array_packer(np_array: np.ndarray) -> pl.DataFrame:
        """
        Convert a numpy array to a polars dataframe, in a way that stores the
        information original shape of the array.

        Should not lose any information, such that calling _numpy_array_unpacker should
        recover the original numpy array.

        Parameters
        ----------
        np_array: np.ndarray
            The numpy array to convert to a polars dataframe

        Returns
        -------
        pl.DataFrame
            The polars dataframe representation of the numpy array
        """
        original_shape = np_array.shape
        array_2d = np_array.reshape(-1, np.prod(original_shape) // original_shape[0])
        pl_df = pl.from_numpy(array_2d)
        pl_df.columns = [
            f"data_{i}_original_shape_{original_shape}"
            for i in range(len(pl_df.columns))
        ]
        return pl_df

    @staticmethod
    def _numpy_array_unpacker(pl_df: pl.DataFrame) -> np.ndarray:
        """
        Convert a polars dataframe to a numpy array, in a way that restores the
        original data and shape of the array. The input should be the output of
        _numpy_array_packer.

        Parameters
        ----------
        pl_df: pl.DataFrame
            The polars dataframe to convert to a numpy array, should be the output of
            _numpy_array_packer

        Returns
        -------
        np.ndarray
            The recovered numpy array.
        """
        array_2d = pl_df.to_numpy()
        original_shape_text = pl_df.columns[0].split("_original_shape_")[-1]
        original_shape = tuple(
            int(dim_size)
            for dim_size in original_shape_text.strip("() ").split(",")
            if dim_size
        )
        return array_2d.reshape(original_shape)

    @classmethod
    def get_from_object(cls, obj: DataType) -> tuple[
        type,
        str,
        Callable[[DataType], pl.DataFrame],
        Callable[[pl.DataFrame], DataType],
    ]:
        """
        Get the full mapping entry for a given object.

        Parameters
        ----------
        obj: DataType
            The object for which to get the mapping entry

        Raises
        ------
        TypeError
            If the object is not of a valid class

        Returns
        -------
        tuple[
            type,
            str,
            Callable[[DataType], pl.DataFrame],
            Callable[[pl.DataFrame], DataType],
        ]
            The mapping entry for the object
        """
        for type_, name, packer, unpacker in cls.mapping:
            if isinstance(obj, type_):
                return type_, name, packer, unpacker
        raise TypeError(f"{obj} is not a of a valid class: {cls.types}")


class _EstimatorCachingWrapperMetadataEncoder(json.JSONEncoder):
    """
    A json encoder that can efficiently json dump data for pandas dataframes, series,
    numpy arrays and sklearn estimators.
    """

    @staticmethod
    def polars_dataframe_to_hashes_string(pl_df: pl.DataFrame) -> str:
        """
        Convert a polars object to a unique string. Useful when we care about checking
        if two polars dataframes are the same and don't need it to be readable

        This is needed because the inbuilt .to_string() method is way too slow for large
        dataframes (because it spends a lot of time making it readable)

        This assumes that the order of the columns matters. This is because inside the
        estimators, the dataframe column order might matter.
        Similaryly, it assumes that the data type of the dataframe matters. If the data
        is the same but with different data types, then the dataframe is different.

        Parameters
        ----------
        pl_df: pd.DataFrame
            A dataframe to convert to a string

        Returns
        -------
        str
            The string representation of the polars dataframe
        """
        pl_df = pl_df.rename(
            {col: f"{col}_{str(pl_df[col].dtype)}" for col in pl_df.columns}
        )

        hashes = ""
        for i in range(0, len(pl_df), BLOCK_SIZE):
            pl_df_block = pl_df[i : i + BLOCK_SIZE]
            buffer = BytesIO()
            pl_df_block.write_csv(buffer)  # this writes it to string
            buffer.seek(0)
            data_bytes = buffer.read()
            hash_object = hashlib.blake2b(data_bytes)
            hashes += f"block_{i}_{hash_object.hexdigest()}__"
        return hashes[:-2]  # don't need the final "__"

    def default(self, obj: Any) -> str:
        """
        Convert an object to a json string. If the object is a polars or pandas
        dataframe/series or numpy array, convert it to a string first. If the object is
        an estimator, get the metadata for it.

        Parameters
        ----------
        obj: Any
            The object to convert to a json string
            Can be anything that is json dumpable by default, or a pandas/polars
            dataframe/series, numpy array or sklearn estimator.

        Returns
        -------
        str
            The json string representation of the object
        """
        if isinstance(obj, DataTypePacking.types):
            # ensure is dataframe first so we can use the pandas_to_string function
            _, _, packer, *_ = DataTypePacking.get_from_object(obj)
            pl_df = packer(obj)
            dict_data: dict[str, Any] = {
                type(obj).__name__: self.polars_dataframe_to_hashes_string(pl_df=pl_df)
            }

            if isinstance(obj, (pd.DataFrame, pd.Series)):
                dict_data["attrs"] = getattr(obj, "attrs", {})
            return str(dict_data)

        try:
            return json.JSONEncoder.default(self, obj)  # check is json dumpable
        except TypeError as e:
            if not hasattr(obj, "get_params"):
                raise e  # if it's not an estimator
        return str({type(obj).__name__: obj.get_params(deep=True)})


def get_metadata(estimator, **kwargs) -> str:
    """
    Get a metadata dictionary for this wrapped estimator, including any data about
    the estimator itself (the class, parameters etc.) and including any extra
    information passed in as kwargs.

    Warning: Depending on the estimator used, the returned string might vary across
    runtimes. For example if you use a custom estimator, which has properties that
    don't have a well-defined __repr__, then the returned string might have the
    memory address of the instance, which will vary across runs. Make sure the
    estimators (and their components) are well-defined.

    Parameters
    ----------
    estimator: BaseEstimator
        The estimator to get metadata for
    kwargs:
        kwargs here will include data defining the run, for example whether it was a
        fit/predict/transform/fit_predict/fit_transform call and what data and
        kwargs were passed to it.

    Returns
    -------
    str
        The metadata dict as a string
    """
    logger.debug(
        f"Getting metadata for the estimator with the following kwarg names: "
        f"{kwargs.keys()}"
    )
    # Sort because we don't want dict order to have an impact
    return json.dumps(
        {
            "estimator_name": type(estimator).__name__,
            "estimator_params": dict(sorted(estimator.get_params(deep=True).items())),
            "_get_metadata_kwargs": dict(sorted(kwargs.items())),
        },
        cls=_EstimatorCachingWrapperMetadataEncoder,
    )


def validate_func_name(
    estimator_func_name: str, allowed_funcs: tuple[str, ...] = ("predict", "transform")
):
    """
    Validate that a given function name is one of the given allowed functions.

    Parameters
    ----------
    estimator_func_name: str
        The function name to validate
    allowed_funcs: tuple[str]
        The allowed function names

    Raises
    ------
    ValueError
        If the function name is not one of the allowed functions
    """
    logger.debug(
        f"Validating that the function name {estimator_func_name} is one of "
        f"{allowed_funcs}"
    )
    if estimator_func_name not in allowed_funcs:
        msg = (
            f"estimator_func must be one of {allowed_funcs}, have {estimator_func_name}"
        )
        raise ValueError(msg)


def _get_object_save_path(
    experiment_save_path: Path,
    estimator_func: Literal["fit", "predict", "transform"],
    metadata: str,
) -> Path:
    """
    Get the path to save the data/object to for a given function call.

    This will be unique for each of fit/predict/transform call (including the
    args/kwargs it was called with)

    Note that fit_predict/fit_transform will call this once for the fit part and
    once for the predict/transform part, meaning that a call to fit_predict will
    still point to same path as the call to fit for the estimator, and the same path
    as the call to predict for the data (and similarly for transform).

    Parameters
    ----------
    experiment_save_path: Path
        The path to the experiment save directory
    estimator_func: Literal["fit", "predict", "transform"]
        The function that was called i.e. fit/predict/transform
    metadata: str
        The metadata dictionary as a string for this function call (it should have
        been generated by get_metadata passing in the same args/kwargs that were
        passed to the function call)
    """
    validate_func_name(
        estimator_func_name=estimator_func,
        allowed_funcs=("fit", "predict", "transform"),
    )

    byte_data = metadata.encode()
    hash_object = hashlib.blake2b(byte_data)
    hashed_name = hash_object.hexdigest()

    p = experiment_save_path / estimator_func / hashed_name
    logger.debug(
        f"Got save path {p=}, {estimator_func=}",
    )
    return p


def save_fitted_estimator_to_file(
    estimator: BaseEstimator,
    experiment_save_path: Path,
    unfitted_estimator_metadata: str,
):
    """
    Save the estimator object to disk for a given fit function call (with unique
    inputs and class/instance setup).

    Only use this within the fit methods, it will save the sklearn estimator and
    metadata to disk

    Parameters
    ----------
    estimator: BaseEstimator
        The estimator to save
    experiment_save_path: Path
        The path to the experiment save directory
    unfitted_estimator_metadata: str
        The metadata dictionary as a string for this function call (it should have
        been generated by get_metadata passing in the same args/kwargs that were
        passed to the function call).
        This should have been generated using the unfitted estimator, because in
        load_fitted_estimator_from_file we'll want to load a fitted estimator from it's
        unfitted version (assuming the input data lines up).
    """
    save_path = _get_object_save_path(
        experiment_save_path=experiment_save_path,
        estimator_func="fit",
        metadata=unfitted_estimator_metadata,
    )
    save_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving the estimator and metadata to file at {save_path}")
    (save_path / "metadata.json").write_text(unfitted_estimator_metadata)
    joblib.dump(estimator, save_path / "object.joblib")


def load_fitted_estimator_from_file(
    experiment_save_path: Path,
    unfitted_estimator_metadata: str,
) -> BaseEstimator:
    """
    For a given fit function call (with unique inputs and class/instance setup) try
    and load existing estimator object from a previously cached run.

    Only use this within the fit method, it will load the sklearn estimator from
    disk.

    Parameters
    ----------
    experiment_save_path: Path
        The path to the experiment save directory
    unfitted_estimator_metadata: str
        The metadata dictionary as a string for this function call (it should have
        been generated by get_metadata passing in the same args/kwargs that were
        passed to the function call).
        This should have been generated using the unfitted estimator, because in
        we want to load a fitted estimator from it's unfitted version (assuming the
        input data lines up). i.e. it should be the same metadata as was used in
        save_fitted_estimator_to_file.

    Returns
    -------
    BaseEstimator
        The estimator that was cached to disk for this function call.
    """
    # metadata should have been generated by get_metadata using the unfitted estimator
    save_path = _get_object_save_path(
        experiment_save_path=experiment_save_path,
        estimator_func="fit",
        metadata=unfitted_estimator_metadata,
    )
    logger.info(f"Loading the estimator from file {save_path}")
    object_ = joblib.load(save_path / "object.joblib")
    return object_


def save_processed_data_to_file(
    experiment_save_path: Path,
    save_data: DataType,
    process_data_func_name: Literal["predict", "transform"],
    fitted_estimator_process_metadata,
):
    """
    Save data to disk for a given function call (predict or transform and unique
    inputs and class/instance setup).

    This will be cached as a parquet file regardless of the data type.

    Parameters
    ----------
    experiment_save_path: Path
        The path to the experiment save directory
    save_data: DataType
        The data to save. Will always convert to a df before saving
    process_data_func_name: Literal["predict", "transform"]
        The function that was called i.e. predict/transform
    fitted_estimator_process_metadata: str
        The metadata dictionary as a string for this function call (it should have
        been generated by get_metadata passing in the same args/kwargs that were
        passed to the function call).
        It should have been generated from the fitted estimator. It is the fitted model
        that processes the data. (think for example if a model was re-fit, we wouldn't
        want to save/load old data).
        This should be the same metadata as was used in load_process_data_from_file
    """
    validate_func_name(
        estimator_func_name=process_data_func_name,
        allowed_funcs=("predict", "transform"),
    )

    save_path = _get_object_save_path(
        experiment_save_path=experiment_save_path,
        estimator_func=process_data_func_name,
        metadata=fitted_estimator_process_metadata,
    )

    save_path.mkdir(exist_ok=True, parents=True)
    logger.info(
        f"Saving the data to file at {save_path}, {process_data_func_name=}",
    )
    (save_path / "metadata.json").write_text(fitted_estimator_process_metadata)

    _, name, packer, *_ = DataTypePacking.get_from_object(save_data)
    packer(save_data).write_parquet(save_path / f"{name}.parquet")
    # todo compare compression and compression-level for speed vs file size

    if isinstance(save_data, (pd.DataFrame, pd.Series)):
        (save_path / "data_attrs.json").write_text(
            json.dumps(getattr(save_data, "attrs", {}))
        )


def load_processed_data_from_file(
    experiment_save_path: Path,
    process_data_func_name: Literal["predict", "transform"],
    fitted_estimator_process_metadata: str,
) -> DataType:
    """
    For a given function call (predict or transform and unique inputs and
    class/instance setup) try and load existing output data from a previously cached
    run.

    Parameters
    ----------
    experiment_save_path: Path
        The path to the experiment save directory
    process_data_func_name: Literal["predict", "transform"]
        The data outputting function that was called i.e. predict/transform
    fitted_estimator_process_metadata: str
        The metadata dictionary as a string for this function call (it should have
        been generated by get_metadata passing in the same args/kwargs that were
        passed to the function call).
        It should have been generated from the fitted estimator. It is the fitted model
        that processes the data. (think for example if a model was re-fit, we wouldn't
        want to save/load old data).
        This should be the same metadata as was used in save_process_data_to_file

    Returns
    -------
    DataType
        The data that was cached to disk for this function call.
        Will convert back to the required data type.
    """
    validate_func_name(
        estimator_func_name=process_data_func_name,
        allowed_funcs=("predict", "transform"),
    )

    save_path = _get_object_save_path(
        experiment_save_path=experiment_save_path,
        estimator_func=process_data_func_name,
        metadata=fitted_estimator_process_metadata,
    )

    logger.info(f"Loading the data from file {save_path}")

    files = [p for p in save_path.iterdir() if p.stem in DataTypePacking.names]
    if not files:
        raise FileNotFoundError(f"No cached data found in {save_path}")
    assert len(files) == 1, f"Multiple files found in {save_path}"

    file = files.pop()
    data = DataTypePacking.name_unpacking_mapping[file.stem](pl.read_parquet(file))
    logger.info(
        f"Successfully loaded data of type {type(data)}, {process_data_func_name=}",
    )

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data.attrs = json.loads((save_path / "data_attrs.json").read_text())
    return data
