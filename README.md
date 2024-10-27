# sklearn-estimator-caching
A wrapper for sklearn estimators for automatically saving and loading estimators and data for expensive fit/predict/transform calls

# Overview

[Sklearn estimators](https://scikit-learn.org/stable/developers/develop.html) (and therefore [pipelines](https://scikit-learn.org/stable/modules/compose.html)) can sometimes take a long time to run (e.g. a slow running custom estimator). When prototyping, experimenting, and building an sklearn pipeline, this can slow you down a lot if an earlier pipeline stage is very slow and we need to run it to be able to get to a later stage of the pipeline that we are developing. It could also slow us down in the case that we have a finished pipeline but we need to run inference on it more than once (which will almost always be the case).

To solve this here is functionality to cache estimators (useful when the fit process is slow) and their outputs (useful when their transform or predict methods are slow).

To cache an estimator when it runs you simply wrap your estimator as follows:

```python
from sklearn_estimator_caching import EstimatorCachingWrapperGetter

wrapper = EstimatorCachingWrapperGetter(
    project="project_name",
    experiment="experiment_name"
    base_dir="/path/to/caching/location",
    **kwargs, # these act as labels to identify a wrapped instance.
)

wrapped_estimator = wrapper(sklearn_estimator_instance)
another_wrapped_estimator = wrapper(another_sklearn_estimator_instance) # use the same wrapper for all estimators you wish to wrap.
```
Now you use `wrapped_estimator` exactly as you would have used `sklearn_estimator_instance`, but it will cache the estimator and output data. The first run will take about as long as a normal run of `sklearn_estimator_instance` (with some slight overhead), and subsequent runs will be a lot quicker (on the order of seconds for large datasets). It doesn't matter if the slowdown comes from the fitting or the inference or both, if any of them are slow then you will see a speedup.

**It will save the output data of the fit/predict/transform calls, so you should periodically clean out the cache dir and/or only wrap those estimators that are known to be slow.**

Two things to note:
- If the input data to a `fit`/`predict`/`transform` method is changed, then the cache will be invalid and the estimator will be rerun. This is desired behaviour and ensures that we don't load cached data when actually the pipeline has changed.
- If you make changes to the code of an estimator, then the estimator cacher may recognise them as different if the class name, attribute names, or attribute values change, but not if there is a change to code logic within a method. If you update the code of an estimator, you should update one of the kwarg labels to reflect this.
For example if you instantiate `EstimatorCachingWrapperGetter` with `label="some label` and then you make a change to the estimator code inside one of its methods, if you run the wrapped estimator it may incorrectly load the old version, so to be sure you would instantiate `EstimatorCachingWrapperGetter` with `label="some new label`. You can use Python's inspect module to get a string representation of the code (which you could then hash) to pass as a kwarg to the `EstimatorCachingWrapperGetter`. This works well if the estimator code is imported from a package, but will not work if prototyping within a notebook (this is a limitation from the inspect module).


Because both the estimator instance and the input data to the method are checked, the `wrapped_estimator` should behave exactly as the `sklearn_estimator_instance`, with the same outputs given the same inputs, just with a much faster runtime on any run after the initial one.

# Content Details

- [`cacheable_estimator.py`](./cacheable_estimator.py) defines `EstimatorCachingWrapper` (this the class of `wrapped_estimator`) and the `EstimatorCacherGetter` utility class for instantiating it.
- [`cache_utils.py`](./cache_utils.py) provides data handing functions, and the `EstimatorCachingWrapperMetadataEncoder` (and functions for this to work efficiently) for converting a data (that may include large dataframes/series/numpy-arrays) to a string, efficiently.
- [`config.py`](./config.py) for some setup.


# Contact People
This code was written by [antngh](https://github.com/antngh)