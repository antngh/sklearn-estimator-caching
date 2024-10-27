from copy import copy

from sklearn.base import BaseEstimator


class DummyEstimator(BaseEstimator):
    def __sklearn_clone__(self):
        return copy(self)

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def some_method(self):
        return 137

    def some_other_method(self):
        return -10
