# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, Trials, space_eval


# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def fix_pipeline_transform(klass):
    class _FixedTransformer(klass):
        def fit(self, x, y=None):
            """this would allow us to fit the model based on the X input."""
            super(_FixedTransformer, self).fit(x)

        def transform(self, x, y=None):
            return super(_FixedTransformer, self).transform(x)

        def fit_transform(self, x, y=None):
            return super(_FixedTransformer, self).fit(x).transform(x)
    return _FixedTransformer


# https://github.com/hyperopt/hyperopt
class ParametersHelper:
    def __init__(self, search_space, classifier_class=tree.DecisionTreeClassifier, suggest_kwargs=None):
        self.classifier_class = classifier_class
        self.suggest_kwargs = suggest_kwargs or dict(
            n_startup_jobs=20, gamma=0.25, n_EI_candidates=24
        )
        self.search_space = search_space

        self.x = None
        self.y = None
        self.trials = None
        self.best = None

    def objective_function(self, params):
        scores = cross_val_score(
            estimator=self.classifier_class(**params),
            X=self.x,
            y=self.y,
            scoring='roc_auc',
            cv=5,
            verbose=True,
            n_jobs=-1
        )
        return -1.0 * np.mean(scores)

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.trials = Trials()
        self.best = fmin(
            self.objective_function,
            space=self.search_space,
            algo=partial(tpe.suggest, **self.suggest_kwargs),
            max_evals=60,
            trials=self.trials,
            rstate=np.random.RandomState(seed=2018)
        )

    def space_eval(self):
        return space_eval(self.search_space, self.best)
