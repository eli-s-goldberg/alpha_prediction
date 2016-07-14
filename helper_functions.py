import os
import pandas as pd
import errno
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, grid_search
from sklearn.metrics import classification_report
from sklearn import cross_validation
import sklearn.metrics
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor


def one_hot_dataframe(data, cols, replace=False):
    """
    Do hot encoding of categorical columns in a pandas DataFrame.
    See:
    http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing
    .OneHotEncoder
    http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html
    https://gist.github.com/kljensen/5452382
    Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vec_data = pd.DataFrame(vec.fit_transform(
        data[cols].apply(mkdict, axis=1)).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vec_data)

    return (data, vec_data, vec)


def make_dirs(path):
    """Recursively make directories, ignoring when they already exist.
    See: http://stackoverflow.com/a/600612/1275412
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise
