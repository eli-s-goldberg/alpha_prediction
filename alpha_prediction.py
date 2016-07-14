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
DATABASE_PATH = os.path.join('alpha_database.csv')

GBR_INITIAL_PARAMS_ = {'learning_rate': 0.1,
                       'max_depth': 4,
                       'min_samples_leaf': 5,
                       'max_features': 'sqrt',
                       'loss': 'ls',
                       'n_estimators': 1000}

GBR_PARAMETER_GRID_ = {'learning_rate': [0.1, 0.01],
                       'max_depth': [4, 5, 9, None],
                       'min_samples_leaf': [2, 20],
                       'max_features': ['auto', 'sqrt', 'log2'],
                       'loss': ['ls', 'lad'],
                       'n_estimators': [100]}
# redefine the GBC so we can do RFECV


class GradientBoostingRegressorWithCoef(GradientBoostingRegressor):

    def fit(self, *args, **kwargs):
        super(GradientBoostingRegressorWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


def one_hot_dataframe(data, cols, replace=False):
    """Do hot encoding of categorical columns in a pandas DataFrame.
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


ONE_HOT_REFORM_CATEGORIES_ = [
    'nmId', 'shape', 'nomLayer', 'dissNomType', 'saltType', 'prepMethod']


def main(
        iterations=1,
        output_dir='output',
        one_hot_reform=True,
        rfecv_eval=True,
        deterministic=False,
        grid_search_eval=True,
        shuffle_holdout=True,
        plot_rfecv_gridscore=True,
        holdout_size=0.15,
        crossfolds=5,
        one_hot_reform_categories=ONE_HOT_REFORM_CATEGORIES_,
        database_path=DATABASE_PATH,
        target_data_column_name='depAttEff',
        gbr_parameter_grid_=GBR_PARAMETER_GRID_,
        gbr_initial_params=GBR_INITIAL_PARAMS_):

    # input database
    database_basename = os.path.basename(DATABASE_PATH)

    # output directory
    output_dir = os.path.join(output_dir, 'classifier')
    make_dirs(output_dir)

    training_data = pd.read_csv(database_path)
    target_data = training_data[target_data_column_name]

    # make sure to drop the target data from the training data
    training_data = training_data.drop([target_data_column_name], 1)

    # initialize the regressor with initial params
    clf = GradientBoostingRegressorWithCoef(**gbr_initial_params)

    if one_hot_reform:
        training_data, _, _ = one_hot_dataframe(
            training_data, one_hot_reform_categories, replace=True)

    for run in xrange(iterations):
        print run
        y_all = np.array(target_data)
        x_all = training_data.as_matrix()

        if shuffle_holdout:
            random_state = _SEED if deterministic else None
            sss = cross_validation.ShuffleSplit(len(y_all),
                                                n_iter=1,
                                                test_size=holdout_size,
                                                random_state=random_state)

            for train_index, test_index in sss:
                x_train, x_holdout = x_all[train_index], x_all[test_index]
                y_train, y_holdout = y_all[train_index], y_all[test_index]

        '''The logic is to optimize the parameters for all the features before
		RFECV'''
        if grid_search_eval:
            grid_searcher = grid_search.GridSearchCV(estimator=clf,
                                                     cv=crossfolds,
                                                     param_grid=gbr_parameter_grid_,
                                                     n_jobs=-1)

            # call the grid search fit using the data
            grid_searcher.fit(x_train, y_train)

            # store and print the best parameters
            best_params = grid_searcher.best_params_

        else:
            ''' The logic is that if we don't do grid search, use initial
                    params as 'best' '''
            best_params = gbr_initial_params

        # re-initialize and fit with the "best params"
        clf = GradientBoostingRegressorWithCoef(**best_params)
        clf.fit(x_train, y_train)

        if rfecv_eval:
            rfecv = RFECV(
                estimator=clf,
                step=1,
                cv=crossfolds,
                scoring='mean_absolute_error')

            rfecv.fit(x_train, y_train)
            rfecv_y_predicted = rfecv.predict(x_holdout)
            print metrics.mean_absolute_error(rfecv_y_predicted, y_holdout)
            print metrics.r2_score(rfecv_y_predicted, y_holdout)

            plt.plot(rfecv_y_predicted, y_holdout, '+')
            plt.plot(y_holdout, y_holdout, 'r-')
            plt.show()

            if plot_rfecv_gridscore and rfecv_eval:
                plt.xlabel("Number of features selected")
                plt.ylabel("Cross validation score (r2)")
                plt.title(str('CV: ' + regName + '; dataset: ' + name[:-4]))
                plt.plot(range(1, len(rfecv.grid_scores_) + 1),
                         rfecv.grid_scores_)
                plt.show()


main(grid_search_eval=True)
