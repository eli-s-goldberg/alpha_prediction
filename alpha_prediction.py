import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import compress
from sklearn import metrics, grid_search, cross_validation
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor

from helper_functions import *

DATABASE_PATH = os.path.join('alpha_database.csv')

GBR_INITIAL_PARAMS_ = {'learning_rate': 0.1,
                       'max_depth': 4,
                       'min_samples_leaf': 5,
                       'max_features': 'sqrt',
                       'loss': 'ls',
                       'n_estimators': 50}

GBR_PARAMETER_GRID_ = {'learning_rate': [0.1, 0.01],
                       'max_depth': [5],
                       'min_samples_leaf': [2,15,30],
                       'max_features': ['auto','sqrt', 'log2'],
                       'loss': ['ls', 'lad','huber','quantile'],
                       'n_estimators': [500]}


ONE_HOT_REFORM_CATEGORIES_ = ['nmId', 'shape', 'nomLayer',
                              'dissNomType', 'saltType', 'prepMethod']


# redefine the GBC so we can do RFECV
class GradientBoostingRegressorWithCoef(GradientBoostingRegressor):

    def fit(self, *args, **kwargs):
        super(GradientBoostingRegressorWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


def main(
        iterations=1,
        output_dir='output',
        one_hot_reform=True,
        rfecv_eval=True,
        deterministic=False,
        grid_search_eval=True,
        shuffle_holdout=True,
        plot_rfecv_gridscore=True,
        optimum_gbr_estimate = True,
        max_gbr_iterations = 1000,
        plot_all_gridscores=True,
        holdout_size=0.20,
        crossfolds=5,
        one_hot_reform_categories=ONE_HOT_REFORM_CATEGORIES_,
        database_path=DATABASE_PATH,
        target_data_column_name='depAttEff',
        gbr_parameter_grid_=GBR_PARAMETER_GRID_,
        gbr_initial_params=GBR_INITIAL_PARAMS_):

    # input database
    database_basename = os.path.basename(DATABASE_PATH)

    # output directory
    output_dir = os.path.join(output_dir, 'regressor')
    make_dirs(output_dir)

    # initialize predicted and holdout tracking
    rfecv_y_predicted_track = []
    rfecv_y_holdout_track = []

    # initialize score tracking
    score_track_mae = []
    score_track_r2 = []
    rfecv_gridscore_track = []

    # initialize feature tracking and ranking
    feature_track = []
    feature_rank = []

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
            if optimum_gbr_estimate:
                # determine minimum number of estimators with least overfitting
                x = np.arange(max_gbr_iterations) + 1
                test_score = heldout_score(clf, x_train, y_train,max_gbr_iterations)
                test_score -= test_score[0]
                test_best_iter = x[np.argmin(test_score)]
                print test_best_iter, "optimum number of iterations"
                gbr_parameter_grid_['n_estimators'] = [test_best_iter*3]

                # then implement grid search alg.
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

            # perform rfecv fitting
            rfecv.fit(x_train, y_train)

            # track predicted y values
            rfecv_y_predicted = rfecv.predict(x_holdout)
            rfecv_y_predicted_track.append(rfecv_y_predicted)

            # track truth y_holdout values
            rfecv_y_holdout_track.append(y_holdout)

            # track grid score rankings
            rfecv_gridscore_track.append(rfecv.grid_scores_)

            # track MAE performance of estimtor to predict holdout
            score_track_mae.append(metrics.mean_absolute_error(
                rfecv_y_predicted, y_holdout))

            # track overall r2 performance to predict holdout
            score_track_r2.append(metrics.r2_score(
                rfecv_y_predicted, y_holdout))

            # create array of feature ranks (contains all featuers)
            feature_rank.append(rfecv.ranking_)
            feat_names = np.array(list(training_data), copy=True)

            # create array of only selected features
            rfecv_bool = np.array(rfecv.support_, copy=True)
            sel_feat = list(compress(feat_names, rfecv_bool))
            feature_track.append(sel_feat)

        if plot_rfecv_gridscore and rfecv_eval:
            plt.plot(rfecv_y_predicted, y_holdout, '+')
            plt.plot(y_holdout, y_holdout, 'r-')
            plt.show()

            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (MAE)")
            plt.plot(range(1, len(rfecv.grid_scores_) + 1),
                     rfecv.grid_scores_)
            plt.show()

    # Output used to plot the rank of each feature relatively. 
    feature_rank_df = pd.DataFrame(feature_rank)
    feature_rank_df.columns = feat_names 
    feature_rank_df = feature_rank_df.transpose()
    feature_rank_df.to_csv('feature_rank_df.csv')

    # Output used to plot only the best features
    feature_track = pd.DataFrame(feature_track)
    feature_track = feature_track.transpose()
    feature_track.to_csv('feature_track.csv')

    # overall r2 value for all runs

    overall_r2 = metrics.r2_score(
        np.array(rfecv_y_predicted_track).ravel(order='C'), np.array(
            rfecv_y_holdout_track).ravel(order='C'))
    print overall_r2

    # Output to plot the predicted y values 
    rfecv_y_predicted_track = pd.DataFrame(rfecv_y_predicted_track).transpose()
    rfecv_y_predicted_track.to_csv('rfecv_y_predicted_track.csv')

    # Output to plot the holdout y values (truth)
    rfecv_y_holdout_track = pd.DataFrame(rfecv_y_holdout_track).transpose()
    rfecv_y_holdout_track.to_csv('rfecv_y_holdout_track.csv')

    # rfecv_x_holdout_track = pd.DataFrame(rfecv_x_holdout_track)
    # rfecv_x_holdout_track.to_csv('rfecv_x_holdout_track.csv')

    # Output used to plot the optimum model MAE 
    score_track_mae = pd.DataFrame(score_track_mae).transpose()
    score_track_mae.to_csv('score_track_mae.csv')

    # Output used to plot the optimum model r2 
    score_track_r2 = pd.DataFrame(score_track_r2).transpose()
    score_track_r2.to_csv('score_track_r2.csv')

    # transpose dataframe for ease of viewing and plotting
    rfecv_gridscore_track = pd.DataFrame(rfecv_gridscore_track)
    rfecv_gridscore_track = rfecv_gridscore_track.transpose()
    rfecv_gridscore_track.to_csv('rfecv_gridscore_track.csv')

    if plot_all_gridscores:
        rfecv_gridscore_track.plot(kind='line')
        plt.show()


if __name__=="__main__":
    main(iterations=1, plot_rfecv_gridscore=False, plot_all_gridscores=True)

