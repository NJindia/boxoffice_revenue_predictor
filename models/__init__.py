#!/usr/bin/env python
import math
import pickle
from os.path import dirname, abspath

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

if __name__ == '__main__':
    SPLIT_RATIO = .6


    # @X_train IS 2D
    def eigen(X_train):
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        cov_mat = np.cov(X_train_std.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        # calculate cumulative sum of explained variances
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        # plot explained variances
        plt.bar(range(1, X_train.shape[1] + 1), var_exp, alpha=0.5,
                align='center', label='individual explained variance')
        plt.step(range(1, X_train.shape[1] + 1), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.show()


    path = dirname(dirname(abspath(__file__))) + '/extract_data/pickled_data/'
    with open(path + 'features.pickle', 'rb') as f:
        df = pickle.load(f)
    with open(path + 'movie_data.pickle', 'rb') as f:
        df_raw = pickle.load(f)

    X = df.drop(columns=['tconst', 'domestic_revenue', 'opening_revenue', 'directors'])
    split_idx = int(len(X) * SPLIT_RATIO)


    def get_train_test(X):
        X_train = X.iloc[0:split_idx]
        X_test = X.iloc[split_idx:, ]
        return X_train, X_test


    # eigen(X)

    y_domestic = df['domestic_revenue']
    y_opening = df['opening_revenue']
    y_opening = y_opening.apply(lambda row: math.log(row))
    y_train = y_opening.iloc[0:split_idx]
    y_test = y_opening.iloc[split_idx:, ]
    pca = PCA()
    pca.fit(X)


    # FROM https://github.com/PermanAtayev/Movie-revenue-prediction/blob/master/movie_revenue_predict.ipynb
    def show_metrics(y_test, y_pred):
        print("Mean Squared Log Error=" + str(metrics.mean_squared_log_error(y_test, y_pred)))
        print("Root Mean Squared Log Error=" + str(np.sqrt(metrics.mean_squared_log_error(y_test, y_pred))))
        print("Mean Squared Error =" + str(metrics.mean_squared_error(y_test, y_pred)))
        print("Root Mean Squared Error=" + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
        print("R^2=" + str(metrics.r2_score(y_test, y_pred)))


    def test_model(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        show_metrics(y_test, y_pred)


    # TODO: ONLY PREDICT ON LATEST MOVIES, POWER IS NOT GOOD FOR PREDICTING OLDER MOVIES

    # Base test
    regr = linear_model.LinearRegression()
    X_train, X_test = get_train_test(X)
    print('Lin Reg Train Baseline')
    test_model(regr, X_train, y_train, X_train, y_train)

    print('Lin Reg Baseline')
    test_model(regr, X_train, y_train, X_test, y_test)

    print('Lin Reg No MPAA')
    X_feature_eval_train, X_feature_eval_test = get_train_test(
        X.drop(columns=['mpaa_g', 'mpaa_pg', 'mpaa_pg-13', 'mpaa_r', 'mpaa_nc-17', 'mpaa_unrated']))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Genres')
    genre_cols = [col_name for col_name in X if 'genre_' in col_name]
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=genre_cols))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Distributors')
    distributor_cols = [col_name for col_name in X if 'distributor_' in col_name]
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=distributor_cols))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Opening Theatres')
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=['opening_theaters']))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Runtime')
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=['runtime_minutes']))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Release Day')
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=['release_day']))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Release Month')
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=['release_month']))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    print('Lin Reg No Release Year')
    X_feature_eval_train, X_feature_eval_test = get_train_test(X.drop(columns=['release_year']))
    test_model(regr, X_feature_eval_train, y_train, X_feature_eval_test, y_test)

    # X_sum = X.drop(columns=['director_power_p'])
    # X_power = X.drop(columns=['director_power_s'])
    # test_model(regr, X_sum, y_opening)
    # test_model(regr, X_power, y_opening)
    kernel = 'rbf'
    c = 1.0
    eps = .2
    svr = SVR(kernel=kernel, C=c, epsilon=eps)
    print('SVR')
    test_model(svr, X_train, y_train, X_test, y_test)
    # test_model(svr, X_sum, y_opening)
    # test_model(svr, X_power, y_opening)
    # ridge = linear_model.Ridge()
    # test_model(ridge, X, y_opening)
    # lasso = linear_model.Lasso()
    # test_model(lasso, X, y_opening)
