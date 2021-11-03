#!/usr/bin/env python
from os.path import dirname, abspath
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

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


    def mpaa_graph(df):
        # mpaa_df = df.filter(items=['opening_revenue', 'mpaa_g', 'mpaa_pg', 'mpaa_pg-13', 'mpaa_r', 'mpaa_nc-17', 'mpaa_unrated'])
        x = df['mpaa']
        y = df['opening_revenue']
        plt.scatter(x, y)

        # colors = ['red', 'blue', 'green', 'purple', 'yellow', 'orange']
        # for col, color in zip(mpaa_cols, colors):
        #     plt.scatter(df[col], y, color=color)
        # print(mpaa_df)
        plt.show()


    path = dirname(dirname(abspath(__file__))) + '/extract_data/pickled_data/'
    with open(path + 'features.pickle', 'rb') as f:
        df = pickle.load(f)
    with open(path + 'movie_data.pickle','rb') as f:
        df_raw = pickle.load(f)


    # mpaa_graph(df_raw[['opening_revenue', 'mpaa']])

    X = df.drop(columns=['tconst', 'domestic_revenue', 'opening_revenue'])
    X_no_mpaa = X.drop(columns=['mpaa_g','mpaa_pg','mpaa_pg-13','mpaa_r','mpaa_nc-17','mpaa_unrated'])
    # eigen(X)
    y_domestic = df['domestic_revenue']
    y_opening = df['opening_revenue']
    pca = PCA()
    pca.fit(X)


    def test_model(model, X, y):
        scores = cross_validate(model, X, y, scoring='r2', cv=5)
        print(f"Base Linear Regression Opening R2: {scores['test_score']}")


    # Base test
    regr = linear_model.LinearRegression()
    test_model(regr, X, y_opening)
    test_model(regr, X_no_mpaa, y_opening)
    kernel = 'rbf'
    c = 1.0
    eps = .2
    svr = SVR(kernel=kernel, C=c, epsilon=eps)
    test_model(svr, X, y_opening)
    test_model(svr, X_no_mpaa, y_opening)
    # ridge = linear_model.Ridge()
    # test_model(ridge, X, y_opening)
    # lasso = linear_model.Lasso()
    # test_model(lasso, X, y_opening)
