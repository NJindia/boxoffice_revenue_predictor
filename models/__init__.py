#!/usr/bin/env python
import datetime
import math
import pickle
from collections import Counter
from os.path import dirname, abspath

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from skopt import BayesSearchCV


# DO BEFORE SCALING
class StarPower(BaseEstimator, TransformerMixin):
    def _get_average_power(self, nconsts: list, df: pd.DataFrame):
        total = 0
        n = 0
        for nconst in nconsts:
            if len(df.loc[df['director'] == nconst]) == 0:
                # Todo HANDLE MISSING DIRECTOR
                return 0
            else: total = total + df.loc[df['director'] == nconst, 'wciar_power'].to_numpy()[0]
            n = n + 1
        avg_p = total / n
        return avg_p

    def _get_director_df(self, X: pd.DataFrame):
        # Get list of director nconsts
        directors_arr_raw = np.unique(X['directors'].to_numpy(dtype=str))
        directors_lists = np.char.split(directors_arr_raw, sep=',')
        directors = []
        for director_list in directors_lists:
            for name in director_list:
                if name not in directors:
                    directors.append(name)
        director_df = pd.DataFrame(columns=['director', 'sum_power', 'wciar_power'])
        for director in directors:
            # Get all movies with director
            director_stats_df = X.loc[
                X['directors'].str.contains(director), ['directors', 'domestic_revenue', 'release_year']]

            revenue_sum = math.log(director_stats_df['domestic_revenue'].sum())

            curr_year = datetime.datetime.now().year
            wciars = [math.log(iar * math.pow(.8, curr_year - year)) for iar, year in
                      zip(director_stats_df['domestic_revenue'], director_stats_df['release_year'])]
            power = (np.array(wciars).sum())
            row = {'director': director, 'sum_power': revenue_sum, 'wciar_power': power}
            director_df = director_df.append(row, ignore_index=True)
        return director_df

    def _add_director_star_power(self, train_df: pd.DataFrame, director_df: pd.DataFrame):
        return_df = train_df.copy()
        return_df['director_power_p'] = return_df.apply(
            lambda row: self._get_average_power(row['directors'].split(sep=','), director_df), axis=1)
        return_df = return_df.drop(columns=['directors'])
        return return_df

    def __init__(self):
        self.director_df = None

    def fit(self, X, y):
        self.director_df = self._get_director_df(X)
        return self

    def transform(self, X):
        X_train_power = self._add_director_star_power(X, self.director_df).dropna()
        return X_train_power.drop(columns=['domestic_revenue'])


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None


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
        plt.bar(range(1, X_train.shape[1] + 1), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(1, X_train.shape[1] + 1), cum_var_exp, where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.show()


    path = dirname(dirname(abspath(__file__))) + '/extract_data/pickled_data/'
    with open(path + 'features.pickle', 'rb') as f:
        df = pickle.load(f)
    with open(path + 'movie_data.pickle', 'rb') as f:
        df_raw = pickle.load(f)

    data = df.drop(columns=['tconst'])
    data['budget'] = data['budget'].apply(lambda row: math.log(row))
    data['opening_revenue'] = data['opening_revenue'].apply(lambda row: math.log(row))

    X = data.drop(columns=['opening_revenue'])
    y = data['opening_revenue']

    categorical_cols = ['distributor', 'release_day', 'release_month', 'mpaa']
    numerical_cols = ['release_year', 'opening_theaters', 'runtime_minutes', 'budget']
    ct = make_column_transformer((StandardScaler(), numerical_cols), (OneHotEncoder(handle_unknown='ignore'), categorical_cols))


    # eigen(X)

    def print_metrics(stats: dict):
        # print(f"Mean Squared Log Error={stats['Mean Squared Log Error']}")
        # print(f"Root Mean Squared Log Error={stats['Root Mean Squared Log Error']}")
        print(f"Mean Squared Error={stats['Mean Squared Error']}")
        # print(f"Root Mean Squared Error={stats['Root Mean Squared Error']}")
        print(f"R^2={stats['R^2']}")


    # FROM https://github.com/PermanAtayev/Movie-revenue-prediction/blob/master/movie_revenue_predict.ipynb
    def get_metrics(y_test, y_pred) -> dict:
        # msle = metrics.mean_squared_log_error(y_test, y_pred)
        # rmsle = np.sqrt(msle)
        mse = metrics.mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)
        return {  # "Mean Squared Log Error": msle,
            # "Root Mean Squared Log Error": rmsle,
            "Mean Squared Error": mse,  # "Root Mean Squared Error": rmse,
            "R^2": r2}


    def test_model(model, X, y):
        pipeline = make_pipeline(StarPower(), ct, model)
        n = 0
        stats_sum = Counter()
        for train_indices, test_indices in TimeSeriesSplit().split(X):
            pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
            y_pred = pipeline.predict(X.iloc[test_indices])
            y_pred[y_pred < 0] = 0
            stats_sum.update(Counter(get_metrics(y.iloc[test_indices], y_pred)))
            n += 1
        stats = {k: v / n for k, v in stats_sum.items()}
        print_metrics(stats)


    # Base test
    print('Lin Reg Baseline')
    regr = linear_model.LinearRegression()
    ridge = linear_model.Ridge()
    lasso = linear_model.Lasso()

    test_model(regr, X, y)
    test_model(ridge, X, y)
    test_model(lasso, X, y)

    feature_select = False
    if feature_select:
        print('Lin Reg No MPAA')
        test_model(regr, data.drop(columns=['mpaa_g', 'mpaa_pg', 'mpaa_pg-13', 'mpaa_r', 'mpaa_nc-17', 'mpaa_unrated']))

        print('Lin Reg No Genres')
        genre_cols = [col_name for col_name in data if 'genre_' in col_name]
        test_model(regr, data.drop(columns=genre_cols))

        print('Lin Reg No Distributors')
        distributor_cols = [col_name for col_name in data if 'distributor_' in col_name]
        test_model(regr, data.drop(columns=distributor_cols))

        print('Lin Reg No Opening Theatres')
        test_model(regr, data.drop(columns=['opening_theaters']))

        print('Lin Reg No Runtime')
        test_model(regr, data.drop(columns=['runtime_minutes']))

        print('Lin Reg No Release Day')
        release_cols = [col_name for col_name in data if 'day_' in col_name]
        test_model(regr, data.drop(columns=release_cols))

        print('Lin Reg No Release Month')
        release_cols = [col_name for col_name in data if 'month_' in col_name]
        test_model(regr, data.drop(columns=release_cols))

        print('Lin Reg No Release Year')
        test_model(regr, data.drop(columns=['release_year']))

    params = {"epsilon": [0.1, 0.2, 0.5, 0.3]}
    lin_params = {"C": np.logspace(-9, 9, 19), "epsilon": [0.1, 0.2, 0.5, 0.3]}
    poly_grid = {"degree": [2, 3, 4, 5, 6], "gamma": np.logspace(-9, 9, 19), "C": np.logspace(-9, 9, 19),
                 "epsilon": [0.1, 0.2, 0.5, 0.3]}

    print('SVR')
    opt = BayesSearchCV(SVR(),
                        {'C': np.logspace(-6, 6, 13), 'gamma': np.logspace(-6, 2, 9), 'epsilon': [.1, .2, .3, .4, .5],
                         'kernel': ['rbf', 'sigmoid']}, n_jobs=-1, cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    opt.fit(X, y)
    print(opt.best_params_)
    print(opt.best_score_)

    svr = SVR(kernel="", C=0, epsilon=0, gamma=0)
    test_model()

    # print('SVR2')
    # lin_grid_reg = BayesSearchCV(LinearSVR(), {'C': np.logspace(-6, 6, 13), 'epsilon': [.1, .2, .3, .4, .5]},
    #                              n_jobs=-1, cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    # lin_grid_reg.fit(X, y)
    # print(lin_grid_reg.best_params_)
    # print(lin_grid_reg.best_score_)

    print('|')
