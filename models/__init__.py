#!/usr/bin/env python
import math
import pickle
from collections import Counter
from datetime import datetime
from os.path import dirname, abspath
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVR, SVR


class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print(X.size)
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class FeatureRemove(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.drop(columns=[
            # 'mpaa'
            # 'distributor'
            # 'opening_theaters'
            # 'runtime_minutes'
            # 'release_day',
            'release_month'
            # 'release_year'
            # 'budget'
        ])
        return X


# DO BEFORE SCALING
class StarPower(BaseEstimator, TransformerMixin):
    def _get_average_power(self, nconsts: list, star_df: pd.DataFrame, year):
        # TODO ONLY GET FOR PREVIOUS YEARS WORK PER TITLE
        total = 0
        n = 0
        for nconst in nconsts:
            powers = star_df.loc[star_df['star'] == nconst]
            valid = powers.loc[powers['year'] < year]
            if len(valid) != 0:
                wciars = [math.log(iar * math.pow(.8, year - m_year)) for iar, m_year in zip(valid['revenue'], valid['year'])]
                power = (np.array(wciars).sum())
                # power = valid['revenue'].sum()
                total += power
                n = n + 1
        if n: return total / n
        else: return None

    def _get_star_df(self, X: pd.DataFrame, star_type: str):
        if star_type not in ['actors', 'directors']: raise KeyError('Invalid Star Type')
        # Get list of star nconsts
        star_arr_raw = np.unique(X[star_type].to_numpy(dtype=str))
        star_lists = np.char.split(star_arr_raw, sep=',')
        stars = []
        for star_list in star_lists:
            for name in star_list:
                if name not in stars:
                    stars.append(name)
        star_df = pd.DataFrame(columns=['star', 'power', 'year'])
        for star in stars:
            # Get all movies with star
            star_stats_df = X.loc[
                X[star_type].str.contains(star), ['domestic_revenue', 'release_year']].sort_values(
                'release_year')
            rows = [{'star': star, 'revenue': revenue, 'year': year} for revenue, year in
                    zip(star_stats_df['domestic_revenue'], star_stats_df['release_year'])]
            star_df = star_df.append(rows, ignore_index=True)
        return star_df

    def _add_star_powers(self, train_df: pd.DataFrame, director_df: pd.DataFrame, actor_df: pd.DataFrame):
        return_df = train_df.copy()
        return_df['director_power'] = return_df.apply(
            lambda row: self._get_average_power(row['directors'].split(sep=','), director_df, row['release_year']),
            axis=1)
        # return_df['actor_power'] = return_df.apply(
        #     lambda row: self._get_average_power(row['actors'].split(sep=','), actor_df, row['release_year']), axis=1)
        return_df = return_df.drop(columns=['directors'])
        return_df = return_df.drop(columns=['actors'])
        return return_df

    def __init__(self):
        self.director_df = None
        self.actor_df = None

    def fit(self, X, y=None):
        self.director_df = self._get_star_df(X, 'directors')
        # self.actor_df = self._get_star_df(X, 'actors')
        return self

    def transform(self, X):
        X_train_power = self._add_star_powers(X, self.director_df, self.actor_df)
        # X_train_power['actor_power'] = X_train_power['actor_power'].fillna(X_train_power['actor_power'].mean())
        X_train_power['director_power'] = X_train_power['director_power'].fillna(X_train_power['director_power'].mean())
        return X_train_power.drop(columns=['domestic_revenue'])

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    path = dirname(dirname(abspath(__file__))) + '/extract_data/pickled_data/'
    with open(path + 'features.pickle', 'rb') as f:
        df = pickle.load(f)
    with open(path + 'movie_data.pickle', 'rb') as f:
        df_raw = pickle.load(f)

    data = df.drop(columns=['tconst'])

    X = data.drop(columns=['opening_revenue'])
    y = data['opening_revenue']
    y = y.apply(np.log)
    # X['budget'] = X['budget'].apply(np.log)

    # StarPower().fit(data).transform(data).to_csv('power_feats.csv')

    categorical_cols = [
        'release_day',
        # 'release_month', # CONFIRMED TO BE MALADAPTIVE
        'mpaa',
        'distributor'
    ]
    numerical_cols = [
        'release_year',
        'opening_theaters',
        'budget',
        'runtime_minutes',
        'director_power',
        # 'actor_power'  # CONFIRMED TO BE MALADAPTIVE
    ]
    genre_cols = [col_name for col_name in X if 'genre_' in col_name]
    ct = make_column_transformer(
        (StandardScaler(), numerical_cols),
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('drop', genre_cols)
    )


    def print_metrics(stats: dict):
        print(f"Mean Squared Error={stats['Mean Squared Error']}")
        print(f"R^2={stats['R^2']}")


    def get_metrics(y_test, y_pred) -> dict:
        mse = metrics.mean_squared_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        return {"Mean Squared Error": mse, "R^2": r2}


    def get_pipeline(model):
        return make_pipeline(
            StarPower(),
            FeatureRemove(),
            ct,
            model
        )


    def test_model(model, X, y):
        pipeline = get_pipeline(model)
        n = 0
        stats_sum = Counter()
        for train_indices, test_indices in TimeSeriesSplit().split(X):
            pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
            y_pred = pipeline.predict(X.iloc[test_indices])
            y_pred[y_pred < 0] = 0
            fold_stats = get_metrics(y.iloc[test_indices], y_pred)
            # print_metrics(fold_stats)
            stats_sum.update(Counter(fold_stats))
            n += 1
        stats = {k: v / n for k, v in stats_sum.items()}
        print_metrics(stats)


    # Base test
    print('Lin Reg')
    test_model(LinearRegression(), X, y)

    print('Ridge Regression')
    test_model(Ridge(alpha=5), X, y)
    # ridge_grid = GridSearchCV(get_pipeline(Ridge()),
    #                           {'ridge__alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}, n_jobs=-1,
    #                           cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    # ridge_grid.fit(X, y)
    # print(ridge_grid.best_params_)
    # print(ridge_grid.best_score_)

    print('Lasso Regression')
    test_model(Lasso(alpha=1e-15), X, y)
    # lasso_grid = GridSearchCV(get_pipeline(Lasso()),
    #                           {'lasso__alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}, n_jobs=-1,
    #                           cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    # lasso_grid.fit(X, y)
    # print(lasso_grid.best_params_)
    # print(lasso_grid.best_score_)

    print('SVR')
    test_model(SVR(C=100.0, epsilon=0.1, gamma=0.01, kernel='rbf'), X, y) # FINAL
    test_model(SVR(C=10.0, epsilon=0.1, gamma=0.01, kernel='rbf'), X, y)  # BASELINE
    pipeline = get_pipeline(SVR())
    pg = {'svr__C': np.logspace(-4, 4, 9), 'svr__gamma': np.logspace(-4, 2, 7),
          'svr__epsilon': [0, 0.01, 0.1, 0.5, 1, 2, 4], 'svr__kernel': ['rbf', 'sigmoid']}

    opt = GridSearchCV(pipeline, pg, n_jobs=-1, cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    opt.fit(X, y)
    print(opt.best_params_)
    print(opt.best_score_)

    print('LinearSVR')
    # test_model(LinearSVR(C=1.0, epsilon=1), X, y)
    test_model(LinearSVR(C=1.0, epsilon=.5), X, y)  # GENRES FINAL

    # pipeline = get_pipeline(LinearSVR())
    # lin_grid_reg = GridSearchCV(pipeline, {'linearsvr__C': np.logspace(-4, 4, 9),
    #                                        'linearsvr__epsilon': [0, 0.01, 0.1, 0.5, 1, 2, 4]}, n_jobs=-1,
    #                             cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    # lin_grid_reg.fit(X, y)
    # print(lin_grid_reg.best_params_)
    # print(lin_grid_reg.best_score_)

    # print('PolySVR')
    # ppg = {'svr__C': np.logspace(-4, 4, 9), 'svr__gamma': np.logspace(-4, 2, 7), 'svr__degree': [2, 3, 4],
    #        'svr__epsilon': [0.01, 0.1, 0.5, 1, 2, 4], 'svr__kernel': ['poly']}
    # opt = GridSearchCV(pipeline, ppg, n_jobs=-1, cv=TimeSeriesSplit(), verbose=3, scoring="r2")
    # opt.fit(X, y)
    # print(opt.best_params_)
    # print(opt.best_score_)
