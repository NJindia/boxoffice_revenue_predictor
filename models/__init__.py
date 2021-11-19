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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.svm import SVR

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None

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

    data = df.drop(columns=['tconst', 'domestic_revenue'])


    # eigen(X)

    def print_metrics(stats: dict):
        print(f"Mean Squared Log Error={stats['Mean Squared Log Error']}")
        print(f"Root Mean Squared Log Error={stats['Root Mean Squared Log Error']}")
        print(f"Mean Squared Error={stats['Mean Squared Error']}")
        print(f"Root Mean Squared Error={stats['Root Mean Squared Error']}")
        print(f"R^2={stats['R^2']}")


    # FROM https://github.com/PermanAtayev/Movie-revenue-prediction/blob/master/movie_revenue_predict.ipynb
    def get_metrics(y_test, y_pred) -> dict:
        msle = metrics.mean_squared_log_error(y_test, y_pred)
        rmsle = np.sqrt(metrics.mean_squared_log_error(y_test, y_pred))
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        return {"Mean Squared Log Error": msle, "Root Mean Squared Log Error": rmsle, "Mean Squared Error": mse,
                "Root Mean Squared Error": rmse, "R^2": r2}


    def test_model(model, data):
        def get_average_power(nconsts: list, df: pd.DataFrame):
            total_s = 0
            total_p = 0
            n = 0
            for nconst in nconsts:
                if len(df.loc[df['director'] == nconst]) == 0:
                    # Todo HANDLE MISSING DIRECTOR
                    return [0, 0]
                    pass
                else:
                    total_s = total_s + df.loc[df['director'] == nconst, 'sum_power'].to_numpy()[0]
                    total_p = total_p + df.loc[df['director'] == nconst, 'wciar_power'].to_numpy()[0]
                n = n + 1
            avg_s = total_s / n
            avg_p = total_p / n
            return [avg_s, avg_p]

        def get_director_df(train_df: pd.DataFrame):
            # Get list of director nconsts
            directors_arr_raw = np.unique(train_df['directors'].to_numpy(dtype=str))
            directors_lists = np.char.split(directors_arr_raw, sep=',')
            directors = []
            for director_list in directors_lists:
                for name in director_list:
                    if name not in directors:
                        directors.append(name)
            director_df = pd.DataFrame(columns=['director', 'sum_power', 'wciar_power'])
            for director in directors:
                # Get all movies with director
                director_stats_df = train_df.loc[
                    train_df['directors'].str.contains(director), ['directors', 'opening_revenue_temp',
                                                                   'release_year_temp']]

                revenue_sum = math.log(director_stats_df['opening_revenue_temp'].sum())

                curr_year = datetime.datetime.now().year
                wciars = [math.log(iar * math.pow(.8, curr_year - year)) for iar, year in
                          zip(director_stats_df['opening_revenue_temp'], director_stats_df['release_year_temp'])]
                power = (np.array(wciars).sum())
                row = {'director': director, 'sum_power': revenue_sum, 'wciar_power': power}
                director_df = director_df.append(row, ignore_index=True)
            return director_df

        def add_director_star_power(train_df: pd.DataFrame, director_df: pd.DataFrame):
            return_df = train_df.copy()
            # train_df['director_power_s'] = train_df.apply(
            #     lambda x: get_average_power(x['directors'].split(sep=','), director_df)[0], axis=1)
            return_df['director_power_p'] = return_df.apply(
                lambda row: get_average_power(row['directors'].split(sep=','), director_df)[1], axis=1)
            return_df = return_df.drop(columns=['directors'])
            return return_df

        data['release_year_temp'] = data['release_year']
        data['opening_revenue_temp'] = data['opening_revenue']

        split_indices = TimeSeriesSplit().split(data)
        stats_sum = Counter()
        n = 0
        for train_indices, test_indices in split_indices:
            director_df = get_director_df(data.iloc[train_indices])

            model_data = data.drop(columns=['release_year_temp', 'opening_revenue_temp'])
            train = model_data.iloc[train_indices]

            X_train_power = train.drop(columns=['directors'])
            # X_train_power = add_director_star_power(train, director_df)
            X_train_power = X_train_power.drop(columns=['opening_revenue'])

            scaler = StandardScaler()
            scale_cols = ['release_year', 'opening_theaters', 'runtime_minutes']
            X_train_power[scale_cols] = scaler.fit_transform(X_train_power[scale_cols])
            y_train = train['opening_revenue'].apply(lambda row: math.log(row))

            test = model_data.iloc[test_indices]
            test_power = test.drop(columns=['directors'])
            # test_power = add_director_star_power(test, director_df).dropna()
            X_test = test_power.drop(columns=['opening_revenue'])
            X_test[scale_cols] = scaler.transform(X_test[scale_cols])
            y_test = test_power['opening_revenue'].apply(lambda row: math.log(row))

            model.fit(X_train_power, y_train)
            y_pred = model.predict(X_test)
            stats_sum += Counter(get_metrics(y_test, y_pred))
            n += 1
        stats = {k: v / n for k, v in stats_sum.items()}
        print_metrics(stats)


    # Base test
    regr = linear_model.LinearRegression()

    print('Lin Reg Baseline')
    ridge = linear_model.Ridge()
    lasso = linear_model.Lasso()
    test_model(regr, data)
    test_model(ridge, data)
    test_model(lasso, data)

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

    print('SVR')
    params = [{"kernel": ["rbf"], "gamma": np.logspace(-9, 9, 19), "C": np.logspace(-9, 9, 19),
               "epsilon": [0.1, 0.2, 0.5, 0.3]}]

    X = data.drop(columns=['opening_revenue'])
    y = data['opening_revenue']
    grid_reg = GridSearchCV(SVR(), params, n_jobs=-1, cv=TimeSeriesSplit(), verbose=2)
    grid_reg.fit(X, y)
    print(grid_reg.cv_results_)
