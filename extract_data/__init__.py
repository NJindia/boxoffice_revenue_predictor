#!/usr/bin/env python
import gzip
import pickle
from warnings import simplefilter

import dateparser
import numpy as np
import pandas as pd
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor, defer

from spiders import imdb_spider, box_office_spider
from utils.util import adjust_for_inflation, get_cpi_df

if __name__ == '__main__':
    SPLIT_RATIO = 0.8
    pd.options.mode.chained_assignment = None
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    get_raw_data = False
    scrape_boxofficemojo = False
    scrape_mpaa = False
    load_imdb_data = False

    if get_raw_data:
        if load_imdb_data:
            # Datasets (tsv.gz files) downloaded from https://datasets.imdbws.com/
            # Documentation for datasets can be found at https://www.imdb.com/interfaces/
            print('Loading title.basics.tsv.gz')
            with gzip.open("data/title.basics.tsv.gz", "rt", encoding='utf-8') as f:
                df = pd.read_csv(f, sep="\t", header=0, encoding='utf-8', dtype='str')
                # if IMDB doesn't have this info, we can assume the movie is invalid
                df = df[df["titleType"] == "movie"]
                df = df[df["runtimeMinutes"] != "\\N"]
                df = df[df["startYear"] != "\\N"]
                df = df[df["genres"] != "\\N"]
                basics_df = df.drop(columns=['endYear', 'originalTitle', 'isAdult', 'genres', 'titleType'])

            print('Loading title.crew.tsv.gz')
            with gzip.open("data/title.crew.tsv.gz", "rt", encoding='utf-8') as f:
                df = pd.read_csv(f, sep="\t", header=0, encoding='utf-8', dtype='str')
                crew_df = df[df["directors"] != "\\N"]
            movie_raw_trimmed = pd.merge(basics_df, crew_df, on='tconst')

            print('Loading title.principals.tsv.gz')
            with gzip.open("data/title.principals.tsv.gz", "rt", encoding='utf-8') as f:
                df = pd.read_csv(f, sep="\t", header=0, encoding='utf-8', dtype='str')
                nconsts = df[['tconst', 'nconst']]
                actors = df.loc[(df['category'] != 'director') & (df['category'] != 'writer'), ['tconst', 'nconst']]
                actors = actors.rename(columns={"nconst": "actors"})
                tconsts = pd.unique(nconsts['tconst'])
                # Remove titles with no cast
                filtered_tconsts = pd.merge(movie_raw_trimmed, pd.DataFrame(data={'tconst': tconsts}), on='tconst')[
                    'tconst']

                c_df = pd.merge(filtered_tconsts, nconsts, on='tconst', how='left').groupby('tconst',
                                                                                            as_index=False).agg(
                    {'nconst': ','.join})
                actors_m = actors.groupby('tconst', as_index=False).agg({'actors': ','.join})
            cast_df = pd.merge(c_df, actors_m, on='tconst', how='left')

            imdb_data = pd.merge(movie_raw_trimmed, cast_df, on='tconst')

            # TODO CHECK IF EVERY MEMBER OF CAST IS DEAD
            # print('Loading name.basics.tsv.gz')
            # with gzip.open("data/name.basics.tsv.gz", "rt", encoding='utf-8') as f:
            #     df = pd.read_csv(f, sep="\t", header=0, encoding='utf-8', dtype='str')
            #     names_df = df[['nconst', 'deathYear']]
            # dead = names_df.loc[names_df['deathYear'] != '\\N', 'nconst'].to_numpy()
            # imdb_data = imdb_data_untrimmed.apply(lambda row: row if np.all(np.in1d(row['nconst'].split(sep=','),dead, invert=True)) else np.nan, axis=1).dropna()

            with open('pickled_data/imdb_movie_data.pickle', 'wb') as f:
                pickle.dump(imdb_data,
                            f)  # ratings_data = pd.merge(imdb_data, ratings_df, on='tconst')  # ratings_data.to_csv("ratings.csv", encoding='utf-8-sig')
        else:
            with open('pickled_data/imdb_movie_data.pickle', 'rb') as f:
                imdb_data = pickle.load(f)

        settings = get_project_settings()
        process = CrawlerProcess(settings)


        @defer.inlineCallbacks
        def crawl(process: CrawlerProcess):
            if scrape_boxofficemojo: yield process.crawl(box_office_spider.BoxOfficeSpider, df=imdb_data)
            with open('pickled_data/box_office_data.pickle', 'rb') as f:
                box_office_data_temp = pickle.load(f)
            # TODO: IMPROVE THIS! IMDB has bad ratings for movies that have been re-rated, FilmRatings is hard to crawl but will be 100% accurate
            if scrape_mpaa: yield process.crawl(imdb_spider.IMDBSpider, df=box_office_data_temp)
            if reactor.running: reactor.callFromThread(reactor.stop)


        if scrape_mpaa or scrape_boxofficemojo:
            crawl(process)
            reactor.run()
        with open('pickled_data/box_office_data.pickle', 'rb') as f: box_office_data_raw = pickle.load(f)
        with open('pickled_data/mpaa_data.pickle', 'rb') as f: mpaa_data = pickle.load(f)
        box_office_data = pd.merge(box_office_data_raw, mpaa_data, how='left', on='tconst', suffixes=('_l', '_r'))
        box_office_data.loc[box_office_data['mpaa_l'] == '\\N', 'mpaa_l'] = box_office_data.loc[
            box_office_data['mpaa_l'] == '\\N', 'mpaa_r']
        box_office_data = box_office_data.rename(columns={'mpaa_l': 'mpaa'})
        movie_data = pd.merge(imdb_data, box_office_data, how="right", on="tconst")
        movie_data = movie_data.drop(columns=['primaryTitle', 'mpaa_r'])
        movie_data.to_csv("movie_data.csv", encoding='utf-8-sig')
        with open('pickled_data/movie_data.pickle', 'wb') as f: pickle.dump(movie_data, f)

    with open('pickled_data/movie_data.pickle', 'rb') as f: movie_data = pickle.load(f)

    # Get features
    feature_df = pd.DataFrame(data=movie_data[['tconst', 'directors', 'actors', 'distributor', 'mpaa']],
                              columns=['tconst', 'directors', 'actors', 'distributor', 'mpaa'])
    feature_df['release_year'] = pd.to_numeric(movie_data['startYear'])
    feature_df['budget'] = pd.to_numeric(movie_data['budget'].str.replace(r'\D+', '', regex=True))
    feature_df['opening_revenue'] = pd.to_numeric(movie_data['opening_revenue'].str.replace(r'\D+', '', regex=True))
    feature_df['domestic_revenue'] = pd.to_numeric(movie_data['domestic_revenue'].str.replace(r'\D+', '', regex=True))
    feature_df['opening_theaters'] = pd.to_numeric(movie_data['opening_theaters'].str.replace(r'\D+', '', regex=True))
    feature_df['runtime_minutes'] = pd.to_numeric(movie_data['runtimeMinutes'].str.replace(r'\D+', '', regex=True))

    # Convert release_date to release month and sort by release date
    feature_df['release_month'] = movie_data['release_date'].apply(lambda x: dateparser.parse(x).month)
    feature_df['release_date'] = movie_data['release_date'].apply(lambda x: dateparser.parse(x))
    feature_df['release_day'] = movie_data['release_date'].apply(lambda x: dateparser.parse(x).weekday())
    feature_df = feature_df.sort_values(by='release_date', axis=0, ascending=True).drop(columns=['release_date'])

    # Adjust monetary fields for inflation using CPI
    cpi_target_month = 1
    cpi_target_year = 2021
    cpi_df = get_cpi_df()
    feature_df['budget'] = feature_df.apply(
        lambda row: adjust_for_inflation(cpi_df, row['budget'], row['release_month'], row['release_year'],
                                         cpi_target_month, cpi_target_year), axis=1)
    feature_df['domestic_revenue'] = feature_df.apply(
        lambda row: adjust_for_inflation(cpi_df, row['domestic_revenue'], row['release_month'], row['release_year'],
                                         cpi_target_month, cpi_target_year), axis=1)
    feature_df['opening_revenue'] = feature_df.apply(
        lambda row: adjust_for_inflation(cpi_df, row['opening_revenue'], row['release_month'], row['release_year'],
                                         cpi_target_month, cpi_target_year), axis=1)

    # # Get distributor features
    # distributors = pd.unique(movie_data['distributor'])
    # d_df = movie_data[['tconst', 'distributor']]
    # for distributor in distributors:
    #     col_name = ('distributor_' + '_'.join(distributor.split(' '))).lower()
    #     d_df[col_name] = 0
    #     d_df.loc[d_df['distributor'] == distributor, col_name] = 1
    # d_df = d_df.drop(columns=['distributor'])
    # feature_df = pd.merge(feature_df, d_df, on='tconst')
    #
    # Get genre features
    genres = np.array([], dtype=str)
    for entry in movie_data['genres']:
        if ',' in entry:
            for genre in entry.split(','):
                if genre not in genres:
                    genres = np.append(genres, genre)
    g_df = movie_data[['tconst', 'genres']]
    for genre in genres:
        col_name = ('genre_' + '_'.join(genre.split(' '))).lower()
        g_df[col_name] = 0
        g_df.loc[g_df['genres'].str.contains(genre), col_name] = 1
    g_df = g_df.drop(columns=['genres'])
    feature_df = pd.merge(feature_df, g_df, on='tconst')

    # # Get Release Day features
    # movie_data['release_day'] = movie_data['release_date'].apply(lambda x: dateparser.parse(x).weekday())
    # m_df = movie_data[['tconst', 'release_day']]
    # for i in range(7):
    #     col_name = 'day_' + str(i)
    #     m_df[col_name] = 0
    #     m_df.loc[m_df['release_day'] == i, col_name] = 1
    # m_df = m_df.drop(columns=['release_day'])
    # feature_df = pd.merge(feature_df, m_df, on='tconst')

    # # Get Release Month features
    # m_df = feature_df[['tconst', 'release_month']]
    # for i in range(1, 13):
    #     col_name = 'month_' + str(i)
    #     m_df[col_name] = 0
    #     m_df.loc[m_df['release_month'] == i, col_name] = 1
    # m_df = m_df.drop(columns=['release_month'])
    # feature_df = pd.merge(feature_df, m_df, on='tconst').drop(columns='release_month')
    #
    # # Get MPAA features
    # mpaa_ratings = pd.unique(movie_data['mpaa'])
    # m_df = movie_data[['tconst', 'mpaa']]
    # for rating in mpaa_ratings:
    #     col_name = 'mpaa_' + rating.lower()
    #     m_df[col_name] = 0
    #     m_df.loc[m_df['mpaa'] == rating, col_name] = 1
    # m_df = m_df.drop(columns=['mpaa'])
    # feature_df = pd.merge(feature_df, m_df, on='tconst')
    #
    feature_df.to_csv('features.csv')
    with open('./pickled_data/features.pickle', 'wb') as f:
        pickle.dump(feature_df, f)

    graph_df = feature_df.merge(movie_data[['tconst', 'mpaa', 'distributor', 'opening_revenue', 'budget']], on='tconst',
                                suffixes=('_cpi', '_base'))
    graph_df.to_csv('graph_feats.csv')
