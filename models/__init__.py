#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import pickle

if __name__ == '__main__':
    path = Path(__file__).parent / 'extract_data/movie_data_m.csv'
    print(path)
    with path.open('rb') as f:
        df = pd.read_csv(f, header=0, encoding='utf-8', dtype='str')

