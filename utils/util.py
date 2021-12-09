from os.path import dirname, abspath

import pandas as pd


def adjust_for_inflation(cpi_df: pd.DataFrame, value: float, from_month: int, from_year: int, to_month: int,
                         to_year: int) -> float:
    if from_year == cpi_df['Year'].iat[-1] and int((cpi_df['Period'].iat[-1])[1:]) < from_month:
        from_cpi = cpi_df['Value'].iat[-1]
    else:
        from_period = f'M{from_month:02}'
        from_cpi = cpi_df.loc[cpi_df['Year'] == from_year].loc[cpi_df['Period'] == from_period, 'Value'].iat[0]

    to_period = f'M{to_month:02}'
    to_cpi = cpi_df.loc[cpi_df['Year'] == to_year].loc[cpi_df['Period'] == to_period, 'Value'].iat[0]

    adjusted_value = value * (to_cpi / from_cpi)
    return adjusted_value


def get_cpi_df() -> pd.DataFrame:
    cpi_df = pd.read_csv(f'{dirname(dirname(abspath(__file__)))}\\data\\cpi_data.csv', header=0,
                         encoding='utf-8')
    cpi_df = cpi_df.drop(columns=['Series ID', 'Label'])
    return cpi_df
