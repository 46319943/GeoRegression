from importlib.resources import path

import pandas as pd

random_state = 1003
TOD_weekend_field_list = [
    'Departure interval (seconds)', 'Operating time (min)', 'Average distance between stations(m)',
    'Building density',
    'Residential land proportion', 'Education, health, cultural facilities land proportion',
    'Industrial land proportion',
    'Average distance to training institutions(m)',
    'Average distance to leisure spaces(m)',
    'Average distance to retail/hotel and catering places(m)',
    'Number of red shared bicycle sites', 'Road network betweenness', 'Educated people proportion',
    'ITI * INI '
]
TOD_workday_field_list = [
    'Departure interval (seconds)', 'Operating time (min)',
    'Accessible metro stations number within 20-minutes', 'Building density',
    'Residential land proportion',
    'Industrial land proportion', 'Average distance to training institutions(m)',
    'Average distance to leisure spaces(m)',
    'Average distance to retail/hotel and catering places(m)',
    'Number of red shared bicycle sites',
    'Road network betweenness', 'Educated people proportion',
    'ITI * INI '
]


def load_TOD(workday=True):
    """
    Q: 这个数据去掉时间，直接做回归好像没什么意义。因为每个站点都存在多条数据，相当于直接引入了噪声

    Args:
        workday ():

    Returns:

    """
    if workday:
        with path('georegression.test.data', 'TOD_workday.csv') as filepath:
            df = pd.read_csv(filepath)
    else:
        with path('georegression.test.data', 'TOD_weekend.csv') as filepath:
            df = pd.read_csv(filepath)

    # Store and Drop Predictive Value
    y = df['Ridership'].values
    df = df.drop(columns=['Ridership'])

    # Specify Space and Temporal Feature
    xy_vector = df[['station.X', 'station.Y']].values
    time = df['Time'].values.reshape(-1, 1)

    # Select specific field
    if workday:
        df = df.loc[:, TOD_workday_field_list]
    else:
        df = df.loc[:, TOD_weekend_field_list]
    X = df.values

    # Do Not split the dataset.
    # No necessary for local model and OLS. Use Out-of-bag error for random forest.
    return X, y, xy_vector, time


def load_HP():
    with path('georegression.test.data', 'HousePrice_Shanghai.csv') as filepath:
        df = pd.read_csv(filepath)

    # Reorder the columns
    last_cols = ['Lon84', 'Lat84', 'Time', 'Rent', 'Price']
    new_cols = [col for col in df.columns if col not in last_cols] + last_cols
    df = df[new_cols]

    # Sample only one entity for multiple time slice.
    df = df.sample(frac=1, random_state=random_state).drop_duplicates(subset=new_cols[:-3])

    # Store and Drop Predictive Value
    y = df['Rent'].values
    df = df.drop(columns=['Rent', 'Price'])

    # Specify Space and Temporal Feature
    xy_vector = df[['Lon84', 'Lat84']].values
    time = df['Time'].values.reshape(-1, 1)
    df = df.drop(columns=['Lon84', 'Lat84', 'Time', 'ID', 'FID'])

    X = df.values

    return X, y, xy_vector, time


def load_ESI():
    with path('georegression.test.data', 'EcosystemServicesIndicator.csv') as filepath:
        df = pd.read_csv(filepath)

    # Use subset data.
    # choice_index = np.random.choice(42410, 1000, replace=False)
    # df = df[df['Id'].isin(choice_index)]

    # Specify Space and Temporal Feature
    xy_vector = df[['经度', '纬度']].values
    time = df['年份'].values.reshape(-1, 1)
    df = df.drop(columns=['经度', '纬度', '年份'])

    df = df[['生境', '养分']]

    # Store and Drop Predictive Value
    y_true = df['生境'].values
    df = df.drop(columns=['生境'])

    X = df.values

    return X, y_true, xy_vector, time

    pass


if __name__ == '__main__':
    load_TOD()
