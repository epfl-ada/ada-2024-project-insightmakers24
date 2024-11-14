import os.path

import pandas as pd


def txt_to_csv(path):
    """
    Load and convert a txt file to a csv file

    Parameters
    ----------
    path: The path to the txt file
    """

    with open(path, 'r') as file:
        row = {}
        index = 0
        # Reads the file line by line
        for line in file:
            line = line.strip()
            # Split only at the first ':'
            parts = line.split(":", 1)
            # If there was a split between key:value, we add it to the row
            # Otherwise we are at the end of the row, and we append the row to the csv file
            if len(parts) > 1:
                column_name = parts[0].strip()
                value = parts[1].strip()
                row[column_name] = value
            else:
                df = pd.DataFrame(data=row, index=[index])
                index += 1
                row = {}
                new_path = path.replace(".txt", ".csv")
                header = False if os.path.exists(new_path) else True  # Header for the first row only
                df.to_csv(new_path, mode='a', index=False, header=header)


def load_country_bias_data(path):
    """
    Loads and do basic cleaning for the country bias analysis

    Parameters
    ----------
    path: The path to the analysis files

    Returns
    -------
    Clean dataset used for the analysis
    """

    # Load rating, user and breweries infos
    df_ratings = pd.read_csv(path + 'ratings.csv')
    df_users = pd.read_csv(path + 'users.csv')
    df_breweries = pd.read_csv(path + 'breweries.csv')

    df_users_filtered = df_users[df_users.nbr_ratings > 1]
    df_ratings_clean = df_ratings.dropna(subset=['appearance', 'aroma', 'palate', 'taste', 'overall', 'rating'])

    # Select relevant columns
    df_ratings_locs = df_ratings_clean[
        ['beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'user_id', 'rating']]

    # Merge the location for the beers and users
    df_ratings_locs = df_ratings_locs.merge(df_users_filtered[['user_id', 'location']], on='user_id', how='left')
    df_ratings_locs = df_ratings_locs.rename(columns={'location': 'user_location'})
    df_ratings_locs = df_ratings_locs.merge(df_breweries[['id', 'location']], left_on='brewery_id', right_on='id',
                                            how='left')
    df_ratings_locs = df_ratings_locs.drop(columns='id')
    df_ratings_locs = df_ratings_locs.rename(columns={'location': 'beer_location'})

    return df_ratings_locs.dropna()
