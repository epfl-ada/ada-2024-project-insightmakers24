import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data_first_rating(data_path='data/BeerAdvocate/ratings.csv', min_count=1000, first=True):
    """
    Return the dataframe used for first vs other rating influence

    Parameters
    ----------
    data_path: the path to the data files
    min_count: the minimum of review a beer need so that its rating is considered valid
    first: if true, compare the first rating, if false take the last one (only to show differences)

    Returns
    -------
    The dataframe used for the analysis
    """

    # load the rating dataset
    ratings_df = pd.read_csv(data_path)

    # select usefull columns and sort by date
    rating_per_date = ratings_df[['beer_id', 'beer_name', 'date', 'rating']].sort_values(by=['beer_id', 'date'], ascending=first)
    rating_per_date = rating_per_date.dropna()

    # add a columns containing the count of review per beers
    rating_per_date['count'] = rating_per_date.groupby('beer_id')['beer_id'].transform('count')

    # drop the columns with unsignificative final rate
    rating_per_date = rating_per_date[rating_per_date['count'] > min_count]

    # split the dataset between first rating and every other
    first_rating = rating_per_date.groupby('beer_id').head(1)
    others_rating = rating_per_date.drop(first_rating.index)
    others_rating = others_rating.groupby(by='beer_id').agg(beer_name=('beer_name', 'min'), other_rating=('rating', 'mean'), other_std=('rating', 'std'))

    # merge the two dataset with each rating in distinct columns
    first_vs_other_rating = pd.merge(others_rating, first_rating[['beer_id', 'rating']], on='beer_id', how='left')
    first_vs_other_rating = first_vs_other_rating.rename(columns={'rating': 'first_rating'})

    return first_vs_other_rating


def loadRatingDate(data_path='data/BeerAdvocate/ratings.csv', min_count=1000):
    """
    Return the dataframe used comparing value vs date of a rating

    Parameters
    ----------
    data_path: the path to the data files
    min_count: the minimum of review a beer need so to be considered

    Returns
    -------
    The dataframe used for the analysis
    """

    # load the rating dataset
    ratings_df = pd.read_csv(data_path)

    # keep meaningful columns and sort by beer (using beer_id) and by date
    rating_per_date = ratings_df[['beer_id', 'beer_name', 'date', 'rating']].sort_values(by=['beer_id', 'date'],
                                                                                         ascending=True)
    rating_per_date = rating_per_date.dropna()

    # count the total number of rating per beers
    rating_per_date['count'] = rating_per_date.groupby('beer_id')['beer_id'].transform('count')

    # keep only the popular beer that have enought rating for this analysis
    rating_per_date = rating_per_date[rating_per_date['count'] > min_count]

    return rating_per_date


def moving_average(data_col, window_size=5):
    """
    Return a smooth version of the data column

    Parameters
    ----------
    data_df: df column we want to smooth
    window_size: number of sample we use to calculate the mean

    Returns
    -------
    The smoothed version of the data column
    """

    data_series = pd.Series(data_col)
    return data_series.rolling(window=window_size, min_periods=1).mean()


def plot_moving_average_unique(data_df, idx, window_size):
    """
    Plot the distribution of rating sorted in order, and smoothed using moving average

    Parameters
    ----------
    data_df: the dataframe used in the plot
    idx: the index of the beer we want to plot
    window_size: number of sample we use to calculate the mean
    """

    # calculate moving average
    smoothed_rating = moving_average(data_df['rating'].iloc[idx], window_size)

    # plot the rating over time
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_rating)
    plt.title(f'Mean ratings over time for {data_df["beer_name"].iloc[idx]}', fontsize=14, fontweight='bold')
    plt.xlabel('Rating index')
    plt.ylabel('Rating')

    plt.tight_layout()
    plt.show()


def plot_smoothed_rating_diff(df, min_count, window_size=100):
    """
    Plot the distribution of distance between rating and the mean, sorted in order, and smoothed using moving average

    Parameters
    ----------
    df: the dataframe used in the plot
    min_count: the count f rating of the less popular item
    window_size: number of sample we use to calculate the mean
    """

    # calculate the mean of all ratings on df
    mean_list = np.zeros(min_count)
    for row in range(len(df)):
        for i in range(min_count):
            mean_list[i] += df.iloc[row, 1][i]

    mean_list = np.array(mean_list) / len(df)

    # apply moving average
    smoothed_mean_list = moving_average(mean_list, window_size=window_size)

    # plot the results
    plt.plot(mean_list, alpha=0.4, label='Average Difference')
    plt.plot(smoothed_mean_list, label='Smoothed average Difference')
    plt.legend()
    plt.xlabel("Rating Index")
    plt.ylabel("Average Difference")
    plt.title("Average Difference from the mean rating (3)")
    plt.tight_layout()
    plt.show()


def loadBrosScoreDf(data_path='data/BeerAdvocate/beers.csv'):
    """
    Return the dataframe of the bros score per beer

    Parameters
    ----------
    data_path: the path to the data files

    Returns
    -------
    The dataframe used for the analysis
    """

    # load the bros score df
    bro_score_per_beer = pd.read_csv(data_path, header=0)

    # keep meaningful columns
    bro_score_per_beer = bro_score_per_beer[['beer_id', 'bros_score']]

    # drop row without bros score
    bro_score_per_beer = bro_score_per_beer.dropna()
    return bro_score_per_beer


def rescaleBroScore(df):
    """
    Return the dataframe with a rescaled version of the bros score

    Parameters
    ----------
    data_path: the path to the data files

    Returns
    -------
    The dataframe used for the analysis
    """

    # rescale bro score from 0-100 to 1-5 rating
    df['bros_score_1_5'] = 1 + (4 * df['bros_score']) / 100

    # correct the bias between ratings
    df['bros_score_1_5'] = df['bros_score_1_5'] - df['bros_score_1_5'].mean() + df['other_rating'].mean()

    return df


def prepareHighVsLow(df, bros_score_df, high_lim=4, low_lim=2):
    """
    Return the dataframe used for first vs other rating influence

    Parameters
    ----------
    df: the df containing first rating and overall rating
    bros_score_df: the df containing the bro score
    high_lim: the minimum rating considered as a high rate
    low_lim: the maximum rating considered as a low rate

    Returns
    -------
    Two dataframe containing both overall and bros score difference
    """

    # merges these df and the bros_score df
    merge_df = df.merge(bros_score_df, left_on='beer_id', right_on='beer_id')

    # rescale bros_score
    merge_df = rescaleBroScore(merge_df)

    # separate the high and low first rating using corresponding limits
    high_ratings_with_bros_score = merge_df[merge_df['first_rating'] >= high_lim]
    low_ratings_with_bros_score = merge_df[merge_df['first_rating'] <= low_lim]

    # calculate the diff between bros_score and average rating for both df
    low_ratings_with_bros_score['rating_diff'] = low_ratings_with_bros_score['other_rating'] - \
                                                 low_ratings_with_bros_score['bros_score_1_5']
    high_ratings_with_bros_score['rating_diff'] = high_ratings_with_bros_score['other_rating'] - \
                                                  high_ratings_with_bros_score['bros_score_1_5']

    return high_ratings_with_bros_score, low_ratings_with_bros_score
