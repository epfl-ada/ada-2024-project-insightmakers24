import os

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def load_time_bias_data(path):
    """
    Loads and do basic cleaning for the time bias analysis

    Parameters
    ----------
    path: The path to the analysis files

    Returns
    -------
    time_df_rateBeer: Clean dataset 1 used for the analysis
    time_df_beerAdvocate: Clean dataset 2 used for the analysis
    """

    # load the data
    time_df_rateBeer = pd.read_csv(os.path.join(path, 'RateBeer/reviews.csv'))
    time_df_beerAdvocate = pd.read_csv(os.path.join(path, 'BeerAdvocate/reviews.csv'))

    # Create time column from the unix format for both dataset
    time_df_rateBeer['time'] = pd.to_datetime(time_df_rateBeer['date'], origin='unix', unit='s')
    time_df_rateBeer['year'] = time_df_rateBeer['time'].dt.year
    time_df_rateBeer['month'] = time_df_rateBeer['time'].dt.month
    time_df_rateBeer['day'] = time_df_rateBeer['time'].dt.day
    time_df_beerAdvocate['time'] = pd.to_datetime(time_df_beerAdvocate['date'], origin='unix', unit='s')
    time_df_beerAdvocate['year'] = time_df_beerAdvocate['time'].dt.year
    time_df_beerAdvocate['month'] = time_df_beerAdvocate['time'].dt.month
    time_df_beerAdvocate['day'] = time_df_beerAdvocate['time'].dt.day

    # Data cleaning
    time_df_rateBeer = time_df_rateBeer.dropna()
    time_df_beerAdvocate = time_df_beerAdvocate.dropna()
    # Remove the beer with less than 10 reviews
    time_df_rateBeer = time_df_rateBeer.groupby('beer_id').filter(lambda x: len(x) > 10)
    time_df_beerAdvocate = time_df_beerAdvocate.groupby('beer_id').filter(lambda x: len(x) > 10)

    return time_df_rateBeer, time_df_beerAdvocate


def get_periode(db_time):
    """
    Separate the data into three periods: Christmas and New Year holiday, Oktoberfest, and the rest of the year

    Parameters
    ----------
    db_time: the dataframe from which to extract the periods

    Returns
    -------
    xmas_hol: Subset of the data for the Christmas and New Year holiday period
    oktoberfest: Subset of the data for the Oktoberfest period
    rest_year: Subset of the data for the remaining time of the year
    """
    # Create two variables that represent the time from Christmas until new year and Oktober fest period
    xmas_hol = db_time[(db_time['month'] == 12) & (db_time['day'] >= 23) | (db_time['month'] == 1) & (db_time['day'] <= 2)]
    oktoberfest = db_time[(db_time['month'] == 9) & (db_time['day'] >= 16) | (db_time['month'] == 10) & (db_time['day'] <= 3)]

    # Create a variable that represents the rest of the year
    rest_year = db_time[~(((db_time['month'] == 12) & (db_time['day'] >= 23) | (db_time['month'] == 1) & (db_time['day'] <= 2)) & (db_time['month'] == 9) & (db_time['day'] >= 16) | (db_time['month'] == 10) & (db_time['day'] <= 3))]

    return xmas_hol, oktoberfest, rest_year


def time_period_rating(db_time):
    """
    Analyzes and visualizes the average rating per year for our time periods

    Parameters
    ----------
    db_time: the dataframe for which to do the plot

    Returns
    -------
    None (plot the data)
    """
    # For this part we want the 2 first days of january to be considered as part of the previous year
    condition = (db_time['month'] == 1) & (db_time['day'] <= 2)
    db_time_mod = db_time.copy()
    db_time_mod.loc[condition, 'year'] -= 1

    xmas_hol, oktoberfest, rest_year = get_periode(db_time_mod)

    # Group by year and compute the average rating for each period per year
    xmas_hol_mean = xmas_hol.groupby(db_time_mod['year'])['rating'].mean()
    oktoberfest_mean = oktoberfest.groupby(db_time_mod['year'])['rating'].mean()
    rest_year_mean = rest_year.groupby(db_time_mod['year'])['rating'].mean()

    # Delete entry for year 2017 as it is not present for all period
    rest_year_mean = rest_year_mean[rest_year_mean.index.isin(oktoberfest_mean.index)]
    xmas_hol_mean = xmas_hol_mean[xmas_hol_mean.index.isin(oktoberfest_mean.index)]

    # Plot the data
    plt.figure(figsize=(12, 6))

    plt.plot(xmas_hol_mean.index, xmas_hol_mean.values, label='Christmas and new year holiday', color='gold', marker='o')
    plt.plot(rest_year_mean.index, rest_year_mean.values, label='Rest of the Year', color='skyblue', marker='o')
    plt.plot(oktoberfest_mean.index, oktoberfest_mean.values, label='Oktoberfest', color='red', marker='o')

    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average rating', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.6)
    plt.xticks(rest_year_mean.index)
    plt.xlim(2000)


def time_period_per_day(db_time):
    """
    Analyzes and compares the number of ratings per day for each year for our time periods

    Parameters
    ----------
    db_time: the dataframe for which to do the plot

    Returns
    -------
    None (plot the data)
    """
    # Number of day per period
    xmas_hol_days = 11
    oktoberfest_days = 18
    rest_year_Days = 365 - xmas_hol_days - oktoberfest_days

    # For this part we want the 2 first days of january to be considered as part of the previous year
    condition = (db_time['month'] == 1) & (db_time['day'] <= 2)
    db_time_mod = db_time.copy()
    db_time_mod.loc[condition, 'year'] -= 1

    xmas_hol, oktoberfest, rest_year = get_periode(db_time_mod)

    # Count the number of ratings per day for each period per year
    xmas_hol_rating = xmas_hol.groupby(db_time['year'])['rating'].count() / xmas_hol_days
    oktoberfest_rating = oktoberfest.groupby(db_time['year'])['rating'].count() / oktoberfest_days
    rest_year_rating = rest_year.groupby(db_time['year'])['rating'].count() / rest_year_Days

    # Delete entry for year 2017 as it is not present for all period
    rest_year_rating  = rest_year_rating[rest_year_rating.index.isin(oktoberfest_rating.index)]
    xmas_hol_rating = xmas_hol_rating[xmas_hol_rating.index.isin(oktoberfest_rating.index)]

    # Plot the data
    plt.figure(figsize=(12, 6))

    plt.plot(xmas_hol_rating.index, xmas_hol_rating.values, label='Christmas and new year holiday', color='gold', marker='o')
    plt.plot(rest_year_rating.index, rest_year_rating.values, label='Rest of the Year', color='skyblue', marker='o')
    plt.plot(oktoberfest_rating.index, oktoberfest_rating.values, label='Oktoberfest', color='red', marker='o')

    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Ratings per Day', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.6)
    plt.xticks(rest_year_rating.index)
    plt.xlim(2000)


def regression_analysis(time_dataset):
    """
    Do a logistic regression analysis to identify factors affecting ratings

    Parameters
    ----------
    time_dataset: the dataframe for which to do the analysis

    Returns
    -------
    None (print the results of the regression)
    """
    time_dataset_cat = time_dataset.copy()
    # Compute the mean of all rating to see which column affect the rating around the mean
    mean = time_dataset_cat['rating'].mean()
    time_dataset_cat['binary_rating'] = (time_dataset_cat['rating'] >= mean).astype(int)

    # Check which columns will impact the most the rating using a linear regression analysis
    log = smf.logit(formula='binary_rating ~ year + month + day', data=time_dataset_cat)
    log = log.fit()
    print(log.summary())
