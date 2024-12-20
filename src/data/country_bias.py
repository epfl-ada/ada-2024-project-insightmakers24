import os

import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

from psmpy import PsmPy


def load_country_data(data_path='data', beer_advocate=True):
    """
    Loads either BeerAdvocate or RateBeer dataset and does basic data cleaning and features selection.

    Parameters
    ----------
    data_path: The path to the BeerAdvocate dataset.
    beer_advocate: True if chooses BeerAdvocate dataset, False if RateBeer dataset.

    Returns
    -------
    The cleaned ratings, users, breweries and beers data
    """

    if beer_advocate:
        data_path = os.path.join(data_path, 'BeerAdvocate')
    else:
        data_path = os.path.join(data_path, 'RateBeer')

    # Keep columns of interest and drop nans
    df_ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    df_ratings_clean = df_ratings.drop(columns=['beer_name', 'brewery_name', 'style', 'user_name', 'text'])
    print(f'Keep {len(df_ratings_clean.dropna()):,} ratings out of '
          f'{len(df_ratings_clean):,} ({100 * len(df_ratings_clean.dropna()) / len(df_ratings_clean):.2f}%)')
    df_ratings_clean = df_ratings_clean.dropna()

    df_users = pd.read_csv(os.path.join(data_path, 'users.csv'))
    col_to_remove = ['nbr_reviews'] if beer_advocate else []
    df_users_clean = df_users.drop(columns=['user_name', 'nbr_ratings', 'joined'] + col_to_remove)
    print(f'Keep {len(df_users_clean.dropna()):,} users out of '
          f'{len(df_users_clean):,} ({100 * len(df_users_clean.dropna()) / len(df_users_clean):.2f}%)')
    df_users_clean = df_users_clean.dropna()

    df_breweries = pd.read_csv(os.path.join(data_path, 'breweries.csv'))
    df_breweries_clean = df_breweries.drop(columns=['name', 'nbr_beers']).rename(columns={'id': 'brewery_id'})
    print(f'Keep {len(df_breweries_clean.dropna()):,} breweries out of '
          f'{len(df_breweries_clean):,} ({100 * len(df_breweries_clean.dropna()) / len(df_breweries_clean):.2f}%)')
    df_breweries_clean = df_breweries_clean.dropna()

    df_beers = pd.read_csv(os.path.join(data_path, 'beers.csv'))
    col_to_keep = ['bros_score'] if beer_advocate else []
    df_beers_clean = df_beers[['beer_id'] + col_to_keep]
    print(f'Keep {len(df_beers_clean.dropna()):,} beers out of '
          f'{len(df_beers_clean):,} ({100 * len(df_beers_clean.dropna()) / len(df_beers_clean):.2f}%)')
    df_beers_clean = df_beers_clean.dropna()

    return df_ratings_clean, df_users_clean, df_breweries_clean, df_beers_clean

def merge_country_data(df_ratings, df_users, df_breweries, df_beers):
    """
    Merge ratings, users, breweries and beers dataframes together and check if the rating is from the same place
    as the beer.

    Parameters
    ----------
    df_ratings: The cleaned ratings dataframe.
    df_users: The cleaned users dataframe.
    df_breweries: The cleaned breweries dataframe.
    df_beers: The cleaned beers dataframe.

    Returns
    -------
    The merged dataframe
    """

    # Merge ratings, users, breweries and beers together
    df_merged = df_ratings.merge(df_beers, how='inner', on='beer_id')
    df_merged = df_merged.merge(df_users, how='inner', on='user_id')
    df_merged = df_merged.merge(df_breweries, how='inner', on='brewery_id')

    # Add a domestic_rating column that is 1 when the user comes from the same location as the beer
    df_merged['domestic_rating'] = df_merged.location_x == df_merged.location_y
    # Remove columns not used in the analysis
    df_merged = df_merged.drop(columns=['location_x', 'location_y', 'user_id', 'brewery_id'])

    print(f'There are {len(df_merged[df_merged.domestic_rating == True]):,} domestic ratings vs '
          f'{len(df_merged[df_merged.domestic_rating == False]):,} international ratings')

    return df_merged

def regression_analysis(df, beer_advocate=True):
    """
    Regression analysis of the given dataset features.

    Parameters
    ----------
    df: Given dataset.
    beer_advocate: True if BeerAdvocate dataset, False if RateBeer dataset.
    """

    # Standardize features except categorical features, beer ids and the dependant variable "rating"
    Z = df.copy()
    if beer_advocate:
        columns = ['domestic_rating', 'beer_id', 'review', 'rating']
    else:
        columns = ['domestic_rating', 'beer_id', 'rating']
    for column in Z.columns:
        if column not in columns:
            Z[column] = (Z[column] - Z[column].mean()) / Z[column].std()
    # Transform binary variable to int instead of bool
    if beer_advocate:
        Z['review'] = Z['review'].astype(int)
    Z['domestic_rating'] = Z['domestic_rating'].astype(int)

    # Create and fit regression model
    if beer_advocate:
        formula = 'rating ~ abv + date + C(domestic_rating) + C(review) + bros_score'
    else:
        formula = 'rating ~ abv + date + C(domestic_rating)'
    reg_mod = smf.ols(formula=formula, data=Z)
    reg_res = reg_mod.fit()

    coefficients = reg_res.params.drop("Intercept")
    standard_errors = reg_res.bse.drop("Intercept")

    plt.figure(figsize=(10, 6))
    plt.bar(coefficients.index, coefficients, yerr=standard_errors, color='skyblue', edgecolor='black', capsize=5)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Linear Regression Coefficients with Standard Errors")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Variables")
    plt.xticks(rotation=45)
    plt.tight_layout()
    print(reg_res.summary())

def compare_control_and_treatment(df, beer_advocate=True):
    """
    Comparison of control and treatment groups for the given dataframe.

    Parameters
    ----------
    df: Treatment and control dataset.
    beer_advocate: True if BeerAdvocate dataset, False if RateBeer dataset.
    """

    df_treatment = df[df.domestic_rating == True]
    df_control = df[df.domestic_rating == False]

    # Plot two histograms in the same cell
    plt.figure()
    axe1 = sns.histplot(df_treatment.rating, stat='density', kde=True, color='blue', label='Treated', bins=50)
    axe1 = sns.histplot(df_control.rating, stat='density', kde=True, color='orange', label='Control', bins=50)
    axe1.set(title='Ratings relative frequencies', xlabel='Ratings', ylabel='Frequencies')
    axe1.legend()
    if beer_advocate:
        axe1.set_ylim(0, 1.5)
    else:
        axe1.set_ylim(0, 1)

    # Plot boxplot comparing basic stats of the two groups
    plt.figure()
    axe2 = sns.boxplot(x='domestic_rating', y='rating', data=df)
    axe2.set(title='Ratings statistics', xlabel='Treatment', ylabel='Ratings')

    # Plot the means for each group and error bars
    plt.figure()
    axe3 = sns.barplot(x='domestic_rating', y='rating', data=df)
    axe3.set(title='Mean ratings', xlabel='Treatment', ylabel='Rating')
    if beer_advocate:
        axe3.set_ylim(3.8, 4)
    else:
        axe3.set_ylim(3.2, 3.4)

    plt.tight_layout()

def psm_balancing(df, frac_kept=0.02):
    """
    Does a propensity score matching on the data set to balance it. Does it on a subset as the matching can take some
    time.

    Parameters
    ----------
    df: The dataset to balance
    frac_kept: Fraction of the data kept

    Returns
    -------
    The balanced dataframe.
    """

    df_prop = df.reset_index()
    psm = PsmPy(
        df_prop.sample(frac=frac_kept),
        treatment='domestic_rating',
        exclude=['beer_id', 'rating'],
        indx='index'
    )
    psm.logistic_ps(balance=True)
    psm.knn_matched(matcher='propensity_logit', replacement=False, drop_unmatched=True)
    psm.plot_match()

    return df.iloc[psm.df_matched['index']]

def load_beer_consumption_data(data_path, beer_advocate=True):
    """
    Return the dataframe used for the beer consumption analysis

    Parameters
    ----------
    data_path:
    beer_advocate:

    Returns
    -------
    The dataframe used for the analysis
    """
    df_consm = pd.read_csv(os.path.join(data_path, 'BeerConsumption.csv'))

    if beer_advocate:
        data_path = os.path.join(data_path, 'BeerAdvocate')
    else:
        data_path = os.path.join(data_path, 'RateBeer')

    # load the 3 differents datasets
    df_breweries = pd.read_csv(os.path.join(data_path, 'breweries.csv'))
    df_ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))

    # merge the 3 dataset by beer_id and location
    df_merged = df_ratings.merge(df_breweries, left_on='beer_id', right_index=True)

    df_merged['location'] = df_merged['location'].str.strip().str.lower()
    df_consm['country'] = df_consm['country'].str.strip().str.lower()

    df_merged = df_merged.merge(df_consm, left_on='location', right_on='country')

    # merge 2020 dans 2021 beer conumption to get an average
    df_merged['beer_consumption_per_capita'] = (df_merged['BeerConsumptionPerCapitakg2021'] + df_merged['beerConsumptionByCountry_consmPerCap'])/2

    # keep usefull columns
    df_merged = df_merged[['beer_id', 'beer_name', 'rating', 'location', 'beer_consumption_per_capita']]

    print(f'Keep {len(df_merged.dropna()):,} breweries out of '
          f'{len(df_merged):,} ({100 * len(df_merged.dropna()) / len(df_merged):.2f}%)')

    df_merged.dropna()

    # group by country and keep the average rating per country
    df_merged = df_merged.groupby('location').agg(rating=('rating', 'mean'), beer_consumption_per_capita=('beer_consumption_per_capita', 'min'))

    return df_merged