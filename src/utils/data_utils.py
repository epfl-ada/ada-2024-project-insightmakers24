import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import string

# Import useful nltk packages
nltk.download("punkt")
nltk.download("stopwords")

# Constants
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def convert_name_to_stemmed_keywords(name):
    """
    Take a beer name and convert it to it's stemmed word tokens

    Parameters
    ----------
    name: The name of a beer

    Returns
    -------
    List of stemmed word tokens
    """
    # Lowercase and remove punctuation
    processed_name = name.lower()
    processed_name = processed_name.translate(str.maketrans('', '', string.punctuation))

    # Split by word and remove stop words
    tokens = word_tokenize(processed_name)
    tokens = [word for word in tokens if word not in stop_words]

    # Stem words to handle similar forms of words (tense, plural, ...)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # Return these stemmed words as the keywords of the beer name
    return stemmed_tokens


def load_name_keyword_data(path):
    """
    Load the relevant data for the name keyword analysis

    Parameters
    ----------
    path: The path to the data folder

    Returns
    -------
    The relevant dataframe for the name keyword analysis
    """
    beers = pd.read_csv(os.path.join(path, "beers.csv"))

    # Keep only significant beers and columns (at least 10 reviews)
    relevant_beers = beers[beers["nbr_ratings"] >= 10][["beer_id", "beer_name", "nbr_ratings", "avg"]]

    return relevant_beers


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
    Load and do basic cleaning for the country bias analysis

    Parameters
    ----------
    path: The path to the analysis files

    Returns
    -------
    Clean dataset used for the analysis
    """

    # Load rating, user and breweries infos
    df_ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
    df_users = pd.read_csv(os.path.join(path, 'users.csv'))
    df_breweries = pd.read_csv(os.path.join(path, 'breweries.csv'))

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


def load_data_first_rating(data_path='data/BeerAdvocate/ratings.csv', min_count=1000, first=True):
    """
    Return the dataframe used for first vs other rating influence

    Parameters
    ----------
    data_path: The path to the analysis files
    min_count: The minimum of review a beer need so that its rating is considered valid
    first: If True, compare the first rating, if false take the last one (only to show differences)

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


def load_beer_consumption_dataframe(beer_consumption_path='data/BeerConsumption.csv', breweries_path='data/BeerAdvocate/breweries.csv', rating_path='data/BeerAdvocate/ratings.csv'):
    """
    Return the dataframe used for the beer consumption analysis

    Parameters
    ----------
    beer_consumption_path: The path to the beer consumption file
    breweries_path: The breweries dataset path
    rating_path: The rating dataset path

    Returns
    -------
    The dataframe used for the analysis
    """

    # load the 3 differents datasets
    beerConsumption_df = pd.read_csv(beer_consumption_path)
    breweries_df = pd.read_csv(breweries_path)
    ratings_df = pd.read_csv(rating_path)

    # merge the 3 dataset by beer_id and location
    merge_df = ratings_df.merge(breweries_df, left_on='beer_id', right_index=True)

    merge_df['location'] = merge_df['location'].str.strip().str.lower()
    beerConsumption_df['country'] = beerConsumption_df['country'].str.strip().str.lower()

    merged_df = merge_df.merge(beerConsumption_df, left_on='location', right_on='country')

    # merge 2020 dans 2021 beer conumption to get an average
    merged_df['beer_consumption_per_capita'] = (merged_df['BeerConsumptionPerCapitakg2021'] + merged_df['BeerConsumptionPerCapitakg2021'])/2

    # keep usefull columns
    merged_df = merged_df[['beer_id', 'beer_name', 'rating', 'location', 'beer_consumption_per_capita']]

    merged_df.dropna()

    # group by country and keep the average rating per country
    merged_df = merged_df.groupby('location').agg(rating=('rating', 'mean'), beer_consumption_per_capita=('beer_consumption_per_capita', 'min'))

    return merged_df


def get_sentiment_textblob(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity

    # Map the sentiment in 1-5 range
    return ((score + 1) * 2 )+ 1

def get_sentiment_vader(text, sia):
    # Map the sentiment in 1-5 range
    return ((sia.polarity_scores(text)['compound'] + 1) * 2 )+ 1

def load_dataframe_sentiment(DATA_PATH='data/BeerAdvocate/ratings.csv'):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    # load the dataset
    ratings_df = pd.read_csv(DATA_PATH)

    # calculate the mean rating per beer
    rating_per_beer = ratings_df.groupby('beer_id').agg(beer_name=('beer_name', 'min'),  rating=('rating', 'mean')).sort_values(by='rating')

    # keep only usefull columns
    rating_per_beer = rating_per_beer.dropna()

    # add column for sentiment in beers name
    rating_per_beer['name_sentiment_textblob']  = rating_per_beer['beer_name'].apply(get_sentiment_textblob)

    rating_per_beer['name_sentiment_vader']  = rating_per_beer['beer_name'].apply(lambda x:get_sentiment_vader(x, sia))

    return rating_per_beer