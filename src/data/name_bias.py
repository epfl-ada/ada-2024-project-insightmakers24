import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

# Constants
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

def convert_name_to_stemmed_keywords(name, stemmer=None):
    """
    Take a beer name and convert it to it's stemmed word tokens

    Args:
        name (str): The name of a beer
        stemmer (Optional[PorterStemmer]): One of PorterStemmer or None. Processes tokens of the name.

    Returns
        List of stemmed word tokens
    """
    # Lowercase and replace punctuation by spaces (so that A/B or A-B is tokenized as A B instead of AB)
    processed_name = name.lower()
    processed_name = processed_name.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Split by word and remove stop words
    tokens = word_tokenize(processed_name)
    tokens = [word for word in tokens if word not in STOP_WORDS]

    if isinstance(stemmer, PorterStemmer):
        # Stem words to handle similar forms of words (tense, plural, ...)
        return [stemmer.stem(word) for word in tokens]
    else:
        # Return the unmodified tokens
        return tokens


def has_type_in_name(name_keywords, type_keywords):
    """Check if any type keyword is in the name keywords."""
    return any(keyword in name_keywords for keyword in type_keywords)

def remove_type_keywords(name_keywords, type_keywords):
    """Remove type keywords from name keywords."""
    return [keyword for keyword in name_keywords if keyword not in type_keywords]

def preprocess_type_in_name_analysis(beers):
    """
    Preprocess a dataset of beers to prepare it for analysis of keywords in beer names.
    
    This function filters the dataset to include only beers with a minimum number of ratings,
    extracts keywords from beer names, brewery names, and styles, and identifies whether 
    specific keywords (e.g., style or brewery names) appear in the beer name. It also generates 
    cleaned versions of beer name keywords by removing style and brewery terms.
    
    Args:
        beers (DataFrame): A pandas DataFrame containing beer data with columns 
            'beer_name', 'brewery_name', 'style', 'nbr_ratings', 'avg', 'abv'.
    
    Returns:
        DataFrame: A preprocessed DataFrame with additional columns for keyword analysis.
    """
    # Keep only beers with at least 2 ratings
    beers = beers[beers["nbr_ratings"] >= 2]

    # Keep only relevant columns
    beers = beers[["beer_name", "brewery_name", "style", "nbr_ratings", "avg", "abv"]]

    # Convert name, brewery and type to keywords for analysis
    beers["name_keywords"] = beers["beer_name"].map(lambda x: convert_name_to_stemmed_keywords(x, STEMMER))
    beers["brewery_keywords"] = beers["brewery_name"].map(lambda x: convert_name_to_stemmed_keywords(x, STEMMER))
    beers["style_keywords"] = beers["style"].map(lambda x: convert_name_to_stemmed_keywords(x, STEMMER))

    # Specify if a beer has brewery/type keyword in its name
    beers["has_style_in_name"] = beers.apply(
        lambda row: has_type_in_name(row["name_keywords"], row["style_keywords"]), axis=1
    )
    beers["no_style_name"] = beers.apply(
        lambda row: remove_type_keywords(row["name_keywords"], row["style_keywords"]), axis=1
    )
    beers["has_brewery_in_name"] = beers.apply(
        lambda row: has_type_in_name(row["name_keywords"], row["brewery_keywords"]), axis=1
    )
    beers["no_brewery_name"] = beers.apply(
        lambda row: remove_type_keywords(row["name_keywords"], row["brewery_keywords"]), axis=1
    )
    beers["no_style_no_brewery_name"] = beers.apply(
        lambda row: remove_type_keywords(row["no_style_name"], row["brewery_keywords"]), axis=1
    )

    return beers

def per_style_presence_in_name(df):
    grouped = df.groupby("style")

    total_beers_per_style = grouped["beer_name"].count()
    beers_with_style_in_name = grouped["has_style_in_name"].sum()
    proportion_with_style_in_name = beers_with_style_in_name / total_beers_per_style
    beers_with_brewery_in_name = grouped["has_brewery_in_name"].sum()
    proportion_with_brewery_in_name = beers_with_brewery_in_name / total_beers_per_style
    
    style_proportion = pd.DataFrame({
        "total_beers": total_beers_per_style,
        "beers_with_style_in_name": beers_with_style_in_name,
        "proportion_with_style_in_name": proportion_with_style_in_name,
        "beers_with_brewery_in_name": beers_with_brewery_in_name,
        "proportion_with_brewery_in_name": proportion_with_brewery_in_name,
    }).reset_index()

    return style_proportion