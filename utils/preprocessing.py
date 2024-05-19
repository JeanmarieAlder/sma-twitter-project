import emoji
import pandas as pd

from nltk import download as nltk_download
from nltk.corpus import words
from re import sub as re_sub
from re import findall as re_findall
from utils.constants import mental_health_words, stop_words






# Download english words to filter tweets later.
nltk_download('words')
# Load English words
english_words = set(words.words())


def clean_text(text):
    """
    Clean and preprocess the input text by performing the following steps:
    1. Convert text to lowercase.
    2. Remove mentions (@username).
    3. Remove special characters and punctuations.
    4. Remove extra whitespaces.
    5. Filter stop words.
    6. Filter out non-English words.

    Parameters:
    - text (str): The input text to be cleaned.

    Returns:
    - str: The cleaned text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove mentions (@username)
    text = re_sub(r'@\w+', '', text)
    # Remove special characters and punctuations
    text = re_sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re_sub(r'\s+', ' ', text)
    # Filter stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Filter out non-English words
    text = ' '.join(word for word in text.split() if word in english_words)
    return text


def extract_emojis(text):
    """
    Extract emojis from the input text.

    Parameters:
    - text (str): Input text containing emojis.

    Returns:
    - list: List of emojis extracted from the text.
    """
    emoji_list = []
    for e in emoji.EMOJI_DATA:
        if e in text and e not in emoji_list:
            emoji_list.append(e)
    return emoji_list



def get_non_mental_health_tweets(df):
    """
    Filter non-mental health tweets from the DataFrame that are long enough.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing tweet information.

    Returns:
    - pandas.DataFrame: DataFrame containing non-mental health tweets with at least 4 words.
    """
    return df[
        (~df['words'].apply(lambda words: any(word in words for word in mental_health_words))) & 
        (df['words'].apply(len) >= 4)
    ]


def get_sample_tweets(df, nsize):
    return df.sample(n=nsize, random_state=69)


def preprocess_tweets(df):
    """
    Preprocess tweets DataFrame by cleaning text, extracting emojis, and filtering tweets.
    
    Parameters:
    - df (pandas.DataFrame): Input DataFrame containing tweet data.

    Returns:
    - df (pandas.DataFrame): Processed input DataFrame (slightly modified with new columns).
    - df_words (pandas.DataFrame): DataFrame containing tweets for the initial community detection.
    """
    print("preprocess_tweets()")

    # Extract emojis from the 'text' column
    df['emojis'] = df['text'].apply(extract_emojis)

    # Clean the 'text' column
    df['text'] = df['text'].apply(clean_text)

    # Extract words from the 'text' column
    df['words'] = df['text'].str.split()

    # Concatenate emojis into the 'words' column
    df['words'] = df.apply(lambda row: row['words'] + row['emojis'], axis=1)
    
    # Filter tweets containing exact matches to mental health words
    filtered_df = df[df['words'].apply(lambda words: any(word in words for word in mental_health_words))]

    # Get tweets that do not contain mental health words and have at least 4 words
    non_mental_health_df = get_non_mental_health_tweets(df)

    # Sample n tweets from non_mental_health_df
    non_mental_health_sample = get_sample_tweets(non_mental_health_df, 0)

    # Concatenate the filtered_df with the non_mental_health_sample
    combined_df = pd.concat([filtered_df, non_mental_health_sample], ignore_index=True)

    # Reset index of the combined DataFrame
    combined_df.reset_index(drop=True, inplace=True)

    # Extract words from the 'text' column of combined_df
    df_words = pd.DataFrame({'words': combined_df['text'].str.split()})

    # Add the 'user_location' column to df_words
    df_words['user_location'] = combined_df['user_location']
    
    # Save df_words to a temporary CSV file inside 'temp' folder.
    df_words.to_csv('temp/df_words.csv', index=False)

    # Count words
    word_counts = {}
    for text in df['text']:
        words = text.split()
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    # Create a dictionary to store the counts of mental health words
    mental_health_counts = {}

    # Populate the mental_health_counts dictionary with counts from word_counts
    for word in mental_health_words:
        mental_health_counts[word] = word_counts.get(word, 0)

    # Print the count of each mental health word
    print("Count of mental health words: ")
    for word, count in mental_health_counts.items():
        print(f"{word}: {count}")

    return df, df_words


def load_tweets():
    """
    Load tweets from the input CSV file.

    Returns:
    - pandas.DataFrame: DataFrame containing tweet information.
    """
    df = pd.DataFrame([])
    try:
        # Load the CSV file
        df = pd.read_csv('covid19_tweets.csv')
        print("Successfully loaded Tweets. Sample:")
        print(df.head())
    except Exception as e:
        print(e)
        print("Couldn't load tweets. Make sure to download them and place them in the root folder. More information about this in the readme file.")
        raise SystemExit(1)

    return df