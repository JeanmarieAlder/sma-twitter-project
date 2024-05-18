import os
import pandas as pd

from nltk import download as nltk_download
from nltk.corpus import words
from re import sub as re_sub


# Stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
              'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
              'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
              'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
              'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
              'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
              'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
              'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
              "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
              'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
              'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# Mental health related words
# TODO: Choose final mental health words array. Currently smaller for faster tests.
# mental_health_words = [
#     "anxiety", "depression", "stress", "happiness", "sadness", "therapy", "counseling", 
#     "support", "resilience", "coping", "wellness", "recovery", "selfcare", "empowerment", 
#     "hope", "optimism", "isolation", "loneliness", "selfesteem", "motivation", "mindfulness", 
#     "meditation", "psychologist", "psychiatrist", "trauma", "ptsd", "bipolar", "schizophrenia", 
#     "ocd", "panic", "medication", "selfharm", "suicidal", "supportgroup", "socialanxiety", 
#     "phobia", "trigger", "recovery", "adjustment", "acceptance", "cognitivebehavioraltherapy", 
#     "mentalillness", "mentalhealth", "wellbeing", "copingmechanism", "emotionalregulation", 
#     "selfawareness", "copingstrategies", "socialsupport"
# ]
mental_health_words = [
    "anxiety", "depression", "stress", "happiness", "sadness"
]

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


def preprocess_tweets(df):
    print("preprocess_tweets()")

    # Clean the 'text' column
    df['text'] = df['text'].apply(clean_text)

    # Filter tweets containing mental health words
    filtered_df = df[df['text'].str.contains('|'.join(mental_health_words), regex=True)]

    # Get tweets that do not contain mental health words
    non_mental_health_df = df[~df['text'].str.contains('|'.join(mental_health_words), regex=True)]

    # Sample 200 tweets from non_mental_health_df
    non_mental_health_sample = non_mental_health_df.sample(n=200, random_state=69)

    # Reset index of filtered_df and non mental health tweets
    filtered_df.reset_index(drop=True, inplace=True)
    non_mental_health_sample.reset_index(drop=True, inplace=True)

    # Extract words from the 'text' column of filtered_df
    df_words = pd.DataFrame({'words': filtered_df['text'].str.split()})
    df_additional_words = pd.DataFrame({'words': non_mental_health_sample['text'].str.split()})


    # Add the 'user_location' column to df_words
    df_words['user_location'] = filtered_df['user_location']
    df_additional_words['user_location'] = non_mental_health_sample['user_location']

    # Save df_words to a temporary CSV file inside 'temp' folder.
    df_words.to_csv('temp/df_words.csv', index=False)
    df_additional_words.to_csv('temp/df_aditional_words.csv', index=False)

    return df_words, df_additional_words
