import pandas as pd
import re
import nltk
from nltk.corpus import words

# Define list of stop words
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

mental_health_words = [
    "anxiety", "depression", "stress", "happiness", "sadness"
]

# Load English words
english_words = set(words.words())

# Function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Filter stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Filter out non-English words
    text = ' '.join(word for word in text.split() if word in english_words)
    return text




# Function to detect language
# def detect_language(text):
#     lang = ''
#     print("e")
#     try:
#         lang = detect(text)
#         if lang != 'en':
#             print(lang)
#         return lang == 'en'
#     except:
#         print("Found : ", lang)
#         return False

# Load the CSV file
df = pd.read_csv('covid19_tweets.csv')

# Clean the 'text' column
df['text'] = df['text'].apply(clean_text)

# Print the head of the cleaned 'text' column
print(df['text'].head())
print(len(df))

# Filter out tweets that are not in English
# df = df[df['text'].apply(detect_language)]


# Count words
word_counts = {}
for text in df['text']:
    words = text.split()
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

# Create DataFrame with word counts
word_df = pd.DataFrame(list(word_counts.items()), columns=['word', 'count'])

# Print the total number of distinct words
print("Total number of distinct words:", len(word_df))

# Print the 10 most used words
print("\nTop 100 most used words:")
print(word_df.sort_values(by='count', ascending=False).head(100))
print("\nTop 100 least used words:")
print(word_df.sort_values(by='count', ascending=False).tail(100))


# Create a dictionary to store the counts of mental health words
mental_health_counts = {}

# Populate the mental_health_counts dictionary with counts from word_counts
for word in mental_health_words:
    mental_health_counts[word] = word_counts.get(word, 0)

# Print the count of each mental health word
for word, count in mental_health_counts.items():
    print(f"{word}: {count}")


# Filter tweets containing mental health words
filtered_df = df[df['text'].str.contains('|'.join(mental_health_words), regex=True)]

# Reset index of filtered_df
filtered_df.reset_index(drop=True, inplace=True)

# Display filtered_df
print(filtered_df["text"].head())
print(len(filtered_df))

# Extract words from the 'text' column of filtered_df
df_words = pd.DataFrame({'words': filtered_df['text'].str.split()})

# Display df_words
print(df_words.head())
print(len(df_words))

# Save df_words to a CSV file
df_words.to_csv('df_words_sample.csv', index=False)