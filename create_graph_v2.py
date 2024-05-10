import pandas as pd

# Load the CSV file into df_words DataFrame
df_words = pd.read_csv('df_words_sample.csv')

# Display the DataFrame
print(df_words)

# Initialize an empty list to store edges
edges = []

# Mental health words to exclude from creating edges
mental_health_words = {"anxiety", "depression", "stress", "happiness", "sadness"}

# Iterate over each tweet
for i in range(len(df_words)):
    # Split words of the current tweet
    words1 = set(eval(df_words.at[i, 'words']))

    # Remove mental health words from words1
    words1 = words1 - mental_health_words
    
    # Compare with subsequent tweets
    for j in range(i+1, len(df_words)):
        # Split words of the other tweet
        words2 = set(eval(df_words.at[j, 'words']))
        
        # Find common words
        common_words = words1.intersection(words2)

        # If common words exist, add an edge
        if common_words:
            edges.append((i, j))

# Write edges to a text file
with open('tweet_sample_v3.graph', 'w') as file:
    for edge in edges:
        file.write(f"{edge[0]} {edge[1]}\n")