import pandas as pd

def generate_graph(df_words):
    """
    Generate a graph based on common words between tweets and save it to a file.

    This function generates a graph based on the common words between tweets.
    It takes a DataFrame `df_words` containing information about words in each tweet.
    The function iterates over each tweet, compares it with subsequent tweets to find
    common words, and adds edges to represent connections between tweets.
    The generated graph is saved to a file named 'tweet.graph' in the 'temp' directory.

    Parameters:
    - df_words (pandas.DataFrame): A DataFrame containing information about words in each tweet.
    Returns:
    - df_words: Same dataframe, with correct "words" column.
    """
    print("generate_graph()")

    # Load the CSV file into df_words DataFrame
    # TODO: understand why it won't work with df_words from previous step instead of re-reading the csv file...
    df_words = pd.read_csv('temp/df_words.csv')


    # Initialize an empty list to store edges
    edges = []
    # Iterate over each tweet
    for i in range(len(df_words)):
        # Split words of the current tweet
        words1 = set(eval(df_words.at[i, 'words']))
        
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
    with open('temp/tweet.graph', 'w') as file:
        for edge in edges:
            file.write(f"{edge[0]} {edge[1]}\n")

    return df_words
