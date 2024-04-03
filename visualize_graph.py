import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the graph from tweet.graph
G = nx.read_edgelist('tweet.graph', nodetype=int)

# Load the CSV file into df_words DataFrame
df_words = pd.read_csv('df_words.csv')


# Get a sample of tweets (nodes) from ID 0 to 500
sample_tweets = list(G.nodes())[:100]

# Create a subgraph with only the selected nodes
subgraph = G.subgraph(sample_tweets)

# Display the sample tweets
for tweet_id in sample_tweets:
    print(f"Tweet ID: {tweet_id}")
    print(df_words.at[tweet_id, 'words'])  # Print the words of the tweet
    print()

# Draw the subgraph
nx.draw(subgraph, with_labels=True)
plt.show()
