import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the graph from tweet.graph
G = nx.read_edgelist('tweet_sample.graph', nodetype=int)

# Load the CSV file into df_words DataFrame
df_words = pd.read_csv('df_words_sample.csv')

mental_health_words = [
    "anxiety", "depression", "stress", "happiness", "sadness"
]
colors = [
    "red", "green", "blue", "orange", "grey"
]

# Get a sample of tweets (nodes) from ID 0 to 500
sample_tweets = list(G.nodes())

# Create a subgraph with only the selected nodes
subgraph = G.subgraph(sample_tweets)

# print(df_words["words"].iloc[1:269])

# Assign colors to tweets based on mental_health_words
node_colors = []
for node in subgraph.nodes():
    tweet_words = eval(df_words.at[node, 'words'])
    tweet_color = 'pink' if len(tweet_words) >= 2 else 'black'  # Default color for tweets with less than two words
    for word in tweet_words:
        if word in mental_health_words:
            index = mental_health_words.index(word)
            tweet_color = colors[index]
            break
    node_colors.append(tweet_color)

# Filter out nodes that are pink or black
filtered_nodes = [node for node, color in zip(subgraph.nodes(), node_colors) if color not in ['pink', 'black']]
filtered_subgraph = subgraph.subgraph(filtered_nodes)
filtered_node_colors = [color for color in node_colors if color not in ['pink', 'black']]

# Draw the filtered subgraph with node colors
nx.draw(filtered_subgraph, with_labels=True, node_color=filtered_node_colors)

# Save the subgraph to a new graph file
nx.write_edgelist(filtered_subgraph, "tweet_sample_v2.graph")

# Save an image of the graph with higher resolution
plt.savefig("graph.jpg", dpi=1200)

plt.show()


# # Draw the subgraph with node colors
# nx.draw(subgraph, with_labels=True, node_color=node_colors)
# plt.show()


# # Display the sample tweets
# for tweet_id in sample_tweets:
#     print(f"Tweet ID: {tweet_id}")
#     print(df_words.at[tweet_id, 'words'])  # Print the words of the tweet
#     print()

# # Draw the subgraph
# nx.draw(subgraph, with_labels=True)
# plt.show()
