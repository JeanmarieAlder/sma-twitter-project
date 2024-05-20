import networkx as nx
import community as community_louvain  # Louvain algorithm
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.colors as mcolors

# Initialize NLTK's Sentiment Intensity Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the graph from tweet_sample_v2.graph
G = nx.read_edgelist('tweet_sample_v2.graph', nodetype=int)

# Load the CSV file into df_words DataFrame
df_words = pd.read_csv('df_words_sample.csv')

# Apply Louvain algorithm to detect communities
partition = community_louvain.best_partition(G)

# Organize nodes by community
communities = defaultdict(list)
for node, community_id in partition.items():
    communities[community_id].append(node)

# Words to count occurrences for
mental_health_words = ["anxiety", "depression", "stress", "happiness", "sadness"]

# Count occurrences of words in each community
word_counts = defaultdict(lambda: defaultdict(int))
for community_id, nodes in communities.items():
    for node in nodes:
        # Get the tweet words associated with the node
        tweet_words = eval(df_words.at[node, 'words'])
        for word in tweet_words:
            if word in mental_health_words:
                word_counts[community_id][word] += 1

# Print information about communities and word counts
for community_id, nodes in communities.items():
    print(f"Community {community_id}:")
    print(f"Number of nodes: {len(nodes)}")
    print("Nodes:", nodes)
    print("Word counts:", word_counts[community_id])
    print()

# Sentiment analysis
def get_sentiment(tweet):
    scores = sia.polarity_scores(tweet)
    if scores['compound'] > 0.05:
        return 'positive'
    elif scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Get the true classes (known classes) based on the words ["anxiety", "depression", "stress", "happiness", "sadness"]
true_classes = []
for index, row in df_words.iterrows():
    tweet_words = eval(row['words'])
    tweet_class = 'None'
    for word in tweet_words:
        if word in ["anxiety", "depression", "stress", "happiness", "sadness"]:
            tweet_class = word
            break
    true_classes.append(tweet_class)

# Convert the true classes to integers
true_classes_int = [["anxiety", "depression", "stress", "happiness", "sadness"].index(c) if c != 'None' else -1 for c in true_classes]

# Convert the partition to a list of labels
partition_labels = [partition[node] for node in G.nodes()]

# Filter out nodes without associated words
valid_nodes = [node for node in G.nodes() if partition[node] != -1]
true_classes_int_filtered = [true_classes_int[node] for node in valid_nodes]
partition_labels_filtered = [partition[node] for node in valid_nodes]

# Compute the NMI
nmi = normalized_mutual_info_score(true_classes_int_filtered, partition_labels_filtered)

print("Normalized Mutual Information (NMI):", nmi)

# Define custom colormap function
def get_cmap_color(value, cmap_name='coolwarm', vmin=None, vmax=None):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(value))

# Draw the graph with each community having a dedicated node color
unique_communities = list(set(partition.values()))
vmin = min(unique_communities)
vmax = max(unique_communities)
node_colors = [get_cmap_color(partition[node], vmin=vmin, vmax=vmax) for node in G.nodes()]

plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G, seed=42)  # Use a layout for better visualization
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=100, font_size=10)
plt.title("Tweet Graph with Community Detection using Coolwarm Colormap")
plt.show()

# Sentiment Analysis for each tweet
df_words['sentiment'] = df_words['words'].apply(lambda x: get_sentiment(' '.join(eval(x))))
print(df_words[['words', 'sentiment']])

# Calculate overall sentiment distribution
overall_sentiment_distribution = df_words['sentiment'].value_counts(normalize=True)
print("\nOverall Sentiment Distribution:\n", overall_sentiment_distribution)

# Save overall sentiment distribution
overall_sentiment_distribution.to_csv('overall_sentiment_distribution.csv')

# Calculate sentiment distribution within each community and save results
community_sentiment_distributions = {}

for community_id, nodes in communities.items():
    community_tweets = df_words.iloc[nodes]
    community_sentiment_distribution = community_tweets['sentiment'].value_counts(normalize=True)
    community_sentiment_distributions[community_id] = community_sentiment_distribution
    print(f"\nCommunity {community_id} Sentiment Distribution:\n", community_sentiment_distribution)
    community_sentiment_distribution.to_csv(f'community_{community_id}_sentiment_distribution.csv')

# Save the sentiment analysis results to a CSV file
df_words.to_csv('sentiment_analysis_results.csv', index=False)