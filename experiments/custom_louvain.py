import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.colors as mcolors
from sklearn.metrics import normalized_mutual_info_score  # Importing the NMI function

# Initialize NLTK's Sentiment Intensity Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def louvain_community_detection(G):
    # Step 1: Initialize each node to its own community
    partition = {node: node for node in G.nodes()}

    def modularity_gain(G, partition, node, community, total_weight):
        in_degree = sum(data.get('weight', 1) for neighbor, data in G[node].items() if partition[neighbor] == community)
        tot_degree = sum(data.get('weight', 1) for neighbor, data in G[node].items())
        com_degree = sum(
            data.get('weight', 1) for n in G.nodes() if partition[n] == community for neighbor, data in G[n].items())
        return in_degree - tot_degree * com_degree / (2 * total_weight)

    def one_level(G, partition, total_weight):
        modified = True
        while modified:
            modified = False
            for node in G.nodes():
                best_community = partition[node]
                best_gain = 0
                for neighbor in G[node]:
                    if partition[neighbor] != partition[node]:
                        gain = modularity_gain(G, partition, node, partition[neighbor], total_weight)
                        if gain > best_gain:
                            best_community = partition[neighbor]
                            best_gain = gain
                if best_community != partition[node]:
                    partition[node] = best_community
                    modified = True

    def aggregate_graph(G, partition):
        new_graph = nx.Graph()
        communities = defaultdict(list)
        for node, community in partition.items():
            communities[community].append(node)

        for community, nodes in communities.items():
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1:]:
                    if G.has_edge(node1, node2):
                        if new_graph.has_edge(community, community):
                            new_graph[community][community]['weight'] += G[node1][node2].get('weight', 1)
                        else:
                            new_graph.add_edge(community, community, weight=G[node1][node2].get('weight', 1))

        for (u, v, data) in G.edges(data=True):
            if partition[u] != partition[v]:
                weight = data.get('weight', 1)
                if new_graph.has_edge(partition[u], partition[v]):
                    new_graph[partition[u]][partition[v]]['weight'] += weight
                else:
                    new_graph.add_edge(partition[u], partition[v], weight=weight)

        return new_graph

    total_weight = sum(data.get('weight', 1) for u, v, data in G.edges(data=True))

    while True:
        one_level(G, partition, total_weight)
        new_graph = aggregate_graph(G, partition)
        if len(new_graph) == len(G):
            break
        G = new_graph
        partition = {node: node for node in G.nodes()}

    return partition


# Load the graph from tweet_sample_v2.graph
G = nx.read_edgelist('tweet_sample_v2.graph', nodetype=int)

# Apply custom Louvain algorithm to detect communities
partition = louvain_community_detection(G)

# Organize nodes by community
communities = defaultdict(list)
for node, community_id in partition.items():
    communities[community_id].append(node)

# Load the CSV file into df_words DataFrame
df_words = pd.read_csv('df_words_sample.csv')

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


# Create a ground truth dataset for training the custom classifier
ground_truth_data = {
    'tweet': [
        'I am feeling very happy today!',
        'This is the worst day ever.',
        'I am not sure how I feel about this.',
        'What a wonderful experience!',
        'I am extremely sad and disappointed.',
        'This is just okay, nothing special.',
        'I love the new design!',
        'I hate this so much.',
        'It is neither good nor bad.',
        'I am thrilled with the results!'
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'neutral',
        'positive'
    ]
}

# Create a DataFrame for the ground truth dataset
df_ground_truth = pd.DataFrame(ground_truth_data)


# Define a function to extract features from tweets
def extract_features(tweet):
    words = tweet.lower().split()
    return {word: True for word in words}


# Create a list of labeled features for training
labeled_features = [(extract_features(row['tweet']), row['sentiment']) for _, row in df_ground_truth.iterrows()]

# Train a Naive Bayes classifier using NLTK
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(labeled_features)


# Define a function to classify sentiment using the custom classifier
def classify_sentiment_custom(tweet):
    features = extract_features(tweet)
    return classifier.classify(features)


# Apply custom sentiment classification to each tweet in df_words
df_words['custom_sentiment'] = df_words['words'].apply(lambda x: classify_sentiment_custom(' '.join(eval(x))))

# Compare custom sentiment classification with NLTK's VADER
df_words['nltk_sentiment'] = df_words['words'].apply(lambda x: get_sentiment(' '.join(eval(x))))

# Print the results
print(df_words[['words', 'custom_sentiment', 'nltk_sentiment']])

# Calculate sentiment distribution for both classifiers
custom_sentiment_distribution = df_words['custom_sentiment'].value_counts(normalize=True)
nltk_sentiment_distribution = df_words['nltk_sentiment'].value_counts(normalize=True)

print("\nCustom Sentiment Distribution:\n", custom_sentiment_distribution)
print("\nNLTK Sentiment Distribution:\n", nltk_sentiment_distribution)

# Visualize the sentiment distributions
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharey=True)

# Custom sentiment distribution plot
axes[0].bar(custom_sentiment_distribution.index, custom_sentiment_distribution.values, color=['blue', 'green', 'red'])
axes[0].set_title('Custom Sentiment Distribution')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Proportion')

# NLTK sentiment distribution plot
axes[1].bar(nltk_sentiment_distribution.index, nltk_sentiment_distribution.values, color=['blue', 'green', 'red'])
axes[1].set_title('NLTK Sentiment Distribution')
axes[1].set_xlabel('Sentiment')

plt.tight_layout()
plt.show()

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
true_classes_int = [["anxiety", "depression", "stress", "happiness", "sadness"].index(c) if c != 'None' else -1 for c in
                    true_classes]

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
