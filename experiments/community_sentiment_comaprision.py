import networkx as nx
import community as community_louvain
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.colors as mcolors

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

G = nx.read_edgelist('tweet_sample_v2.graph', nodetype=int)

df_words = pd.read_csv('df_words_sample.csv')

partition = community_louvain.best_partition(G)

communities = defaultdict(list)
for node, community_id in partition.items():
    communities[community_id].append(node)

mental_health_words = ["anxiety", "depression", "stress", "happiness", "sadness"]

word_counts = defaultdict(lambda: defaultdict(int))
for community_id, nodes in communities.items():
    for node in nodes:
        tweet_words = eval(df_words.at[node, 'words'])
        for word in tweet_words:
            if word in mental_health_words:
                word_counts[community_id][word] += 1

for community_id, nodes in communities.items():
    print(f"Community {community_id}:")
    print(f"Number of nodes: {len(nodes)}")
    print("Nodes:", nodes)
    print("Word counts:", word_counts[community_id])
    print()

def get_sentiment(tweet):
    if not isinstance(tweet, str):
        tweet = ""
    scores = sia.polarity_scores(tweet)
    if scores['compound'] > 0.05:
        return 'positive'
    elif scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'

classified_df = pd.read_csv('202405192112_mod_gain_output.csv')

community_to_sentiment = {
    'happy': 'positive',
    'sad': 'negative',
    'hope': 'neutral'
}
classified_df['predicted_sentiment'] = classified_df['community'].map(community_to_sentiment)
classified_df['nltk_sentiment'] = classified_df['text'].apply(get_sentiment)

accuracy = accuracy_score(classified_df['nltk_sentiment'], classified_df['predicted_sentiment'])
print("Sentiment Analysis Accuracy:", accuracy)

print(classified_df[['text', 'community', 'predicted_sentiment', 'nltk_sentiment']].head())

nltk_sentiment_distribution = classified_df['nltk_sentiment'].value_counts(normalize=True)
predicted_sentiment_distribution = classified_df['predicted_sentiment'].value_counts(normalize=True)

print("\nNLTK Sentiment Distribution:\n", nltk_sentiment_distribution)
print("\nPredicted Sentiment Distribution:\n", predicted_sentiment_distribution)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharey=True)

axes[0].bar(nltk_sentiment_distribution.index, nltk_sentiment_distribution.values, color=['blue', 'green', 'red'])
axes[0].set_title('NLTK Sentiment Distribution')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Proportion')

axes[1].bar(predicted_sentiment_distribution.index, predicted_sentiment_distribution.values, color=['blue', 'green', 'red'])
axes[1].set_title('Predicted Sentiment Distribution')
axes[1].set_xlabel('Sentiment')

plt.tight_layout()
plt.show()

df_words['nltk_sentiment'] = df_words['words'].apply(lambda x: get_sentiment(' '.join(eval(x))))
print(df_words[['words', 'nltk_sentiment']].head())

true_classes = []
for index, row in df_words.iterrows():
    tweet_words = eval(row['words'])
    tweet_class = 'None'
    for word in mental_health_words:
        if word in tweet_words:
            tweet_class = word
            break
    true_classes.append(tweet_class)

true_classes_int = [mental_health_words.index(c) if c != 'None' else -1 for c in true_classes]
partition_labels = [partition[node] for node in G.nodes()]
valid_nodes = [node for node in G.nodes() if partition[node] != -1]
true_classes_int_filtered = [true_classes_int[node] for node in valid_nodes]
partition_labels_filtered = [partition[node] for node in valid_nodes]

nmi = normalized_mutual_info_score(true_classes_int_filtered, partition_labels_filtered)
print("Normalized Mutual Information (NMI):", nmi)

def get_cmap_color(value, cmap_name='coolwarm', vmin=None, vmax=None):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(value))

unique_communities = list(set(partition.values()))
vmin = min(unique_communities)
vmax = max(unique_communities)
node_colors = [get_cmap_color(partition[node], vmin=vmin, vmax=vmax) for node in G.nodes()]

plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=100, font_size=10)
plt.title("Tweet Graph with Community Detection using Coolwarm Colormap")
plt.show()

# Calculate overall sentiment distribution
overall_sentiment_distribution = df_words['nltk_sentiment'].value_counts(normalize=True)
print("\nOverall Sentiment Distribution:\n", overall_sentiment_distribution)

overall_sentiment_distribution.to_csv('overall_sentiment_distribution.csv')

community_sentiment_distributions = {}

for community_id, nodes in communities.items():
    community_tweets = df_words.iloc[nodes]
    community_sentiment_distribution = community_tweets['nltk_sentiment'].value_counts(normalize=True)
    community_sentiment_distributions[community_id] = community_sentiment_distribution
    print(f"\nCommunity {community_id} Sentiment Distribution:\n", community_sentiment_distribution)
    community_sentiment_distribution.to_csv(f'community_{community_id}_sentiment_distribution.csv')

df_words.to_csv('sentiment_analysis_results.csv', index=False)
