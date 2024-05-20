import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.colors as mcolors

# Download necessary NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the graph
G = nx.read_edgelist('tweet_sample_v2.graph', nodetype=int)

# Load the words DataFrame
df_words = pd.read_csv('df_words_sample.csv')

# Apply Louvain algorithm to detect communities
partition = community_louvain.best_partition(G)

# Organize nodes by community
communities = defaultdict(list)
for node, community_id in partition.items():
    communities[community_id].append(node)

# Words to count occurrences for mental health
mental_health_words = ["anxiety", "depression", "stress", "happiness", "sadness"]

# Count occurrences of words in each community
word_counts = defaultdict(lambda: defaultdict(int))
for community_id, nodes in communities.items():
    for node in nodes:
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

# Define sentiment analysis functions
def get_sentiment(tweet):
    scores = sia.polarity_scores(tweet)
    if scores['compound'] > 0.05:
        return 'positive'
    elif scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Improved and more diverse ground truth data for sentiment classification
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
        'I am thrilled with the results!',
        'I feel so stressed out!',
        'The anxiety is overwhelming.',
        'Depression is taking over my life.',
        'I am so happy with my progress.',
        'I am feeling very sad and lonely.'
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
        'positive',
        'negative',
        'negative',
        'negative',
        'positive',
        'negative'
    ]
}

df_ground_truth = pd.DataFrame(ground_truth_data)

# Feature extraction for custom sentiment classifier
def extract_features(tweet):
    words = tweet.lower().split()
    return {word: True for word in words}

labeled_features = [(extract_features(row['tweet']), row['sentiment']) for _, row in df_ground_truth.iterrows()]

from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(labeled_features)

def classify_sentiment_custom(tweet):
    features = extract_features(tweet)
    return classifier.classify(features)

# Apply sentiment analysis on the words DataFrame
df_words['custom_sentiment'] = df_words['words'].apply(lambda x: classify_sentiment_custom(' '.join(eval(x))))
df_words['nltk_sentiment'] = df_words['words'].apply(lambda x: get_sentiment(' '.join(eval(x))))

print(df_words[['words', 'custom_sentiment', 'nltk_sentiment']])

# Calculate sentiment distributions
custom_sentiment_distribution = df_words['custom_sentiment'].value_counts(normalize=True)
nltk_sentiment_distribution = df_words['nltk_sentiment'].value_counts(normalize=True)

print("\nCustom Sentiment Distribution:\n", custom_sentiment_distribution)
print("\nNLTK Sentiment Distribution:\n", nltk_sentiment_distribution)

# Plot sentiment distributions
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharey=True)

axes[0].bar(custom_sentiment_distribution.index, custom_sentiment_distribution.values, color=['blue', 'green', 'red'])
axes[0].set_title('Custom Sentiment Distribution')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Proportion')

axes[1].bar(nltk_sentiment_distribution.index, nltk_sentiment_distribution.values, color=['blue', 'green', 'red'])
axes[1].set_title('NLTK Sentiment Distribution')
axes[1].set_xlabel('Sentiment')

plt.tight_layout()
plt.show()

# Load the classified and unclassified datasets
classified_df = pd.read_csv('202405192112_mod_gain_output.csv')
unclassified_df = pd.read_csv('test_tweets_unclassified.csv')

# Validate columns in classified_df
print(f"Classified DataFrame Columns:\n{classified_df.columns}")

# Ensure the correct ground truth sentiment column is used
ground_truth_column = 'community' if 'community' in classified_df.columns else 'sentiment'
ground_truth_sentiments = classified_df[ground_truth_column].tolist()

# Apply sentiment classification on unclassified tweets
predicted_sentiments = unclassified_df['text'].apply(lambda x: classify_sentiment_custom(x) if isinstance(x, str) else 'neutral').tolist()

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'text': unclassified_df['text'],
    'ground_truth': ground_truth_sentiments,
    'predicted': predicted_sentiments
})

# Calculate accuracy
accuracy = (comparison_df['ground_truth'] == comparison_df['predicted']).mean()
print("Sentiment Analysis Accuracy:", accuracy)

# Save the comparison results to a CSV file
comparison_df.to_csv('sentiment_comparison_results.csv', index=False)

# Continue with the community detection analysis
true_classes = []
for index, row in df_words.iterrows():
    tweet_words = eval(row['words'])
    tweet_class = 'None'
    for word in tweet_words:
        if word in mental_health_words:
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

df_words['sentiment'] = df_words['words'].apply(lambda x: get_sentiment(' '.join(eval(x))))
print(df_words[['words', 'sentiment']])

# Calculate overall sentiment distribution
overall_sentiment_distribution = df_words['sentiment'].value_counts(normalize=True)
print("\nOverall Sentiment Distribution:\n", overall_sentiment_distribution)

overall_sentiment_distribution.to_csv('overall_sentiment_distribution.csv')

community_sentiment_distributions = {}

for community_id, nodes in communities.items():
    community_tweets = df_words.iloc[nodes]
    community_sentiment_distribution = community_tweets['sentiment'].value_counts(normalize=True)
    community_sentiment_distributions[community_id] = community_sentiment_distribution
    print(f"\nCommunity {community_id} Sentiment Distribution:\n", community_sentiment_distribution)
    community_sentiment_distribution.to_csv(f'community_{community_id}_sentiment_distribution.csv')

df_words.to_csv('sentiment_analysis_results.csv', index=False)
