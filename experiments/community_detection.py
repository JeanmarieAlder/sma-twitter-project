import networkx as nx
import community  # Louvain algorithm
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

G = nx.read_edgelist('tweet_sample_v2.graph', nodetype=int)

df_words = pd.read_csv('df_words_sample.csv')

# Apply Louvain algorithm to detect communities
partition = community.best_partition(G)

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

# Draw the graph with each community having a dedicated node color
node_colors = [partition[node] for node in G.nodes()]
nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.tab10)
plt.show()