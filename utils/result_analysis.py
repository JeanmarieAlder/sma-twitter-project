import networkx as nx

from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score


def compute_nmi(partition, df_words):
    """
    Compute the Normalized Mutual Information (NMI) between true classes and detected communities.

    This function computes the NMI between the true classes of tweets (based on the occurrence
    of specific words) and the communities detected using the Louvain algorithm. It loads the
    graph from the 'temp/tweet.graph' file, organizes nodes by community, counts occurrences
    of specific words in each community, and prints information about communities and word counts.
    It then computes the NMI between the true classes and detected communities.

    Parameters:
    - partition (dict): A dictionary mapping each node to its corresponding community ID.
    - df_words (pandas.DataFrame): A DataFrame containing information about words in each tweet.

    Returns:
    - float: The Normalized Mutual Information (NMI) between true classes and detected communities.
    """
    # Load the graph from tweet.graph
    G = nx.read_edgelist('temp/tweet.graph', nodetype=int)

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

    return nmi