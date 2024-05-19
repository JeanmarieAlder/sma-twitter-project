import networkx as nx

from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

from utils.constants import mental_health_words


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


    # Get the true classes (known classes) based on the mental health words
    true_classes = []
    for index, row in df_words.iterrows():
        tweet_words = eval(row['words'])
        tweet_class = 'None'
        for word in tweet_words:
            if word in mental_health_words:
                tweet_class = word
                break
        true_classes.append(tweet_class)

    # Convert the true classes to integers
    true_classes_int = [mental_health_words.index(c) if c != 'None' else -1 for c in true_classes]

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


def analyze_unclassified_tweets(partition, df_words):
    """
    Analyze and print tweets that do not contain mental health words.

    This function loads the graph from 'temp/tweet.graph', identifies nodes (tweets)
    that do not contain any of the specified mental health words, and prints their
    IDs, words, and community IDs.

    Parameters:
    - partition (dict): A dictionary mapping each node to its corresponding community ID.
    - df_words (pandas.DataFrame): A DataFrame containing information about words in each tweet.
    """
    print("analyze_unclassified_tweets()")

    # Load the graph from tweet.graph
    G = nx.read_edgelist('temp/tweet.graph', nodetype=int)

    # Organize nodes by community
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)


    # Initialize dictionaries to count total and unclassified tweets in each community
    total_tweet_counts = defaultdict(int)
    unclassified_tweet_counts = defaultdict(int)

    # Identify and print tweets that do not contain mental health words
    for node in G.nodes():
        tweet_words = eval(df_words.at[node, 'words'])
        community = df_words.at[node, 'community']
        total_tweet_counts[community] += 1
        if not any(word in tweet_words for word in mental_health_words):
            unclassified_tweet_counts[community] += 1
            print(f"ID: {node}, Words: {tweet_words}, Community: {community}")

    # Print the total number of tweets inside each community
    print("\nTotal number of tweets in each community:")
    for community, count in total_tweet_counts.items():
        print(f"Community {community}: {count} tweets")

    # Print the number of unclassified tweets inside each community
    print("\nNumber of tweets without mental health words in each community:")
    for community, count in unclassified_tweet_counts.items():
        print(f"Community {community}: {count} tweets")