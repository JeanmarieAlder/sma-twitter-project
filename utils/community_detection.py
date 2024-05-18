from collections import defaultdict
import community  # Louvain algorithm
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def generate_communities(df_words):
    """
    Generate communities in the graph based on the Louvain algorithm.

    This function applies the Louvain algorithm to detect communities in the graph
    represented by the edges loaded from the 'temp/tweet.graph' file. It visualizes
    the communities by drawing the graph, with nodes colored according to their
    community assignment.

    Parameters:
    - df_words (pandas.DataFrame): A DataFrame containing information about words
      in each tweet.

    Returns:
    - partition (dict): A dictionary mapping each node to its corresponding community ID.
    """
    print("generate_communitites()")

    # Load the graph from tweet.graph
    G = nx.read_edgelist('temp/tweet.graph', nodetype=int)

    # Apply Louvain algorithm to detect communities
    # TODO: Recreate the Louvain algorithm by hand.
    partition = community.best_partition(G, random_state=42)

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

    # Determine the most frequent mental health word for each community
    community_names = {}
    for community_id, counts in word_counts.items():
        if counts:
            community_names[community_id] = max(counts, key=counts.get)
        else:
            community_names[community_id] = "None"

    # Add community column to df_words, handle nodes not in partition
    def get_community(node):
        if node in partition:
            community_id = partition[node]
            return community_names.get(community_id, "None")
        else:
            return "None"

    df_words['community'] = df_words.index.map(get_community)

    print(df_words[['words', 'community']])

    # Draw the graph with each community having a dedicated node color
    node_colors = [partition[node] for node in G.nodes()]
    nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.tab10, width=0.2)
    print("Please close the graph to continue the process...")
    plt.show()

    return(partition, df_words)