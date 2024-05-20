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

    # Draw the graph with each community having a dedicated node color
    node_colors = [partition[node] for node in G.nodes()]
    nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.tab10)
    print("Please close the graph to continue the process...")
    plt.show()

    return(partition)