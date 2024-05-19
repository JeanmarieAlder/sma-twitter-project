from datetime import datetime

import networkx as nx

import community as community_louvain

from utils import check_folder_existing 


def classify_tweets(df_words, df_test, partition):
    """
    Classify tweets by adding them to the graph and determining the best community for each.

    This function loads the existing graph from 'temp/tweet.graph', loops through each tweet
    in the test DataFrame `df_test`, and classifies each tweet by adding it to the graph
    and determining the best community based on modularity gain.

    Parameters:
    - df_words (pandas.DataFrame): A DataFrame containing words and community information for each tweet.
    - df_test (pandas.DataFrame): A DataFrame containing words of the tweets to be classified.
    - partition (dict): A dictionary mapping each node to its corresponding community ID.

    Returns:
    - df_test (pandas.DataFrame): A DataFrame containing the classified tweets with their assigned communities.
    """
    print("classify_tweets()")

    # Load the graph from tweet.graph
    G = nx.read_edgelist('temp/tweet.graph', nodetype=int)

    # Loop through all tweets inside df_test
    for index, tweet in df_test.iterrows():
        new_tweet_id = max(G.nodes) + 1  # Assign a new ID to the tweet
        new_tweet_words = tweet['words']

        # Create a copy of the graph
        G_copy = G.copy()

        # Add the new tweet as a node to the graph copy
        G_copy.add_node(new_tweet_id)

        # Add the new tweet to partition inside a temporary new community called "new"
        partition_copy = partition.copy()
        partition_copy[new_tweet_id] = "new"

        # Connect the new node with similar nodes
        for node_id in G_copy.nodes():
            if node_id != new_tweet_id:
                existing_tweet_words = eval(df_words.at[node_id, 'words']) if node_id in df_words.index else []
                common_words = set(new_tweet_words) & set(existing_tweet_words)
                if common_words:
                    # Add an edge between the new node and the similar node
                    G_copy.add_edge(new_tweet_id, node_id)

        # Determine the best community for the new tweet using modularity gain
        community_names = df_words['community'].unique()
        modularity_gains = {}

        for community in community_names:
            modularity_gains[community] = compute_modularity_gain(G_copy, new_tweet_id, community, partition_copy)
            # print(f"Community: {community}, Modularity Gain: {modularity_gains[community]}")

        best_community = max(modularity_gains, key=modularity_gains.get)
        # print(f"Tweet ID: {new_tweet_id}, Best Community: {best_community}")

        # Update df_test with the new tweet's community
        df_test.at[index, 'community'] = best_community

    # Print all rows of df_test with columns "words" and "community"
    # print(df_test[['words', 'community']])

    # Save df_words to a temporary CSV file inside 'output' folder.
    now = datetime.now().strftime("%Y%m%d%H%M")
    check_folder_existing("output")
    output_file = f'output/{now}_mod_gain_output.csv'
    df_test.to_csv(output_file, index=False)
    print(f"Dataframe saved as csv inside {output_file}.")

    return df_test


def compute_modularity_gain(G_copy, node, community, partition):
    """
    Compute the modularity gain of adding a node to a specific community.

    Parameters:
    - G (networkx.Graph): The graph.
    - node (int): The node to be added.
    - community (int): The community to which the node is being added.
    - partition (dict): A dictionary mapping each node to its community.

    Returns:
    - float: The modularity gain.
    """
    original_modularity = community_louvain.modularity(partition, G_copy)

    # Add the node to the specified community
    partition[node] = community
    new_modularity = community_louvain.modularity(partition, G_copy)

    return new_modularity - original_modularity