from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx


def generate_graph(df_words):
    """
    Generate a graph based on common words between tweets and save it to a file.

    This function generates a graph based on the common words between tweets.
    It takes a DataFrame `df_words` containing information about words in each tweet.
    The function iterates over each tweet, compares it with subsequent tweets to find
    common words, and adds edges to represent connections between tweets.
    The generated graph is saved to a file named 'tweet.graph' in the 'temp' directory.

    Parameters:
    - df_words (pandas.DataFrame): A DataFrame containing information about words in each tweet.
    Returns:
    - df_words: Same dataframe, with correct "words" column.
    """
    print("generate_graph()")

    # Load the CSV file into df_words DataFrame
    # TODO: understand why it won't work with df_words from previous step instead of re-reading the csv file...
    df_words = pd.read_csv('temp/df_words.csv')


    # Initialize an empty list to store edges
    edges = []
    # Iterate over each tweet
    for i in range(len(df_words)):
        # Split words of the current tweet
        words1 = set(eval(df_words.at[i, 'words']))
        
        # Compare with subsequent tweets
        for j in range(i+1, len(df_words)):
            # Split words of the other tweet
            words2 = set(eval(df_words.at[j, 'words']))
            
            # Find common words
            common_words = words1.intersection(words2)
            
            # If common words exist, add an edge
            if common_words:
                edges.append((i, j))

    # Write edges to a text file
    with open('temp/tweet.graph', 'w') as file:
        for edge in edges:
            file.write(f"{edge[0]} {edge[1]}\n")

    return df_words


def add_node_to_graph(df_words, additional_word):

    # Load the graph from tweet.graph
    G = nx.read_edgelist('temp/tweet.graph', nodetype=int)

    # Extract information from the new tweet
    new_tweet_id = len(G)  # Assign a new ID for the tweet node
    new_tweet_words = additional_word['words']  # Extract words from the new tweet

    # Add the new node to the graph
    G.add_node(new_tweet_id, words=new_tweet_words)

    # Connect the new node with similar nodes
    print(new_tweet_words)
    for node_id, node_data in df_words.iterrows():
        if node_id != new_tweet_id:
            common_words = set(new_tweet_words) & set(node_data['words'])
            if common_words:
                # Add an edge between the new node and the similar node
                G.add_edge(new_tweet_id, node_id)
                print(f"Found a similar word: {common_words}")

    # Save the updated graph to 'temp/new-tweet.graph'
    nx.write_edgelist(G, 'temp/new-tweet.graph')

    draw_graph_with_communities(G, df_words)

    input("stop")


def draw_graph_with_communities(G, df_words):
    # Extract community information from df_words
    community_map = {node: community for node, community in zip(df_words.index, df_words['community'])}

    # Create a mapping of community names to colors
    community_colors = {
        'happiness': 'red',
        'depression': 'green',
        'anxiety': 'blue',
        'stress': 'yellow'
        # Add more community names and corresponding colors as needed
    }
    # Draw the graph with each community having a dedicated node color
    node_colors = [community_colors.get(community_map.get(node, 'purple'), 'purple') for node in G.nodes()]
    pos = nx.spring_layout(G)  # Position nodes using spring layout algorithm
    # Draw nodes with colors
    for community, color in community_colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[node for node, c in community_map.items() if c == community],
                               node_color=color, label=community)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    plt.legend()
    plt.show()