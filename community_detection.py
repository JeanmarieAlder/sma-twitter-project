import networkx as nx
import community  # Louvain algorithm
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the graph from tweet_sample_v2.graph
G = nx.read_edgelist('tweet_sample_v2.graph', nodetype=int)

# Apply Louvain algorithm to detect communities
partition = community.best_partition(G)

# Organize nodes by community
communities = defaultdict(list)
for node, community_id in partition.items():
    communities[community_id].append(node)

# Print information about communities
for community_id, nodes in communities.items():
    print(f"Community {community_id}:")
    print(f"Number of nodes: {len(nodes)}")
    print("Nodes:", nodes)
    print()


# Draw the graph with each community having a dedicated node color
node_colors = [partition[node] for node in G.nodes()]
nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.tab10)
plt.show()