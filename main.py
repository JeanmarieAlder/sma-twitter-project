from utils import check_folder_existing
from utils.classification import classify_tweets
from utils.community_detection import generate_communities
from utils.graph import generate_graph
from utils.preprocessing import get_non_mental_health_tweets, get_sample_tweets, load_tweets, preprocess_tweets, preprocess_tweets_balanced
from utils.result_analysis import analyze_best_nodes, analyze_unclassified_tweets, compute_nmi

def main():
    """
    Main function to orchestrate the entire process.

    Steps:
    0. General setup
    1. Data preprocessing steps
    2. Graph generation
    3. Graph persistence (saving graph to a neo4j database)
    4. Apply community detection
    5. Analyze results
    """
    
    print("Starting process...")

    # 0. General setup
    check_folder_existing("temp")

    # 1. Data preprocessing steps
    print()
    print("Data Preprocessing")
    df = load_tweets()
    
    # get dataframe of tweet words, ready for graph creation.
    df, df_words = preprocess_tweets(df)

    # 2. Graph generation
    print()
    print("Graph Generation")
    df_words = generate_graph(df_words)


    # 3. Graph persistence
    print()
    print("Graph persistence")
    # TODO: save graph to a neo4j database.

    # 4. Apply community detection
    print()
    print("Community Detection")
    partition, df_words = generate_communities(df_words)

    # 5. Analyse results
    print()
    print("Results Analysis")
    nmi = compute_nmi(partition, df_words)
    analyze_unclassified_tweets(partition, df_words)


def main_modularity_gain():
    print("Starting process...")

    # 0. General setup
    check_folder_existing("temp")

    # 1. Data preprocessing steps
    print()
    print("Data Preprocessing")
    df = load_tweets()
    
    # get dataframe of tweet words, ready for graph creation.
    # df, df_words = preprocess_tweets(df)
    df, df_words = preprocess_tweets_balanced(df)

    # 2. Graph generation
    print()
    print("Graph Generation")
    df_words = generate_graph(df_words)

    # 3. Community Detection
    print()
    print("Community Detection")
    partition, df_words = generate_communities(df_words)

    # 4. Analyse Community Detection Results
    print()
    print("Results Analysis")
    nmi = compute_nmi(partition, df_words)
    analyze_unclassified_tweets(partition, df_words)
    analyze_best_nodes(partition, df_words)

    # 5. Classify Tweets
    print()
    print("Classify Tweets")
    df_test = get_non_mental_health_tweets(df)
    df_test = get_sample_tweets(df_test, 20) #Test sample
    df_results = classify_tweets(df_words, df_test, partition)


if __name__ == "__main__":
    # main()
    main_modularity_gain()
