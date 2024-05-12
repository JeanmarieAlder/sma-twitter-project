import pandas as pd

from utils import check_tmp_folder_existing
from utils.graph import generate_graph
from utils.preprocessing import preprocess_tweets

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
    check_tmp_folder_existing()

    # 1. Data preprocessing steps
    print()
    print("Data Preprocessing")
    df = pd.DataFrame([])
    try:
        # Load the CSV file
        df = pd.read_csv('covid19_tweets.csv')
        print("Successfully loaded Tweets. Sample:")
        print(df.head())
    except Exception as e:
        print(e)
        print("Couldn't load tweets. Make sure to download them and place them in the root folder. More information about this in the readme file.")
        raise SystemExit(1)
    
    # get dataframe of tweet words, ready for graph creation.
    df_words = preprocess_tweets(df)

    # 2. Graph generation
    print()
    print("Graph Generation")
    generate_graph(df_words)


    # 3. Graph persistence
    print()
    print("Graph persistence")
    # TODO: save graph to a neo4j database.

    # 4. Apply community detection
    print()
    print("Community Detection")

    # 5. Analyse results
    print()
    print("Analyse results")


if __name__ == "__main__":
    main()