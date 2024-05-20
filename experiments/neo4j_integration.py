from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
import community as community_louvain
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set up NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Neo4j connection details
uri = "bolt://localhost:7687"
user = "Anudeep"
password = "22jumpstreet"  # Change to your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(user, password))

# Load the preprocessed tweets data
df_words = pd.read_csv('df_words.csv')


# Define sentiment analysis function
def get_sentiment(tweet):
    if not isinstance(tweet, str):
        tweet = ""
    scores = sia.polarity_scores(tweet)
    if scores['compound'] > 0.05:
        return 'positive'
    elif scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'


# Perform sentiment analysis on the tweets
df_words['nltk_sentiment'] = df_words['words'].apply(lambda x: get_sentiment(' '.join(eval(x))))


# Create nodes and relationships in Neo4j
def create_graph(tx, df):
    for index, row in df.iterrows():
        words = ' '.join(eval(row['words']))
        sentiment = row['nltk_sentiment']
        tx.run("CREATE (t:Tweet {id: $id, text: $text, sentiment: $sentiment})",
               id=index, text=words, sentiment=sentiment)

        for word in eval(row['words']):
            tx.run("MERGE (w:Word {text: $word})", word=word)
            tx.run("""
                MATCH (t:Tweet {id: $id}), (w:Word {text: $word})
                CREATE (t)-[:CONTAINS]->(w)
                """, id=index, word=word)


with driver.session() as session:
    session.write_transaction(create_graph, df_words)


# Perform community detection using Louvain method on the graph in Neo4j
def detect_communities(tx):
    result = tx.run("""
        CALL gds.graph.project('tweetGraph', 'Tweet', 'CONTAINS', {relationshipProperties: 'count'})
        CALL gds.louvain.write('tweetGraph', {
            writeProperty: 'community'
        })
        YIELD communityCount, modularity, modularities
        RETURN communityCount, modularity, modularities
        """)
    for record in result:
        print(f"Community Count: {record['communityCount']}, Modularity: {record['modularity']}")


with driver.session() as session:
    session.write_transaction(detect_communities)


# Query the graph to get community sentiment distributions
def get_community_sentiment_distribution(tx):
    result = tx.run("""
        MATCH (t:Tweet)
        WITH t.community AS community, t.sentiment AS sentiment, count(*) AS count
        RETURN community, sentiment, count
        """)
    community_sentiment_distribution = {}
    for record in result:
        community = record['community']
        sentiment = record['sentiment']
        count = record['count']
        if community not in community_sentiment_distribution:
            community_sentiment_distribution[community] = {}
        community_sentiment_distribution[community][sentiment] = count
    return community_sentiment_distribution


with driver.session() as session:
    community_sentiment_distribution = session.read_transaction(get_community_sentiment_distribution)
    for community, distribution in community_sentiment_distribution.items():
        print(f"Community {community} Sentiment Distribution: {distribution}")

# Close the Neo4j connection
driver.close()
