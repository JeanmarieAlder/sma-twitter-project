# Tweet Analysis

Work in process. For a master lecture project at University of Fribourg, Switzerland.

## Introduction
This project aims to analyze a collection of tweets stored in a CSV file. The goal is to extract relevant information, filter out irrelevant tweets, and create a graph representation based on the similarities between tweets.

## Steps
1. **Data Preprocessing**: The CSV file containing tweets is loaded into a DataFrame using Python and Pandas. The tweets are then cleaned to remove unnecessary symbols, mentions, and non-English tweets.
   
2. **Filtering by Mental Health Words**: Tweets containing specific mental health-related words are filtered and stored in a separate DataFrame. These words are chosen to capture a range of mental health-related topics, including anxiety, depression, therapy, support, and more.
   
3. **Graph Creation**: A graph representation of the tweets is created using NetworkX, where each tweet is a vertex, and an edge is added between two tweets if they share common words related to mental health.
   
4. **Visualization**: The graph is visualized using Matplotlib to provide a visual representation of the relationships between tweets.

## Files
- **covid19_tweets.csv**: Input CSV file containing the tweets. Can be downloaded at https://www.kaggle.com/datasets/gpreda/covid19-tweets?resource=download.
- **df_words.csv**: CSV file containing a DataFrame with words extracted from the tweets. This is the output of preprocess_tweets.py.
- **tweet.graph**: Text file containing the edges of the graph representation of tweets. This is the output of create_graph.py.

## Instructions
1. Recommended: create a virtual environment, with something like this (may change on your system, check documentation online):
```bash
python -m venv venv
```
2. Ensure Python and the required libraries (Pandas, NetworkX, Matplotlib) are installed. You can use the requirements.txt file for this with the command:
```bash
pip install -r requirements.txt
```
3. Run the Python scripts in the following order:
   - `dl_nltk.py`: Should be run once only to download word data for word filtering.
   - `preprocess_tweets.py`: Cleans and filters the tweets.
   - `create_graph.py`: Creates a graph representation of the tweets.
   - `visualize_graph.py`: Visualizes the graph.
4. Review the generated files (`df_words.csv`, `tweet.graph`) for further analysis.

## Contributors
- Anu and JM

