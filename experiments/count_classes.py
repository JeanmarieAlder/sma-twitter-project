import pandas as pd

df = pd.read_csv('output/202405231528_mod_gain_output.csv') 
classified_tweets = {
        "happy": 0,
        "hope": 0,
        "sad": 0
    }

for index, row in df.iterrows():
    classified_tweets[row["community"]] += 1

print(classified_tweets)