
import pandas as pd


def compare_simple_method():
    print("Starting method validation.")
    print("Classifying tweets according to word occurences.")

    df = pd.DataFrame([])
    df_test = pd.DataFrame([])
    try:
        # Load the CSV file
        df = pd.read_csv('temp/df_words.csv')
        df_test = pd.read_csv('temp/df_test.csv')
        print("Successfully loaded df_words and df_test from the temp directory. Sample:")
        print("training set:")
        print(df.head())
        print("testing set:")
        print(df_test.head())
    except Exception as e:
        print(e)
        print("Couldn't load input or test tweets. Make sure to have ")
        raise SystemExit(1)
    
    # Initialize an object with three arrays for each sentiment
    classified_tweets = {
        "happy": [],
        "hope": [],
        "sad": []
    }

    # Iterate over each tweet in the DataFrame
    for index, row in df.iterrows():
        words = row['words'] if isinstance(row['words'], list) else eval(row['words'])  # Ensure words is a list
        if 'happy' in words:
            classified_tweets['happy'].append(row['words'])
        if 'hope' in words:
            classified_tweets['hope'].append(row['words'])
        if 'sad' in words:
            classified_tweets['sad'].append(row['words'])

    # Print the classified tweets for validation
    print("Classified tweets:")
    for sentiment, tweets in classified_tweets.items():
        print(f"{sentiment.capitalize()} tweets ({len(tweets)}):")
        for tweet in tweets:
            print(f"  - {tweet}")
    
    # Initialize counters for each class
    test_tweet_counts = []
    for index, tweet in df_test.iterrows():
        print(f"Processing test tweet {index}: {tweet['text']}")
        test_words = tweet['words'] if isinstance(tweet['words'], list) else eval(tweet['words'])  # Ensure words is a list

        count_happy = 0
        count_hope = 0
        count_sad = 0

        test_words_set = set(test_words)  # Convert to set for efficient comparison

        for sentiment in classified_tweets:
            for tw in classified_tweets[sentiment]:
                # Make sure tw is a string
                tw_words_list = eval(tw)
                tw_words_set = set(tw_words_list)  # Convert tweet words to set
                common_words = test_words_set & tw_words_set  # Find common words
                count = len(common_words)
                
                # print(sentiment)
                # print(tw)
                # print(tw_words_list)
                # print(tw_words_set)
                # print(common_words)
                # print(count)
                # input("stop")

                if sentiment == 'happy':
                    count_happy += count
                elif sentiment == 'hope':
                    count_hope += count
                elif sentiment == 'sad':
                    count_sad += count

        # Determine the prediction based on counts
        if count_happy > count_hope and count_happy > count_sad:
            prediction = 'happy'
        elif count_hope > count_happy and count_hope > count_sad:
            prediction = 'hope'
        elif count_sad > count_happy and count_sad > count_hope:
            prediction = 'sad'
        else:
            # Resolve ties by choosing the most positive class
            if count_happy == max(count_happy, count_hope, count_sad):
                prediction = 'happy'
            elif count_hope == max(count_happy, count_hope, count_sad):
                prediction = 'hope'
            else:
                prediction = 'sad'

        # Append counts and prediction to the list
        test_tweet_counts.append({
            'tweet': tweet['text'],
            'happy_count': count_happy,
            'hope_count': count_hope,
            'sad_count': count_sad,
            'prediction': prediction
        })

    # Add the validation_prediction column to df_test
    df_test['validation_prediction'] = [result['prediction'] for result in test_tweet_counts]

    # Count the occurrences of each class in the 'validation_prediction' column
    class_counts = df_test['validation_prediction'].value_counts()

    # Print the total number of tweets for each class
    for class_label, count in class_counts.items():
        print(f"{class_label}: {count}")
    
    # Write results to csv file
    df_test.to_csv('temp/df_test_with_predictions.csv', index=False)

    # Print the test tweet counts and predictions for validation
    # print("Test tweet counts and predictions:")
    # for counts in test_tweet_counts:
    #     print(f"Tweet: {counts['tweet']}")
    #     print(f"  - Happy count: {counts['happy_count']}")
    #     print(f"  - Hope count: {counts['hope_count']}")
    #     print(f"  - Sad count: {counts['sad_count']}")
    #     print(f"  - Prediction: {counts['prediction']}")


if __name__ == "__main__":
    compare_simple_method()