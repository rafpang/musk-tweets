
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)


if __name__ == '__main__':

    df = pd.read_csv('ElonTweets.csv')
    #tweets_and_sentment 
    tweets = df[['Text']].copy()
    sentiments = []
    sentiment_scores = []
    preprocessed_tweets = []
    for i, row in tweets.iterrows():
        text = row.Text
        text = preprocess(text)
        preprocessed_tweets.append(text) 
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        scores_list = []

        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            label_score = (l,s)
            scores_list.append(label_score)
        
        sentiment, sentiment_score = scores_list[0][0], scores_list[0][1]
        sentiments.append(sentiment)
        sentiment_scores.append(sentiment_score)
    df['sentiment'] = sentiments
    df['sentiment_score'] = sentiment_scores
    df['preprocessed_tweet'] = preprocessed_tweets
    df.to_csv('final_data.csv')
