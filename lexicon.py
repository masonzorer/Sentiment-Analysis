# lexicon approaches to sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# VADER model
def vader(data):
    # load in VADER model
    analyzer = SentimentIntensityAnalyzer()

    # run sentiment analysis on each entry
    sentiment_scores = []
    for entry in data['entry']:
        vs = analyzer.polarity_scores(entry)
        score = vs['compound']
        if score >= 0.15:
            sentiment_scores.append('POSITIVE')
        elif score <= -0.15:
            sentiment_scores.append('NEGATIVE')
        else:
            sentiment_scores.append('NEUTRAL')

    # add sentiment scores to dataframe
    data['prediction'] = sentiment_scores
    return data

# textblob model
def textblob(data):

    # run sentiment analysis on each entry
    sentiment_scores = []
    for entry in data['entry']:
        score = TextBlob(entry).sentiment.polarity
        if score >= 0.15:
            sentiment_scores.append('POSITIVE')
        elif score <= -0.15:
            sentiment_scores.append('NEGATIVE')
        else:
            sentiment_scores.append('NEUTRAL')

    # add sentiment scores to dataframe
    data['prediction'] = sentiment_scores
    return data



