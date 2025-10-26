from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Return sentiment and normalized score"""
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        sentiment = 'POSITIVE'
    elif score <= -0.05:
        sentiment = 'NEGATIVE'
    else:
        sentiment = 'NEUTRAL'
    return sentiment, abs(score)
