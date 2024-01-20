import os
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

def analyze_sentiment_vader(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores['compound']  # Returns a compound score between -1 and 1

def analyze_sentiment_vader_detail(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores

def chunk_text(text, max_length):
    for i in range(0, len(text), max_length):
        yield text[i:i + max_length]

def analyze_sentiment(text):
    chunks = chunk_text(text, 512)  # 512 tokens, adjust if needed
    total_score = 0
    num_chunks = 0

    for chunk in chunks:
        result = sentiment_analysis(chunk)
        if result and len(result) > 0:
            total_score += result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
            num_chunks += 1

    return total_score / num_chunks if num_chunks > 0 else 0

if __name__ == '__main__':
    # Directory containing the discussion text files
    parent_dir = os.path.join(os.getcwd(), os.pardir)
    discussion_dir = os.path.join(parent_dir, 'discussion')

    # Load a pre-trained model
    sentiment_analysis = pipeline("sentiment-analysis")

    # List of company PDFs to check
    companies = ['NASDAQ_TSLA', 'NASDAQ_AAPL', 'NASDAQ_MSFT', 'NASDAQ_AMZN', 'NYSE_BRK-A', 'NYSE_PFE', 'NASDAQ_CCBG']
    years = ['2022', '2021', '2020', '2019']

    for company in companies:
        for year in years:
            filename = f"{company}_{year}_DISCUSSION.txt"
            filepath = os.path.join(discussion_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    sentiment_score = analyze_sentiment_vader(text)
                    print(f"{company} {year}: Sentiment Score = {sentiment_score}")
                    scores = analyze_sentiment_vader_detail(text)
                    print(f"  Positive Score: {scores['pos']}")
                    print(f"  Negative Score: {scores['neg']}")
                    print(f"  Neutral Score: {scores['neu']}\n")

    print('end')
