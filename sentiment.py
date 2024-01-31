import os
import pandas as pd
from transformers import pipeline, AutoTokenizer

import sentiment_analysis.sentiment_nltk as n
import sentiment_analysis.sentiment_transformers as t

positive_keywords = [
    'good', 'great', 'positive', 'successful', 'profitable', 'improved', 'increase',
    'beneficial', 'strong', 'growth', 'upturn', 'bullish', 'booming', 'advantageous',
    'rewarding', 'lucrative', 'surplus', 'expansion', 'upswing', 'thriving', 'yielding',
    'gains', 'outperform', 'optimistic', 'upbeat', 'recovery', 'acceleration', 'enhancement',
    'rally', 'surge', 'boom', 'profitability', 'efficiency', 'superior', 'leadership',
    'innovation', 'breakthrough', 'high-demand', 'competitive edge', 'market leader',
    'dividend increase', 'shareholder value', 'capital gain', 'revenue growth', 'cost reduction',
    'strategic acquisition', 'synergy', 'scalability', 'liquidity'
]
negative_keywords = [
    'bad', 'poor', 'negative', 'loss', 'problem', 'decrease', 'difficult', 'weak', 'decline',
    'losses', 'bearish', 'slump', 'downturn', 'adverse', 'challenging', 'deteriorating',
    'declining', 'recession', 'deficit', 'contraction', 'downgrade', 'volatility', 'risk',
    'uncertainty', 'impairment', 'write-off', 'underperform', 'pessimistic', 'downbeat',
    'stagnation', 'erosion', 'turmoil', 'crisis', 'bankruptcy', 'default', 'devaluation',
    'overleveraged', 'layoffs', 'restructuring', 'downsizing', 'liquidation', 'fraud',
    'scandal', 'litigation', 'regulatory penalty', 'market exit', 'competitive pressure',
    'product recall', 'safety concern'
]

def keyword_analysis(text):
    positive_count = sum(text.count(word) for word in positive_keywords)
    negative_count = sum(text.count(word) for word in negative_keywords)
    return positive_count, negative_count

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

# List of companies and years to check
companies = ['NASDAQ_TSLA', 'NASDAQ_AAPL', 'NASDAQ_MSFT', 'NASDAQ_AMZN', 'NYSE_BRK-A', 'NYSE_PFE', 'NASDAQ_CCBG']
years = ['2022', '2021', '2020', '2019']

discussion_dir = os.path.join(os.getcwd(), 'discussion')

data = []

for company in companies:
    for year in years:
        filename = f"{company}_{year}_DISCUSSION.txt"
        filepath = os.path.join(discussion_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                sentiment_score_nltk = n.analyze_sentiment_vader(text)
                print(f"{company} {year}:\n Sentiment Score NLTK = {sentiment_score_nltk}")
                scores = n.analyze_sentiment_vader_detail(text)
                print(f"  Positive Score: {scores['pos']}")
                print(f"  Negative Score: {scores['neg']}")
                print(f"  Neutral Score: {scores['neu']}")

                cleaned_text = t.clean_text(text)
                sentiment_score_transformer = t.analyze_sentiment(cleaned_text, tokenizer, sentiment_analysis)
                print(f"{company} {year}: Sentiment Score Transformer = {sentiment_score_transformer}\n")

                # Keyword analysis
                cleaned_text = t.clean_text(text)
                positive_count, negative_count = keyword_analysis(cleaned_text)

                data.append({
                    "Company": company,
                    "Year": year,
                    "NLTK_Sentiment_Score": sentiment_score_nltk,
                    "Positive_Score": scores['pos'],
                    "Negative_Score": scores['neg'],
                    "Neutral_Score": scores['neu'],
                    "Transformer_Sentiment_Score": sentiment_score_transformer,
                    "Positive_Keywords": positive_count,
                    "Negative_Keywords": negative_count,
                })

df = pd.DataFrame(data)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def make_word_cloud(text, title=None):
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    if title:
        plt.title(title)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def make_map(text, positive, negative):
    # Display word cloud for all words
    print("All Words Word Cloud:")
    make_word_cloud(text, "All Words")

    # Display positive word cloud
    print("Positive Word Cloud:")
    # Filter the text for positive words and create a word cloud
    positive_text = ' '.join([word for word in text.lower().split() if word in positive])
    make_word_cloud(positive_text, "Positive Words")

    # Display negative word cloud
    print("Negative Word Cloud:")
    # Filter the text for negative words and create a word cloud
    negative_text = ' '.join([word for word in text.lower().split() if word in negative])
    make_word_cloud(negative_text, "Negative Words")

# Example usage
filename = f"NYSE_PFE_2021_DISCUSSION.txt"
filepath = os.path.join(discussion_dir, filename)
if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
make_map(text, positive_keywords, negative_keywords)