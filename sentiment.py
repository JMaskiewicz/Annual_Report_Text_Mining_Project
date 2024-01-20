import os
import pandas as pd
from transformers import pipeline, AutoTokenizer

import sentiment_analysis.sentiment_nltk as n
import sentiment_analysis.sentiment_transformers as t

positive_keywords = ['good', 'great', 'positive', 'successful', 'profitable', 'improved', 'increase', 'beneficial', 'strong']
negative_keywords = ['bad', 'poor', 'negative', 'loss', 'problem', 'decrease', 'difficult', 'weak', 'decline']

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