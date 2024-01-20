import os
from transformers import pipeline, AutoTokenizer
import re

def chunk_text(text, tokenizer, max_tokens=500):
    # Tokenize the text, keeping special tokens
    tokens = tokenizer.encode(text, add_special_tokens=True)
    max_chunk_size = max_tokens - 2

    for i in range(0, len(tokens), max_chunk_size):
        chunk_tokens = tokens[i:i + max_chunk_size]
        chunk_tokens = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
        chunk = tokenizer.decode(chunk_tokens)
        yield chunk

def analyze_sentiment(text, tokenizer, sentiment_analysis):
    chunks = chunk_text(text, tokenizer)
    total_score = 0
    num_chunks = 0
    for chunk in chunks:
        result = sentiment_analysis(chunk)
        if result and len(result) > 0:
            total_score += result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
            num_chunks += 1
    return total_score / num_chunks if num_chunks > 0 else 0

def clean_text(text):
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\n\s*\d+\s+\d+.*\n', '\n', text)
    return text

if __name__ == '__main__':
    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

    # Directory containing the discussion text files
    discussion_dir = os.path.join(os.path.dirname(os.getcwd()), 'discussion')

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
                    cleaned_text = clean_text(text)
                    sentiment_score = analyze_sentiment(cleaned_text)
                    print(f"{company} {year}: Sentiment Score = {sentiment_score}")

    print('end')