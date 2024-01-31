# %%
# Import libraries
import pandas as pd
from transformers import pipeline, AutoTokenizer
import requests
import pdfplumber
import concurrent.futures
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import numpy as np
from gensim.downloader import load

# Download NLTK stopwords
nltk.download('stopwords')

# %%
# Import custom modules
import sentiment_analysis.sentiment_nltk as n
import sentiment_analysis.sentiment_transformers as t


# %%
def download_and_extract_text(company, year):
    text = None
    response = None

    if year == '2022':
        url = f'https://www.annualreports.com/HostedData/AnnualReports/PDF/{company}_{year}.pdf'
        response = requests.get(url, stream=True)
    else:
        # This loop tries URLs with different prefixes for non-2022 reports
        for prefix in [chr(97 + i) for i in range(26)]:  # Iterates through all lowercase letters (a to z)
            url = f'https://www.annualreports.com/HostedData/AnnualReportArchive/{prefix}/{company}_{year}.pdf'
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                break  # Exit the loop if a valid response is received

    # Check if a valid response was received
    if response is not None and response.status_code == 200:
        with BytesIO(response.content) as bytes_io:
            with pdfplumber.open(bytes_io) as pdf:
                pages_text = [page.extract_text() for page in pdf.pages if page.extract_text() is not None]
                text = ' '.join(pages_text)
        print(f"Successfully processed {company} {year}")
    else:
        print(f"Failed to process {company} {year}")

    return text


def download_and_process_reports_parallel(companies, years):
    all_texts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for company in companies:
            for year in years:
                future = executor.submit(download_and_extract_text, company, year)
                futures[future] = (company, year)  # Map future to (company, year)

        for future in concurrent.futures.as_completed(futures):
            company, year = futures[future]  # Get company and year from the future
            text = future.result()
            if text:
                all_texts[(company, year)] = text  # Store text with (company, year) as key

    return all_texts


def get_top_n_topics(bow_doc, lda_model, n=5):
    # Get the list of topic probabilities for the document
    doc_topics = lda_model.get_document_topics(bow_doc, minimum_probability=0.0)

    # Sort the topics by their probabilities (highest first)
    sorted_doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)

    # Select the top N topics
    top_n_topics = sorted_doc_topics[:n]

    return top_n_topics


def get_top_keywords_for_topic(lda_model, topic_id, topn=5):
    top_keywords = lda_model.show_topic(topic_id, topn=topn)
    return ', '.join([word for word, prob in top_keywords])


# %%
def keyword_analysis(text):
    positive_count = sum(text.count(word) for word in positive_keywords)
    negative_count = sum(text.count(word) for word in negative_keywords)
    return positive_count, negative_count


def make_word_cloud(text, title=None):
    if not text.strip():  # Checks if the text is empty or contains only whitespace
        print(f"No words found for '{title}'. Skipping word cloud generation.")
        return  # Exits the function early

    # Proceed with word cloud generation if text is not empty
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
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
    positive_text = ' '.join([word for word in text.lower().split() if word in positive])
    make_word_cloud(positive_text, "Positive Words")

    # Display negative word cloud
    print("Negative Word Cloud:")
    negative_text = ' '.join([word for word in text.lower().split() if word in negative])
    make_word_cloud(negative_text, "Negative Words")


# %%
def preprocess_text(text):
    stop_words = stopwords.words('english')
    return [word for word in simple_preprocess(text) if word not in stop_words]


def visualize_topics(model, corpus, id2word, company, year):
    pyLDAvis.enable_notebook()
    vis = gensimvis.prepare(model, corpus, id2word)
    return vis


def compare_topics(topic1, topic2):
    # Compare two topics and return the overlapping keywords
    keywords1 = set([word.strip() for word, _ in [pair.split('*') for pair in topic1.split('+')]])
    keywords2 = set([word.strip() for word, _ in [pair.split('*') for pair in topic2.split('+')]])
    return keywords1.intersection(keywords2)


def print_topics(topics):
    # Print the topics in a readable format
    for i, topic in enumerate(topics):
        keywords = ' + '.join([word for word, _ in [pair.split('*') for pair in topic.split('+')]])
        print(f"Topic {i + 1}: {keywords}")


def analyze_company_topics(df, company_name):
    # Analyze and compare topics for a given company across different years

    # Filter the DataFrame for the specified company
    company_df = df[df['Company'] == company_name]

    # Organize topics by year for the company
    company_topics = {}
    for index, row in company_df.iterrows():
        year = row['Year']
        topics = [row['Topic 1'], row['Topic 2'], row['Topic 3']]
        company_topics[year] = topics

    # Compare topics between years
    for year, topics in company_topics.items():
        print(f"Year: {year}")

        # Print topics for the current year
        print_topics(topics)

        for other_year, other_topics in company_topics.items():
            if year != other_year:  # Avoid comparing the same year
                for i, topic in enumerate(topics):
                    for j, other_topic in enumerate(other_topics):
                        overlap = compare_topics(topic, other_topic)
                        if overlap:
                            print(
                                f"  Overlap between Topic {i + 1} in {year} and Topic {j + 1} in {other_year}: {overlap}")
        print('-' * 50)


def get_vector(word):
    """Return the vector for a given word if it exists in the model."""
    try:
        return model[word]
    except KeyError:
        return np.zeros(model.vector_size)


def average_topic_vector(topic_keywords):
    """Compute the average vector for a list of topic keywords."""
    vectors = [get_vector(word) for word in topic_keywords.split(', ')]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 > 0 and norm_vec2 > 0 else 0


def calculate_cosine_similarities(company_name, df):
    """Calculate and print cosine similarities of topics aggregated by year for a given company."""
    # Filter the DataFrame for the specified company
    company_df = df[df['Company'] == company_name]

    # Group by 'Year' and aggregate 'Topic Keywords'
    yearly_topics = company_df.groupby('Year')['Topic Keywords'].apply(lambda topics: ', '.join(topics)).reset_index()

    # Calculate the average vector for each year's aggregated topics
    yearly_topics['Average Vector'] = yearly_topics['Topic Keywords'].apply(average_topic_vector)

    # Initialize a DataFrame to store cosine similarities
    cosine_similarities = pd.DataFrame(index=yearly_topics['Year'], columns=yearly_topics['Year'])

    # Calculate the cosine similarity between each pair of years
    for i, row_i in yearly_topics.iterrows():
        for j, row_j in yearly_topics.iterrows():
            cosine_similarities.at[row_i['Year'], row_j['Year']] = cosine_similarity(row_i['Average Vector'],
                                                                                     row_j['Average Vector'])

    # Print the cosine similarity matrix
    print(cosine_similarities)


# %%
# Define the list of companies and years
companies = ['NASDAQ_TSLA', 'NASDAQ_AAPL', 'NASDAQ_MSFT', 'NASDAQ_AMZN', 'NYSE_BRK-A', 'NYSE_PFE', 'NASDAQ_CCBG']
years = ['2022', '2021', '2020', '2019', '2018']

# Download and process the reports
all_texts = download_and_process_reports_parallel(companies, years)
# %%
# Define the positive and negative keywords
positive_keywords = [
    'good', 'great', 'positive', 'successful', 'profitable', 'improved', 'improving', 'excellent',
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

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis", model=model_name)
# %%
data = []
lda_models = {}
text_key = [(company, year) for company in companies for year in years]
# %%
for key in text_key:
    company, year = key
    if key in all_texts:
        text = all_texts[key]
        sentiment_score_nltk = n.analyze_sentiment_vader(text)
        print(f"{company} {year}: Keyword Analysis")
        print(f"\n Sentiment Score NLTK = {sentiment_score_nltk}")
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

        print('Positive Keywords:', positive_count)
        print('Negative Keywords:', negative_count)
        print('Positive Keywords Ratio:',
              positive_count / (negative_count + positive_count) if (negative_count + positive_count) != 0 else "N/A")
        make_map(text, positive_keywords, negative_keywords)

        # topic modeling part
        processed_text = preprocess_text(text)
        id2word = corpora.Dictionary([processed_text])
        corpus = [id2word.doc2bow(processed_text)]
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10,
                             alpha='asymmetric',
                             eta='auto',
                             iterations=400,
                             passes=15,
                             eval_every=None)
        lda_models[(company, year)] = lda_model
        bow_text = id2word.doc2bow(processed_text)
        top_topics = get_top_n_topics(bow_text, lda_model, n=1)

        # Extract the top keyword for each of the top 5 topics
        top_keywords = [get_top_keywords_for_topic(lda_model, topic_id) for topic_id, _ in top_topics]

        # Append data with top keywords as separate columns
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
            "Positive_Keywords_Ratio": positive_count / (negative_count + positive_count) if (
                                                                                                         negative_count + positive_count) != 0 else "N/A",
            **{f"Topic Keywords": keyword for i, keyword in enumerate(top_keywords)}
        })
        print('-' * 50, '\n')

df = pd.DataFrame(data)
# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(df)
# %%
key = ('NASDAQ_TSLA', '2022')
model = lda_models[key]
corpus = [id2word.doc2bow(preprocess_text(all_texts[key]))]
# vis = gensimvis.prepare(model, corpus, id2word)
# pyLDAvis.display(vis)
# %%
lda_models
# %%
model = load('glove-wiki-gigaword-100')
# %%
calculate_cosine_similarities('NASDAQ_TSLA', df)
# %%
for company in companies:
    print(f"Company: {company}")
    company_df = df[df['Company'] == company]
    company_topics = {}
    company_df = company_df[['Year', 'Topic Keywords']]
    print(company_df)
    calculate_cosine_similarities(company, df)
# %%
