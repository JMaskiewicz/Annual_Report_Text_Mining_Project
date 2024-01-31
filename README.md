# Corporate Annual Reports Analysis

## Project Overview

This repository hosts a Python-based project aimed at conducting an in-depth analysis of Discussion and Analysis part of corporate annual reports from various companies spanning multiple years. The primary goal is to extract, process, and analyze the textual content of these reports to gain insights into corporate strategies, performance, and market positioning.

## Main Project File

The core analysis and demonstrations of this project are encapsulated in the Jupyter Notebook file named `example.ipynb`. This notebook provides an interactive environment where the project's methodologies are applied step-by-step, from data extraction to in-depth analysis and visualization.

## Key Features

- **Textual Data Extraction**: Automated extraction of text from PDF-formatted annual reports, enabling the analysis of unstructured data.
- **D&A Analysis**: Focused examination of Discussion and Analysis sections to assess companies' innovation efforts and strategic commitments.
- **Sentiment Analysis**: Utilizes both NLTK and advanced transformer models to evaluate the sentiment conveyed in the reports, capturing a range of expressions from explicit to subtle.
- **Keyword Analysis**: Detailed analysis to identify and count specific positive and negative keywords, providing insights into the tonality of the reports.
- **Topic Modeling**: Employs Latent Dirichlet Allocation (LDA) to uncover the main themes within the reports, revealing strategic and operational focuses.
- **Visualization**: Incorporates word clouds for visual representation of key terms and sentiments, enhancing data accessibility and comprehension.
- **Thematic Consistency Analysis**: Applies cosine similarity measures to evaluate the consistency or divergence of themes across different reports and timeframes.

## Technologies and Libraries

This project leverages a suite of Python libraries, including:
- `pandas` for data manipulation,
- `transformers` and `nltk` for natural language processing,
- `pdfplumber` for PDF text extraction,
- `gensim` for topic modeling,
- `matplotlib` and `WordCloud` for data visualization,
- Custom sentiment analysis modules for nuanced sentiment evaluation.

## Repository Structure

- `sentiment_analysis/`: Custom modules for sentiment analysis.
- `reports/`: Sample annual reports in PDF format.
- `discussions/`: Extracted text from the reports for analysis.
- Main analysis scripts demonstrating data extraction, processing, and analysis workflows.

## Data Source

This project is based on annual reports sourced from [AnnualReports.com](https://www.annualreports.com/), a leading provider of corporate annual reports for public companies from various industries worldwide. The textual data for analysis is mined or scraped directly from the website, focusing on reports from selected companies over multiple years.

## Getting Started

To replicate the analysis or adapt it to other datasets, clone the repository and ensure all dependencies are installed:

```bash
git clone <repository-url>
pip install -r requirements.txt
