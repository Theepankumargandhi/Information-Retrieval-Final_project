"""
Configuration file for Information Retrieval Project
Contains all settings and file paths
"""

import os
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
HTML_CORPUS_DIR = DATA_DIR / 'html_corpus'
DEMO_CORPUS_DIR = DATA_DIR / 'demo_corpus'
OUTPUT_DIR = DATA_DIR / 'output'

# Important file paths
INDEX_FILE = OUTPUT_DIR / 'index.json'
RESULTS_FILE = OUTPUT_DIR / 'results.csv'
QUERIES_FILE = DATA_DIR / 'queries.csv'

# Crawler configuration
CRAWLER_START_URL = 'https://en.wikipedia.org/wiki/Information_retrieval'
CRAWLER_DEPTH = 2
CRAWLER_MAX_PAGES = 100
CRAWLER_DELAY = 1  # Delay between requests (seconds)

# TF-IDF settings
USE_LOWERCASE = True
STOP_WORDS = 'english'
TFIDF_NORM = 'l2'

# The 3 official HTML files for grading
OFFICIAL_FILES = [
    '0F64A61C-DF01-4F43-8B8D-F0319C41768E.html',
    '1F648A7F-2C64-458C-BFAF-463A071530ED.html',
    '6B3BD97C-DEF2-49BB-B2B6-80F2CD53C4D3.html'
]

# Flask API settings
API_HOST = '127.0.0.1'
API_PORT = 5000