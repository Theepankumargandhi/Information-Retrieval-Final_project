# Information Retrieval System

A modular end-to-end information retrieval system that crawls web pages, builds TF-IDF indexes, and ranks documents using cosine similarity. Includes both keyword-based and semantic search capabilities.

## Features

- **Web Crawler**: Scrapy-based Wikipedia crawler for collecting HTML documents
- **Document Indexing**: TF-IDF vectorization using scikit-learn
- **Query Processing**: Cosine similarity-based document ranking
- **Dual Search Methods**: 
  - TF-IDF (keyword-based)
  - Word2Vec (semantic similarity)
- **Web Interface**: Flask REST API with interactive search UI
- **Spelling Correction**: NLTK-based query term correction

## Project Structure
```
project/
├── crawler/
│   ├── __init__.py
│   └── wiki_crawler.py          # Scrapy spider for Wikipedia
├── indexer/
│   ├── __init__.py
│   ├── extractor.py             # HTML text extraction
│   ├── indexer.py               # TF-IDF index builder
│   └── utils.py                 # Utility functions
├── processor/
│   ├── __init__.py
│   ├── query_processor.py       # TF-IDF query processing
│   ├── similarity.py            # Similarity calculations
│   └── word2vec_search.py       # Semantic search
├── api/
│   ├── __init__.py
│   ├── app.py                   # Flask REST API
│   └── templates/
│       └── index.html           # Web interface
├── data/
│   ├── html_corpus/             # Official HTML files (3 files)
│   ├── demo_corpus/             # Crawled Wikipedia pages
│   ├── output/                  # Generated index and results
│   └── queries.csv              # Test queries
├── config.py                    # Configuration settings
├── main.py                      # Main pipeline script
└── requirements.txt             # Python dependencies
```
##  Screenshots
**Front Page**  
![Front Page](Screenshots/front_page.png)

**results**  
![results Page](Screenshots/results.png)

## Installation

1. **Clone the repository**
```bash
git clone <https://github.com/Theepankumargandhi/Information-Retrieval-Final_project>
cd project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup directories**
Place the 3 official HTML files in `data/html_corpus/`

## Usage

### Run Complete Pipeline
```bash
python main.py
```

This will:
1. Check if demo corpus exists 
2. Load documents from `data/html_corpus/`
3. Build TF-IDF index
4. Process queries from `data/queries.csv`
5. Generate `data/output/index.json` and `data/output/results.csv`

### Run Web Crawler (Optional)

The crawler runs automatically if demo corpus is empty, or run separately:
```bash
scrapy runspider crawler/wiki_crawler.py
```

### Start Web Interface
```bash
python api/app.py
```

Then open browser: `http://127.0.0.1:5000`

## Configuration

Edit `config.py` to modify:
- Corpus directories
- Output file paths
- Crawler settings (depth, max pages, delay)
- TF-IDF parameters (normalization, stop words)
- API host/port

## Input Files

### Required
- **HTML Corpus**: 3 instructor-provided HTML files in `data/html_corpus/`
- **Queries CSV**: Query file at `data/queries.csv` with format:
```csv
  query_id,query_text
  <uuid>,search query text
```

### Optional
- **Demo Corpus**: Crawled Wikipedia pages in `data/demo_corpus/`

## Output Files

- **index.json**: TF-IDF index containing:
  - Document IDs
  - Vocabulary (4692 terms)
  - TF-IDF matrix
  - Vectorizer parameters

- **results.csv**: Query results with format:
```csv
  query_id,rank,document_id
  <uuid>,1,<doc_id>
```

## API Endpoints

### GET /
Returns the web search interface

### POST /search
Search for documents

**Request:**
```json
{
  "query": "information retrieval",
  "top_k": 3,
  "method": "tfidf"  
}
```

**Response:**
```json
{
  "query": "information retrieval",
  "method": "tfidf",
  "results": [
    {
      "rank": 1,
      "document_id": "6B3BD97C-DEF2-49BB-B2B6-80F2CD53C4D3",
      "score": 0.7248
    }
  ]
}
```

### GET /health
Health check endpoint

## Validation

Results are validated against instructor-provided expected rankings. Manual TF-IDF calculations confirm:

**Query 1: "information overload"**
- Rank 1: 6B3BD97C (Score: 0.264327)
- Rank 2: 0F64A61C (Score: 0.052883)
- Rank 3: 1F648A7F (Score: 0.049492)

All rankings match automated output.

## Technical Details

### TF-IDF Implementation
- **Vectorizer**: sklearn.TfidfVectorizer
- **Normalization**: L2
- **Stop Words**: English
- **IDF Formula**: log((1 + N) / (1 + df)) + 1

### Word2Vec
- **Model**: glove-wiki-gigaword-50
- **Vector Size**: 50 dimensions
- **Similarity**: Cosine similarity on averaged word vectors

### Index Statistics
- Documents: 3
- Vocabulary: 4692 unique terms
- Matrix Shape: (3, 4692)
- Sparsity: ~53.71%

## Dependencies
```
scrapy==2.11.0
beautifulsoup4==4.12.2
lxml==4.9.3
scikit-learn==1.3.0
numpy==1.24.3
flask==3.0.0
gensim==4.3.2
```

## Future Enhancements

- Query expansion using WordNet
- Additional ranking models (BM25)
- Neural ranking models
- Larger document collections
- Advanced query validation
- Relevance feedback

## Author

Theepan Kumar Gandhi
Information Retrieval Course Project

## License

This project is for educational purposes.
