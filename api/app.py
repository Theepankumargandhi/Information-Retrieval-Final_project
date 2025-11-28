"""
Flask REST API for Information Retrieval System
Supports both TF-IDF and Word2Vec search
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import API_HOST, API_PORT, INDEX_FILE, HTML_CORPUS_DIR, OFFICIAL_FILES
from indexer.indexer import load_index
from indexer.extractor import extract_text_from_html
from processor.query_processor import process_query
from processor.word2vec_search import process_query_word2vec, create_document_embeddings

app = Flask(__name__)

# Global variables for TF-IDF
api_vocabulary = None
api_tfidf_matrix = None
api_doc_ids = None

# Global variables for Word2Vec
api_doc_embeddings = None


@app.route('/')
def home():
    """Render the search interface"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """
    Search endpoint with method selection.
    
    Expected JSON:
        {"query": "text", "top_k": 3, "method": "tfidf" or "word2vec"}
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', 3)
        method = data.get('method', 'tfidf')
        
        # Process based on method
        if method == 'word2vec':
            ranked_docs = process_query_word2vec(query_text, api_doc_embeddings)
        else:
            ranked_docs = process_query(
                query_text,
                api_vocabulary,
                api_tfidf_matrix,
                api_doc_ids
            )
        
        # Format results
        results = []
        for doc_id, rank, score in ranked_docs[:top_k]:
            results.append({
                'rank': rank,
                'document_id': doc_id,
                'score': float(score)
            })
        
        return jsonify({
            'query': query_text,
            'method': method,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents': len(api_doc_ids) if api_doc_ids else 0,
        'methods': ['tfidf', 'word2vec']
    })


def load_index_for_api():
    """Load index and create Word2Vec embeddings"""
    global api_vocabulary, api_tfidf_matrix, api_doc_ids, api_doc_embeddings
    
    print("Loading index for API...")
    api_doc_ids, api_vocabulary, api_tfidf_matrix = load_index()
    print("TF-IDF index loaded")
    
    # Load documents for Word2Vec
    print("\nLoading documents for Word2Vec...")
    documents = {}
    for filename in OFFICIAL_FILES:
        file_path = HTML_CORPUS_DIR / filename
        if file_path.exists():
            doc_id = filename.replace('.html', '')
            text = extract_text_from_html(file_path)
            documents[doc_id] = text
    
    # Create Word2Vec embeddings
    api_doc_embeddings = create_document_embeddings(documents)
    print("Word2Vec embeddings ready")
    
    print("\nAPI ready with both TF-IDF and Word2Vec!")


if __name__ == '__main__':
    load_index_for_api()
    app.run(host=API_HOST, port=API_PORT, debug=True)