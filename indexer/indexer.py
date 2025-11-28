"""
TF-IDF Indexer
Builds TF-IDF index from HTML documents
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import HTML_CORPUS_DIR, INDEX_FILE, OFFICIAL_FILES, USE_LOWERCASE, STOP_WORDS, TFIDF_NORM
from indexer.extractor import extract_text_from_html


def load_documents():
    """Load and extract text from official HTML files"""
    documents = {}
    
    print("Loading documents...")
    for filename in OFFICIAL_FILES:
        file_path = HTML_CORPUS_DIR / filename
        
        if file_path.exists():
            # Extract document ID from filename
            doc_id = filename.replace('.html', '')
            
            # Extract text content
            text = extract_text_from_html(file_path)
            documents[doc_id] = text
            
            print(f"Loaded: {filename} ({len(text)} characters)")
        else:
            print(f"Warning: Missing file {filename}")
    
    return documents


def build_index(documents):
    """Build TF-IDF index from documents"""
    print("\nBuilding TF-IDF index...")
    
    # Get document IDs and texts
    doc_ids = list(documents.keys())
    doc_texts = [documents[doc_id] for doc_id in doc_ids]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=USE_LOWERCASE,
        stop_words=STOP_WORDS,
        norm=TFIDF_NORM
    )
    
    # Fit and transform documents
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    vocabulary = vectorizer.get_feature_names_out()
    
    print(f"Index built: {len(doc_ids)} documents, {len(vocabulary)} terms")
    
    return doc_ids, vocabulary, tfidf_matrix


def save_index(doc_ids, vocabulary, tfidf_matrix):
    """Save index to JSON file"""
    index_data = {
        'document_ids': doc_ids,
        'vocabulary': vocabulary.tolist(),
        'tfidf_matrix': tfidf_matrix.toarray().tolist(),
        'vectorizer_params': {
            'lowercase': USE_LOWERCASE,
            'stop_words': STOP_WORDS,
            'norm': TFIDF_NORM
        }
    }
    
    # Save to file
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"\nIndex saved to: {INDEX_FILE}")
    print(f"File size: {INDEX_FILE.stat().st_size / 1024:.2f} KB")


def load_index():
    """Load index from JSON file"""
    print("Loading index...")
    
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    doc_ids = index_data['document_ids']
    vocabulary = index_data['vocabulary']
    tfidf_matrix = np.array(index_data['tfidf_matrix'])
    
    print(f"Index loaded: {len(doc_ids)} documents, {len(vocabulary)} terms")
    
    return doc_ids, vocabulary, tfidf_matrix