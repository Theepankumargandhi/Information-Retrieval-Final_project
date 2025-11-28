"""
Query Processor
Handles query processing and document ranking
"""

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import QUERIES_FILE, RESULTS_FILE, USE_LOWERCASE, STOP_WORDS, TFIDF_NORM
from processor.similarity import compute_cosine_similarity, rank_documents


def load_queries():
    """Load queries from CSV file"""
    queries = []
    
    print("Loading queries...")
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                'query_id': row['query_id'],
                'query_text': row['query_text']
            })
    
    print(f"Loaded {len(queries)} queries")
    return queries


def process_query(query_text, vocabulary, tfidf_matrix, doc_ids):
    """
    Process a single query and return ranked documents.
    
    Args:
        query_text: The search query
        vocabulary: List of terms from index
        tfidf_matrix: Document TF-IDF matrix
        doc_ids: List of document IDs
        
    Returns:
        List of tuples: (doc_id, rank, score)
    """
    # Create vectorizer with same vocabulary as index
    query_vectorizer = TfidfVectorizer(
        lowercase=USE_LOWERCASE,
        stop_words=STOP_WORDS,
        vocabulary=vocabulary,
        norm=TFIDF_NORM
    )
    
    # Transform query to TF-IDF vector
    query_vector = query_vectorizer.fit_transform([query_text]).toarray()
    
    # Calculate similarity scores
    similarities = compute_cosine_similarity(query_vector, tfidf_matrix)
    
    # Rank documents
    ranked_results = rank_documents(doc_ids, similarities)
    
    return ranked_results


def process_all_queries(vocabulary, tfidf_matrix, doc_ids):
    """Process all queries and collect results"""
    queries = load_queries()
    all_results = []
    
    print("\nProcessing queries...")
    for query in queries:
        query_id = query['query_id']
        query_text = query['query_text']
        
        print(f"\nQuery: '{query_text}'")
        
        # Get ranked documents
        ranked_docs = process_query(query_text, vocabulary, tfidf_matrix, doc_ids)
        
        # Display results
        print("Results:")
        for doc_id, rank, score in ranked_docs:
            print(f"  Rank {rank}: {doc_id} (score: {score:.4f})")
            
            # Add to results list
            all_results.append({
                'query_id': query_id,
                'rank': rank,
                'document_id': doc_id
            })
    
    return all_results


def save_results(results):
    """Save query results to CSV file"""
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'rank', 'document_id'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Total entries: {len(results)}")