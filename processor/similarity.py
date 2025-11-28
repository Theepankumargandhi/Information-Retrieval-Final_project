"""
Similarity calculations for ranking documents
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_cosine_similarity(query_vector, document_matrix):
    """
    Compute cosine similarity between query and all documents.
    
    Args:
        query_vector: TF-IDF vector for query
        document_matrix: TF-IDF matrix for all documents
        
    Returns:
        Array of similarity scores
    """
    similarities = cosine_similarity(query_vector, document_matrix)[0]
    return similarities


def rank_documents(doc_ids, similarities):
    """
    Rank documents by similarity scores.
    
    Args:
        doc_ids: List of document IDs
        similarities: Array of similarity scores
        
    Returns:
        List of tuples: (doc_id, rank, score)
    """
    # Combine document IDs with their scores
    doc_scores = list(zip(doc_ids, similarities))
    
    # Sort by score in descending order
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Add ranks (1-indexed)
    ranked_results = [
        (doc_id, rank + 1, score)
        for rank, (doc_id, score) in enumerate(doc_scores)
    ]
    
    return ranked_results