"""
Utility functions for indexing
"""

import os
from pathlib import Path


def create_directories():
    """Create necessary directories if they don't exist"""
    from config import HTML_CORPUS_DIR, DEMO_CORPUS_DIR, OUTPUT_DIR
    
    directories = [HTML_CORPUS_DIR, DEMO_CORPUS_DIR, OUTPUT_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")


def get_index_stats(doc_ids, vocabulary, tfidf_matrix):
    """Display index statistics"""
    import numpy as np
    
    print("\nINDEX STATISTICS")
    print(f"\nDocuments: {len(doc_ids)}")
    print(f"Vocabulary size: {len(vocabulary)} unique terms")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    
    # Calculate sparsity - fix the count_nonzero issue
    if isinstance(tfidf_matrix, np.ndarray):
        non_zero = np.count_nonzero(tfidf_matrix)
    else:
        # If it's a list, convert to array first
        tfidf_matrix = np.array(tfidf_matrix)
        non_zero = np.count_nonzero(tfidf_matrix)
    
    total = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
    sparsity = (1 - non_zero / total) * 100
    
    print(f"Matrix sparsity: {sparsity:.2f}%")