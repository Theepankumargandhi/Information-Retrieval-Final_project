"""
Word2Vec Semantic Search
Uses word embeddings for semantic similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Fix gensim import for different versions
try:
    import gensim.downloader as api
except (ImportError, AttributeError):
    from gensim import downloader as api


# Global Word2Vec model
w2v_model = None


def load_word2vec_model():
    """Load pre-trained Word2Vec model"""
    global w2v_model
    
    if w2v_model is None:
        print("Loading Word2Vec model (this may take a minute)...")
        w2v_model = api.load('glove-wiki-gigaword-50')
        print("Word2Vec model loaded!")
    
    return w2v_model


def get_document_embedding(text, model):
    """Convert document text to embedding by averaging word vectors"""
    words = text.lower().split()
    
    # Get word vectors for words in vocabulary
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])
    
    # Return average or zero vector
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def create_document_embeddings(documents):
    """Create embeddings for all documents"""
    model = load_word2vec_model()
    doc_embeddings = {}
    
    print("Creating document embeddings...")
    for doc_id, text in documents.items():
        embedding = get_document_embedding(text, model)
        doc_embeddings[doc_id] = embedding
    
    return doc_embeddings


def process_query_word2vec(query_text, doc_embeddings):
    """
    Rank documents using Word2Vec semantic similarity.
    
    Args:
        query_text: Search query
        doc_embeddings: Dictionary of document embeddings
        
    Returns:
        List of tuples: (doc_id, rank, score)
    """
    model = load_word2vec_model()
    
    # Get query embedding
    query_embedding = get_document_embedding(query_text, model)
    
    # Compute similarities
    similarities = []
    for doc_id, doc_embedding in doc_embeddings.items():
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            doc_embedding.reshape(1, -1)
        )[0][0]
        similarities.append((doc_id, similarity))
    
    # Sort and rank
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    ranked_results = [
        (doc_id, rank + 1, score)
        for rank, (doc_id, score) in enumerate(similarities)
    ]
    
    return ranked_results