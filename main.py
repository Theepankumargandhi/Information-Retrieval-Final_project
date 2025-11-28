"""
Main script to run the Information Retrieval System
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from indexer.utils import create_directories
from indexer.indexer import load_documents, build_index, save_index, load_index
from indexer.utils import get_index_stats
from processor.query_processor import process_all_queries, save_results
from config import DEMO_CORPUS_DIR


def check_demo_corpus():
    """Check if demo corpus exists, offer to crawl if empty"""
    html_files = list(DEMO_CORPUS_DIR.glob('*.html'))
    
    if len(html_files) > 0:
        print(f"\nDemo corpus found: {len(html_files)} files")
        print("Skipping crawler (demo corpus already exists)")
        return False
    else:
        print("\nDemo corpus is empty.")
        response = input("Do you want to run the crawler? (yes/no): ").lower()
        return response in ['yes', 'y']


def run_crawler():
    """Run the Scrapy crawler"""
    print("\nStarting web crawler")
    print("This will crawl Wikipedia starting from Information Retrieval page")
    print("This will take a few minutes...")
    
    try:
        from scrapy.crawler import CrawlerProcess
        from crawler.wiki_crawler import WikipediaSpider
        
        process = CrawlerProcess()
        process.crawl(WikipediaSpider)
        process.start()
        
        html_files = list(DEMO_CORPUS_DIR.glob('*.html'))
        print(f"\nCrawling complete! Saved {len(html_files)} HTML files to demo_corpus/")
        return True
    except Exception as e:
        print(f"Crawler error: {e}")
        print("Continuing without crawler demo...")
        return False


def main():
    """Main function to run the IR system"""
    
    print("Information Retrieval System")
    
    print("\nStep 1: Creating directories")
    create_directories()
    
    # Optional: Run crawler if demo corpus is empty
    if check_demo_corpus():
        run_crawler()
    
    print("\nStep 2: Loading documents")
    documents = load_documents()
    
    if len(documents) == 0:
        print("Error: No documents found. Please add HTML files to data/html_corpus/")
        return
    
    print("\nStep 3: Building index")
    doc_ids, vocabulary, tfidf_matrix = build_index(documents)
    
    print("\nStep 4: Saving index")
    save_index(doc_ids, vocabulary, tfidf_matrix)
    
    print("\nStep 5: Displaying index statistics")
    # Convert to numpy array before passing to get_index_stats
    import numpy as np
    if not isinstance(tfidf_matrix, np.ndarray):
        tfidf_matrix = tfidf_matrix.toarray()
    get_index_stats(doc_ids, vocabulary, tfidf_matrix)
    
    print("\nStep 6: Processing queries")
    results = process_all_queries(vocabulary, tfidf_matrix, doc_ids)
    
    print("\nStep 7: Saving results")
    save_results(results)
    
    print("\nAll steps completed successfully")
    print("\nGenerated files:")
    print("  - data/output/index.json")
    print("  - data/output/results.csv")
    print("\nTo start the Flask API, run:")
    print("  python api/app.py")


if __name__ == '__main__':
    main()