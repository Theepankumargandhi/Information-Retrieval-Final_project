"""
HTML Text Extraction
Extracts clean text from HTML files
"""

from bs4 import BeautifulSoup
import re


def extract_text_from_html(html_file_path):
    """
    Extract clean text from HTML file.
    
    Args:
        html_file_path: Path to HTML file
        
    Returns:
        Cleaned text string
    """
    try:
        # Read HTML file
        with open(html_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script, style, meta, and link tags
        for script in soup(['script', 'style', 'meta', 'link']):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from {html_file_path}: {e}")
        return ""