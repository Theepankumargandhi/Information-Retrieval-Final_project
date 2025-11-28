"""
Wikipedia Crawler using Scrapy
Crawls Wikipedia pages and saves HTML files
"""

import scrapy
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CRAWLER_START_URL, CRAWLER_DEPTH, CRAWLER_MAX_PAGES, CRAWLER_DELAY, DEMO_CORPUS_DIR


class WikipediaSpider(scrapy.Spider):
    """Simple spider to crawl Wikipedia pages"""
    
    name = 'wiki_crawler'
    start_urls = [CRAWLER_START_URL]
    
    # Crawler settings
    custom_settings = {
        'DEPTH_LIMIT': CRAWLER_DEPTH,
        'CLOSESPIDER_PAGECOUNT': CRAWLER_MAX_PAGES,
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': CRAWLER_DELAY,
        'USER_AGENT': 'Mozilla/5.0 (Educational Project)'
    }
    
    page_count = 0
    
    def parse(self, response):
        """Parse each page and save HTML"""
        # Extract page ID from URL
        page_id = response.url.split('/')[-1]
        filename = DEMO_CORPUS_DIR / f'{page_id}.html'
        
        # Save HTML content
        with open(filename, 'wb') as f:
            f.write(response.body)
        
        self.page_count += 1
        print(f"Crawled [{self.page_count}]: {page_id}")
        
        # Follow links to other Wikipedia articles
        for link in response.css('a::attr(href)').getall():
            if link.startswith('/wiki/') and ':' not in link:
                yield response.follow(link, callback=self.parse)