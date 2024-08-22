# Assessment 3 - Webcrawler and NLP System
# James Ross (14472266)
# PART 1 - Data Collection Pipeline 
# For project on GitHub: https://github.com/jimy-r/MA5851_Assessment3

import requests
from bs4 import BeautifulSoup
import random
import time
from urllib.parse import urlparse
import re
import feedparser
import json
from datetime import datetime, timedelta, timezone
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Setup Chrome options (headful browser)
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Running headless
chrome_options.add_argument("--ignore-certificate-errors")  # Ignore certificate errors
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Replace with the path to your WebDriver
service = Service('C:/Users/jimy/chromedriver.exe')
driver = webdriver.Chrome(service=service, options=chrome_options)

# Function to fetch and parse the robots.txt file
def fetch_robots_txt(base_url):
    robots_url = base_url.rstrip('/') + '/robots.txt'
    try:
        response = requests.get(robots_url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching robots.txt: {str(e)}")
        return ""

# Function to check if a URL is allowed based on robots.txt
def is_allowed_by_robots(robots_txt, url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    disallowed_patterns = []
    for line in robots_txt.splitlines():
        if line.startswith('Disallow:'):
            pattern = line.split(':')[1].strip()
            if pattern:
                disallowed_patterns.append(re.escape(pattern).replace('\\*', '.*'))
    
    for pattern in disallowed_patterns:
        if re.match(pattern, path):
            print(f"Robots.txt forbidding access to: {url}")
            return False
    
    return True

# Function to extract information using multiple selectors
def extract_with_fallback(soup, selectors):
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return element.text.strip()
    return None

# General function to extract title, date, and description
def extract_article_details(soup):
    # Define possible selectors for title, date, and description
    title_selectors = ['h1', 'h2', 'h3', '.post-title', '.article-title', '.headline']
    date_selectors = ['time', '.date', '.published-date', '.pubdate']
    description_selectors = ['meta[name="description"]', '.summary', '.description', 'p']

    title = extract_with_fallback(soup, title_selectors)
    date = extract_with_fallback(soup, date_selectors)
    description = extract_with_fallback(soup, description_selectors)

    # If the date is found in a 'time' tag, try to get the datetime attribute
    if date:
        date_tag = soup.select_one('time')
        if date_tag and 'datetime' in date_tag.attrs:
            date = date_tag['datetime']
        else:
            date = datetime.now(timezone.utc).isoformat()

    return {
        "title": title,
        "date": date,
        "description": description
    }

# Function to extract article summary details from the main page
def extract_article_summary(article, base_url):
    link = article.find('a', href=True)
    if link:
        return base_url + link['href']
    return None

# Function to crawl and parse news articles from a given website using Selenium
def crawl_news_with_selenium(base_url, news_page, article_selector):
    # Fetch and parse robots.txt
    robots_txt = fetch_robots_txt(base_url)
    
    driver.get(base_url + news_page)
    # Random delay to mimic human behavior
    time.sleep(random.uniform(5, 10))  # 5 to 10 seconds delay

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    articles = soup.select(article_selector)

    if not articles:
        print(f"No articles found on {base_url + news_page}.")
        return []
    
    news_data = []
    for article in articles:
        article_url = extract_article_summary(article, base_url)
        if article_url and is_allowed_by_robots(robots_txt, article_url):
            driver.get(article_url)
            time.sleep(random.uniform(5, 10))  # Delay to mimic user behavior
            article_soup = BeautifulSoup(driver.page_source, 'html.parser')
            full_article_data = extract_article_details(article_soup)
            if full_article_data['title'] and full_article_data['description']:
                news_data.append(full_article_data)
                
            # Random delay to avoid triggering anti-scraping mechanisms
            time.sleep(random.uniform(5, 10))  # 5 to 10 seconds delay
    
    if news_data:
        print(f"Successfully crawled {len(news_data)} articles from {base_url + news_page}.")
    else:
        print(f"Failed to crawl articles from {base_url + news_page}.")
    
    return news_data

# Function to parse and format RSS feed data
def parse_rss_feed(feed_url, decrypt=False, cryptonews=False):
    feed = feedparser.parse(feed_url)
    articles = []
    
    for entry in feed.entries:
        if decrypt:
            article = {
                "title": entry.title,
                "description": entry.description,
                "publishedAt": entry.published
            }
        elif cryptonews:
            article = {
                "title": entry.title,
                "description": entry.description
            }
        else:
            article = {
                "author": entry.get('author', 'Unknown'),
                "title": entry.title,
                "description": entry.summary,
                "url": entry.link,
                "urlToImage": entry.get('media_content', [{}])[0].get('url', None),
                "publishedAt": entry.published if 'published' in entry else datetime.now(timezone.utc).isoformat(),
                "content": entry.summary
            }
        articles.append(article)
    
    return articles

# Function to pull data from RSS feeds
def pull_rss_feeds():
    feeds = [
        {"url": "https://cointelegraph.com/rss", "decrypt": False, "cryptonews": False},
        {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "decrypt": False, "cryptonews": False},
        {"url": "https://bitcoinmagazine.com/.rss/full/", "decrypt": False, "cryptonews": False},
        {"url": "https://decrypt.co/feed", "decrypt": True, "cryptonews": False},
        {"url": "https://cryptonews.com/rss/", "decrypt": False, "cryptonews": True}
    ]
    
    all_articles = []
    
    for feed in feeds:
        articles = parse_rss_feed(feed["url"], decrypt=feed["decrypt"], cryptonews=feed["cryptonews"])
        all_articles.extend(articles)
    
    return all_articles

# Function to fetch recent news articles from News API with pagination
def fetch_news_api(api_key, base_url, query='Crypto', days=14, page_size=100, max_calls=1):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    current_page = 1
    all_articles = []

    while current_page <= max_calls:
        url = (
            f"{base_url}/everything?q={query}"
            f"&from={start_date.isoformat()}"
            f"&to={end_date.isoformat()}"
            f"&sortBy=popularity"
            f"&apiKey={api_key}"
            f"&pageSize={page_size}"
            f"&page={current_page}"
        )
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            articles_data = response.json().get('articles', [])
            
            if not articles_data:
                print(f"No more articles found after {len(all_articles)} articles.")
                break

            for article in articles_data:
                all_articles.append({
                    "author": article.get('author', 'Unknown'),
                    "title": article.get('title', 'No Title'),
                    "description": article.get('description', 'No description available.'),
                    "url": article.get('url'),
                    "urlToImage": article.get('urlToImage'),
                    "publishedAt": article.get('publishedAt', datetime.now(timezone.utc).isoformat()),
                    "content": article.get('content', 'No content available.')
                })

            print(f"Page {current_page}: Successfully fetched {len(articles_data)} articles.")
            current_page += 1
            
            # If the number of articles is less than page_size, it means there are no more articles
            if len(articles_data) < page_size:
                break

        except requests.exceptions.HTTPError as http_err:
            print(f"Error during requests to News API on page {current_page}: {http_err}")
            break

        except requests.exceptions.RequestException as req_err:
            print(f"Request error: {req_err}")
            break

    print(f"Total articles fetched from News API: {len(all_articles)}")
    return all_articles

# Function to fetch recent news articles from CryptoPanic API
# Note: Originally set 1000 max_articles but hit API limit at 800.  
def fetch_cryptopanic_news(auth_token, days=14, max_articles=600, requests_per_second=5):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    base_url = f"https://cryptopanic.com/api/v1/posts/?auth_token={auth_token}&metadata=true"
    all_articles = []
    current_page = 1

    while len(all_articles) < max_articles:
        url = f"{base_url}&page={current_page}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            articles_data = response.json().get('results', [])

            if not articles_data:
                print(f"No more articles found after fetching {len(all_articles)} articles.")
                break

            for article in articles_data:
                published_date = datetime.fromisoformat(article['published_at'][:-1]).replace(tzinfo=timezone.utc)
                if start_date <= published_date <= end_date:
                    news_data = {
                        "title": article.get('title', 'No Title'),
                        "description": article.get('metadata', {}).get('description', 'No description available.'),
                        "url": article.get('url'),
                        "publishedAt": article.get('published_at', datetime.now(timezone.utc).isoformat())
                    }
                    
                    # Optionally add image metadata if needed
                    image_url = article.get('metadata', {}).get('image', None)
                    if image_url:
                        news_data['urlToImage'] = image_url

                    all_articles.append(news_data)

                    # Stop if the max number of articles reached
                    if len(all_articles) >= max_articles:
                        print(f"Reached the maximum of {max_articles} articles.")
                        break

            print(f"Page {current_page}: Successfully fetched {len(articles_data)} articles.")
            current_page += 1

            # Respect the rate limit by sleeping between requests
            time.sleep(1 / requests_per_second)

            # Check again after processing the page to exit if we've hit the max
            if len(all_articles) >= max_articles:
                break

        except requests.exceptions.HTTPError as http_err:
            print(f"Error during requests to CryptoPanic API on page {current_page}: {http_err}")
            break

        except requests.exceptions.RequestException as req_err:
            print(f"Request error: {req_err}")
            break

    print(f"Total articles fetched from CryptoPanic API: {len(all_articles)}")
    return all_articles

# Function to save the combined data to a JSON file
def save_to_json(data, filename='crypto_news.json'):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print(f"Data saved to {filename}")

# Function to count unique titles in the JSON file and output the count
def count_unique_titles(filename='crypto_news.json'):
    with open(filename, 'r') as json_file:
        articles = json.load(json_file)
    
    unique_titles = set()
    for article in articles:
        unique_titles.add(article.get('title', 'No Title'))
    
    print(f"{len(unique_titles)} unique news pieces have been collected.")

# Main function to run the web crawler, RSS feed parser, and API fetcher
if __name__ == "__main__":
    # Crawl CoinTelegraph, CoinDesk, Bitcoin Magazine, Decrypt, and CryptoNews news pages
    telegraph_articles = crawl_news_with_selenium('https://cointelegraph.com', '/news', '.post-card-inline__title')
    coindesk_articles = crawl_news_with_selenium('https://www.coindesk.com', '/', '.card-title')
    bitcoinmagazine_articles = crawl_news_with_selenium('https://bitcoinmagazine.com', '/articles', 'h2 a')
    decrypt_articles = crawl_news_with_selenium('https://decrypt.co', '/news', '.post-title')
    cryptonews_articles = crawl_news_with_selenium('https://cryptonews.com', '/news', '.article__title')
    
    webcrawler_articles = (telegraph_articles + coindesk_articles + bitcoinmagazine_articles + 
                           decrypt_articles + cryptonews_articles)
    
    print(f"Total articles fetched from webcrawler: {len(webcrawler_articles)}")
    
    # Pull articles from RSS feeds
    rss_articles = pull_rss_feeds()
    print(f"Total articles fetched from RSS feeds: {len(rss_articles)}")
    
    # Fetch recent news articles using News API
    api_key = 'c4121bc7f94a4427ae7b44109ea518c5'  # Your provided API key
    base_url = 'https://newsapi.org/v2'  # Base URL for News API
    news_api_articles = fetch_news_api(api_key, base_url, query='Crypto', days=14, page_size=100, max_calls=1)
    
    # Fetch recent news articles using CryptoPanic API
    cryptopanic_auth_token = 'e3941a91a81a66a7773b334c8fc9d2fe0272dff4'  # Your provided CryptoPanic API key
    cryptopanic_articles = fetch_cryptopanic_news(cryptopanic_auth_token, days=14)
    
    # Combine the results
    all_articles = webcrawler_articles + rss_articles + news_api_articles + cryptopanic_articles
    
    # Save the combined articles to a JSON file
    if all_articles:
        save_to_json(all_articles)
    
    # Count and output the number of unique titles
    count_unique_titles()

    # Close the Selenium WebDriver
    driver.quit()