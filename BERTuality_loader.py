import wikipediaapi
from newsapi import NewsApiClient
import re

""" 
     API/Scraper Calls for Websites
        - Wikipedia     -- working
        - BBC           -- not working
        - CNN           -- not working
        - Guardian      -- working
        - NYT           -- maybe working
        - overall News  -- NewsAPI.org
"""


def wikipedia_loader(page, summary_or_text="summary", language="en"):
    # Wikipedia Import and Filter
    wiki_wiki = wikipediaapi.Wikipedia(language)
    page_py = wiki_wiki.page(page)

    if summary_or_text == "summary":           
        # get page summary
        return page_py.summary
    elif summary_or_text == "text":            
        # get all page text
        return page_py.text
    
    
def NewsAPI_loader(topic):
    """
        topic: Advanced search is supported here:
        Surround phrases with quotes (") for exact match.
        Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
        Prepend words that must not appear with a - symbol. Eg: -bitcoin
        Alternatively you can use the AND / OR / NOT keywords, and optionally group these with parenthesis. Eg: crypto AND (ethereum OR litecoin) NOT bitcoin
        The complete value for q must be URL-encoded. Max length: 500 chars.
    """
    # Init
    newsapi = NewsApiClient(api_key='be8ca28108a44c49bc15e879d0d4c5dd')
    
    # get overview
    overview = newsapi.get_everything(q=topic, language='en', sort_by='relevancy')
    
    # get all articles
    all_articles = overview['articles']
    
    # get all headlines in list
    content = [] 
    for article in all_articles:
        description = article['description']
        title = article['title']
        content.append(description)
        content.append(title)
        
    # remove distracting chars (clean sentences)
    content_cleaned = []    
    for element in content:
        element = re.sub("[\<].*?[\>]", "", element)    #remove all HTML Tags!
        element = element.replace('- Reuters','')
        element = element.replace ('- CTV News', '')
        element = element.replace ('- TASS', '')
        element = element.replace ('- EMSC', '')
        element = element.replace ('- watchdog', '')
        element = element.replace('Continue reading...', '')
        element = element.replace('.com', '')
        element = element.replace('...', '')
        if element.endswith("…"):           # sometimes a word ends ends in the middle with ... ("He did this and th...") --> remove incomplete word at the end
            words = element.split()         # but most of the time the word is still complete, so the function is not used now, but might be used later
            words.pop(len(words) -1)
            element = ""
            for word in words:
                element += (word + " ")         
        element = element.replace('…', '') 
        element = element.strip()
        if (not element.endswith(".") and not element.endswith("!") and not element.endswith("?")):     #add "." at end of every sentence
            element += "."
        content_cleaned.append(element)
        
    return content_cleaned


"""
    add GUARDIAN_LOADER
"""

