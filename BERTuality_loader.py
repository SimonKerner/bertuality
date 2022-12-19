import wikipediaapi
from newsapi import NewsApiClient
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

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
    content_cleaned = ""    
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
        content_cleaned += element
        
    return content_cleaned


"""
    Guardian API Call:
"""

#Auf HTTPX umstellen. statt requests
#Selectolax anstatt von BS. 


def query_api(page, from_date, to_date, order_by, query, api_key):

    response = requests.get("https://content.guardianapis.com/search?from-date="
                            + from_date + "&to-date=" + to_date + "&order-by=" + order_by 
                            +"&page=" + str(page) + "&page-size=200" + "&q=" + query +  
                            "&api-key=" + api_key)
    return response


def path_lookup_api(page, from_date, to_date, order_by, path, api_key):

    response = requests.get("https://content.guardianapis.com/" + path + "?from-date="
                            + from_date + "&to-date=" + to_date + "&page=" + str(page) 
                            + "&page-size=200&order-by=" + order_by + "&api-key=" + api_key)
    return response



def get_results_for_query(from_date, to_date, order_by, query, api_key):
    """
    Function to run a for loop for results greater than 200. 
    Calls the query_api function accordingly
    returns: a list of JSON results
    """
    json_responses = []
    response = query_api(1, from_date, to_date, order_by, query, api_key).json()
    json_responses.append(response)
    number_of_results = response['response']['total']
    if number_of_results > 200:
        for page in range(2, (round(number_of_results/200))+1):
            response = query_api(page, from_date, to_date, order_by, query, api_key).json()
            json_responses.append(response)
    return json_responses

def get_results_for_path_lookup(from_date, to_date, order_by, path, api_key):
    json_responses = []
    response = path_lookup_api(1, from_date, to_date, order_by, path, api_key).json()
    json_responses.append(response)
    number_of_results = response['response']['total']
    if number_of_results > 200:
        for page in range(2, (round(number_of_results/200))+1):
            response = path_lookup_api(page, from_date, to_date, order_by, path, api_key).json()
            json_responses.append(response)
    return json_responses


def convert_json_responses_to_df(json_responses):
    """
    Function to convert the list of json responses to a dataframe
    """
    df_results = []
    for json in json_responses:
        df = pd.json_normalize(json['response']['results'])
        df_results.append(df)
    all_df = pd.concat(df_results)
    return all_df


def guardian_call(from_date, to_date, query, path, order_by, api_key):
    
    if query != "" and path == "":
        json_responses = get_results_for_query(from_date, to_date, order_by, query, api_key)
    elif query == "" and path != "":
        json_responses = get_results_for_path_lookup(from_date, to_date, order_by, path, api_key)
    
    guardian_df = convert_json_responses_to_df(json_responses)
    return guardian_df

"""
    Create Soup:
        -and delete html-tags
"""

#further prep
def html_cleaner(raw_html):
    filtered = re.sub('<[^>]*>', "", raw_html)
    filtered = filtered.replace("Sign up now! Sign up now! Sign up now? Sign up now!", "")
    filtered = filtered.replace("""Sign up to Business TodayGet set for the working 
                                day – we'll point you to the all the business 
                                news and analysis you need every morning""", "")
    return filtered


def get_articles(link_df):
    filtered_article_list = []
    for i in link_df:
        response = requests.get(i)
        html_doc = BeautifulSoup(response.content, 'html.parser')
        p_content = html_doc.body.main.div.find_all("p")
        
        raw_text = " ".join(map(str, p_content))
        filtered_text = html_cleaner(raw_text) 
        
        if len(filtered_text) > 0:
            filtered_article_list.append(filtered_text)
            
    return filtered_article_list


def guardian_loader(from_date, to_date, query="", path="", order_by="relevance"):
    api_key = 'ec1a9d25-67dc-4f71-8313-589a96c548f9'
    
    guardian_df = guardian_call(from_date, to_date, query, path, order_by, api_key)
    
    guardian_df.drop(guardian_df.loc[guardian_df['type']=="interactive"].index, inplace=True)
    guardian_df.drop(guardian_df.loc[guardian_df['type']=="liveblog"].index, inplace=True)
    guardian_df.drop_duplicates(subset=['webTitle', 'webUrl'], inplace = True)
    
    #guardian_df['webPublicationDate'] = guardian_df['webPublicationDate'].apply(lambda x: pd.to_datetime(x))
    
    return get_articles(guardian_df["webUrl"]), guardian_df
    
    
    

    

    
