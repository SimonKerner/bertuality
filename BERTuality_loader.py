import wikipediaapi
from newsapi import NewsApiClient
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import date
from dateutil.relativedelta import relativedelta

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
      
    
def NewsAPI_loader(from_param, topic):           
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
    
    # from_param: if more than a month ago, an error occurs --> change to exactly one month ago
    if (from_param < (str(date.today() - relativedelta(months = 1)))):
        from_param = (str(date.today() - relativedelta(months = 1)))
    
    # get overview
    overview = newsapi.get_everything(q=topic, from_param=from_param, language='en', sort_by='relevancy')
    
    # get all articles
    all_articles = overview['articles']
    
    # get all headlines in list
    content = []
    
    for article in all_articles:

        description = article['description']
        #title = article['title']
        
        if len(description) > 0:    
            filtered = re.sub(r'<[^>]*>', r"", description)                      # delete html tags
            filtered = re.sub(r'<!--.*', r"", filtered)                          # also html tag
            filtered = re.sub(r'&(?:[a-z\d]+|#\d+|#x[a-f\d]+);', r"", filtered)  # delete html special enteties
            
            filtered = filtered.replace("\"", "")
            
            filtered = filtered.replace(r"…", r"...")
            filtered = re.sub(r'\s\w{0,2}\.{3}', r"", filtered)      # delete a... if len a < 2
            filtered = re.sub(r"\.{2,}", r"", filtered)              # delete if two or more "."
            filtered = re.sub(r'\-{1,}', r" ", filtered)              # delete -
            
            filtered = re.sub(r'\([^)]*\)', r"", filtered)           # delete (asdf) in parenthesis 
            filtered = re.sub(r'\[[^]]*\]', r"", filtered)           # delete [asdf] in brackets
            filtered = re.sub(r'[A-Z]*\b\/[A-Z]+', r"", filtered)    # delete GUANGZHOU/TOKYO/BANGKOK
            filtered = re.sub(r"#\w*", r"", filtered)                # delete hastag
            filtered = re.sub(r"(\d)(\,)(\d)", r"\1.\3", filtered)                 # 12,000 --> 12.000
            
            # special deletion 
            filtered = re.sub(r"[’']s", r"", filtered)              # delete apostroph s
            filtered = filtered.replace("!", ".")
            filtered = filtered.replace("?", ".")
            
            # other replacements and deletions
            filtered = filtered.replace(r"$", r" $ ")
            filtered = filtered.replace(r"%", r" % ")
            filtered = re.sub(r"[—<>|®•:“”\"]", r"", filtered)            

            #line replacements
            filtered = re.sub(r"\r?\s+\n+|\r", r".", filtered)                     # delete line break
            filtered = re.sub(r"\r?\n|\r", r".", filtered)                         # delete line break
            filtered = re.sub(r"\.{2,}", r". ", filtered)                          # if .. after line break
            filtered = re.sub(r'([a-zA-Z])(\.)([a-zA-Z])', r"\1\2 \3", filtered)   # a.a --> a. a (better sentence split)
            filtered = re.sub(r'\s{2,}', r" ", filtered)                           # if two or more whitespace
            
            filtered = filtered.replace(",.", ". ")
            
            filtered = re.sub(r'([a-z]){2}([A-Z]{1}[a-zA-Z])', r'\1 \2', filtered) # yearApple --> year Apple (but not iPhone)
            
            filtered = filtered.strip()
            
            if filtered[-1] != ".":
                filtered = filtered + "."
                
            content.append(filtered)
            
            #content.append(title)
    
    return content


"""
    Guardian API Call:
"""


def query_api(page, from_date, to_date, order_by, query, api_key):
    
    # Convert query name to correct from                                                  
    query = query.replace(" ", "%20")
    
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
    
    
    # find all articels with status = "error" and drop from list
    # else results in error in dataFrame conversion ==> [results]
    clean_json_responses = []
    for i in range(len(json_responses)):
        get_status = json_responses[i].get("response").get("status")
        if get_status == "ok":
            clean_json_responses.append(json_responses[i])

    guardian_df = convert_json_responses_to_df(clean_json_responses)
    
    return guardian_df


def get_articles(link_df):
    filtered_article_list = []
    for i in link_df:
        response = requests.get(i)
        html_doc = BeautifulSoup(response.content, 'html.parser')
        p_content = html_doc.body.main.div.find_all("p")
        
        raw_html = " ".join(map(str, p_content))
        
        filtered = re.sub('<[^>]*>', "", raw_html)
        filtered = filtered.replace("Sign up now! Sign up now! Sign up now? Sign up now!", "")     
        filtered = filtered.replace("""Sign up to Business TodayGet set for the working 
                                    day – we'll point you to the all the business 
                                    news and analysis you need every morning""", "")
        
        if len(filtered) > 0:
            filtered_article_list.append(filtered)
            
    return filtered_article_list


def guardian_loader(from_date, to_date, query="", path="", order_by="relevance"):
    api_key = 'ec1a9d25-67dc-4f71-8313-589a96c548f9'
    guardian_df = guardian_call(from_date, to_date, query, path, order_by, api_key)
    
    # filter for type == "artice" (deletes interactive, audio, liveblog, etc.)
    if len(guardian_df) == 0:                                                              
        return [], guardian_df
    
    guardian_df = guardian_df.loc[guardian_df['type']=="article"]
    
    # drop data witch contains opinion (of guardian writers), gnm-press-office, info & duplicates
    #guardian_df.drop(guardian_df.loc[guardian_df["sectionName"]=="Opinion"].index, inplace=True)
    guardian_df.drop(guardian_df.loc[guardian_df['sectionId']=="crosswords"].index, inplace=True)           
    guardian_df.drop(guardian_df.loc[guardian_df['sectionId']=="gnm-press-office"].index, inplace=True)
    guardian_df.drop(guardian_df.loc[guardian_df['sectionId']=="info"].index, inplace=True)
    guardian_df.drop_duplicates(subset=['webTitle', 'webUrl'], inplace = True)
    
    #guardian_df['webPublicationDate'] = guardian_df['webPublicationDate'].apply(lambda x: pd.to_datetime(x))
    
    return get_articles(guardian_df["webUrl"]), guardian_df


"""
    overall News loader
"""


def news_loader(from_date, topic_list):
    
    topic = " AND ".join(topic_list)                                                    
    
    # get current date
    to_date = date.today().strftime("%Y-%m-%d")
    
    # call different loaders; Wikipedia not included
    # to_date could be added to NewsAPI_loader
    news_api_query = NewsAPI_loader(from_date, topic)   # to_date is automatically the current date; from_date is never older than one month before current date (according to NewsAPI free plan)
    guardian_query, guardian_query_df = guardian_loader(from_date=from_date, to_date=to_date, query=topic)
    
    return news_api_query, guardian_query, guardian_query_df




    
    
    

    

    
