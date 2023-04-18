from dateutil.relativedelta import relativedelta
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
from datetime import date

import pandas as pd
import wikipedia
import requests
import re




"""
    data clean Up
        - prep loaded articles for better text quality
"""


def text_clean_up(str_text):
    prep = re.sub(r'<[^>]*>', r"", str_text)                                # delete html tags
    prep = re.sub(r'<!--.*', r"", prep)                                     # also html tag
    prep = re.sub(r'&(?:[a-z\d]+|#\d+|#x[a-f\d]+);', r"", prep)             # delete html special enteties
    
    prep = prep.replace("\"", "")
    
    prep = prep.replace(r"…", r"...")
    prep = re.sub(r'\s\w{0,2}\.{3}', r"", prep)                             # delete a... if len a < 2
    prep = re.sub(r"\.{2,}", r"", prep)                                     # delete if two or more "."
    prep = re.sub(r'([a-zA-Z])([-—])([a-zA-Z])', r"\1\3", prep)             # delete service-now and make to servicenow
    
    prep = re.sub(r'\-{1,}', r" ", prep)                                    # delete -
    
    prep = re.sub(r'\[[^]]*\]', r"", prep)                                  # delete [asdf] in brackets
    prep = re.sub(r'\([^)]*\)', r"", prep)                                  # delete (asdf) in parenthesis 
    prep = re.sub(r'[A-Z]*\b\/[A-Z]+', r"", prep)                           # delete GUANGZHOU/TOKYO/BANGKOK
    prep = re.sub(r"#\w*", r"", prep)                                       # delete hastag
    prep = re.sub(r"(\d)(\,)(\d)", r"\1.\3", prep)                          # 12,000 --> 12.000
    prep = re.sub(r'([A-Z])s', r'\1', prep)                                 # delete CEOs --> CEO
    
    prep = prep.replace(",", " ")
    
    
    # special deletions -contractions
    prep = re.sub(r"[’']s", r"", prep)                                      # delete apostroph s
    prep = re.sub(r"[’']d", r"", prep)
    prep = re.sub(r"[’']ll", r" will", prep)
    prep = re.sub(r"n[’']t", r" not", prep)
    prep = re.sub(r"n[’']re", r" are", prep)
    prep = re.sub(r"n[’']ve", r" have", prep)
    prep = prep.replace(r"&", r"and")
    prep = prep.replace(r" PM ", r" Prime Minister ")
    prep = prep.replace(r" U. S. ", r" United States ")
    prep = prep.replace(r" US ", r" United States ")
    prep = prep.replace(r" UK ", r" United Kingdom ")
    
    
    # other replacements and deletions
    prep = prep.replace(r"$", r" $ ")
    prep = prep.replace(r"%", r" % ")       
    prep = prep.replace(r"!", r".")
    prep = prep.replace(r"?", r".")
    prep = re.sub(r"[—<>|®•:“”\"+;=–]", r"", prep)            
    
    #line replacements
    prep = re.sub(r"\r?\s+\n+|\r", r".", prep)                              # delete line break
    prep = re.sub(r"\r?\n|\r", r".", prep)                                  # delete line break
    prep = re.sub(r"\.{2,}", r". ", prep)                                   # if .. after line break
    prep = re.sub(r'([a-zA-Z])(\.)([a-zA-Z])', r"\1\2 \3", prep)            # a.a --> a. a (better sentence split)
    prep = re.sub(r'\s{2,}', r" ", prep)                                    # if two or more whitespace
    
    prep = prep.replace(",.", ". ")
    
    prep = re.sub(r' ([a-z]){1}([A-Z]{1}[a-zA-Z])', r'\1 \2', prep)          # yearApple --> year Apple (but not iPhone)
    
    prep = prep.strip()
    
    if prep[-1] != ".":
        prep = prep + "."
    
    return prep




""" 
     API/Scraper calls for websites:

"""


"""
    wikipedia API call:
"""


def wikipedia_loader(keywords, num_pages=1):
    keywords = [i[0].upper() + i[1:] for i in keywords]
    srsearch = " ".join(keywords)
    
    response = requests.get("https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&formatversion=2"
                            + "&srsearch=" + srsearch + "&prop=extracts&srnamespace=0&srinfo=totalhits"
                            + "&srprop=wordcount%7Csnippet&srsort=relevance&srlimit=" + str(num_pages))
    
    json = response.json()
    
    titles = [i["title"] for i in json["query"]["search"]] 
    
    pages = []
    for t in titles:
        
        try: 
            pages += wikipedia.WikipediaPage(title=t).summary,
        except: 
            continue
    
    clean_pages = [text_clean_up(i) for i in pages] 
    
    return clean_pages
    



"""
    NewsAPI call:
"""


def NewsAPI_loader(from_param, to, topic):           
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
    overview = newsapi.get_everything(q=topic, from_param=from_param, to=to, language='en', sort_by='relevancy')
    
    # get all articles
    all_articles = overview['articles']
    
    # get all headlines in list
    content = []
    for article in all_articles:

        description = article['description']
        #title = article['title']
        
        if type(description) != type(None) and len(description) > 0:    
            filtered = text_clean_up(description)
            content.append(filtered)

    return content




"""
    GuardianAPI call:
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
        
        filtered = text_clean_up(raw_html)
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
        return []
    
    guardian_df = guardian_df.loc[guardian_df['type']=="article"]
    
    # drop data witch contains opinion (of guardian writers), gnm-press-office, info & duplicates
    #guardian_df.drop(guardian_df.loc[guardian_df["sectionName"]=="Opinion"].index, inplace=True)
    guardian_df.drop(guardian_df.loc[guardian_df['sectionId']=="crosswords"].index, inplace=True)           
    guardian_df.drop(guardian_df.loc[guardian_df['sectionId']=="gnm-press-office"].index, inplace=True)
    guardian_df.drop(guardian_df.loc[guardian_df['sectionId']=="info"].index, inplace=True)
    guardian_df.drop_duplicates(subset=['webTitle', 'webUrl'], inplace = True)
    
    #guardian_df['webPublicationDate'] = guardian_df['webPublicationDate'].apply(lambda x: pd.to_datetime(x))
    
    return get_articles(guardian_df["webUrl"])




"""
    overall news-loader
"""


def news_loader(from_date, to_date, keywords, use_NewsAPI=True, use_gaurdian=True, use_wikipedia=True):
    
    topic = " AND ".join(keywords)                                                    
    
    # get current date
    if to_date == "":
        to_date = date.today().strftime("%Y-%m-%d")
    
    loader = []
    if use_NewsAPI:
        try: news_api_query = NewsAPI_loader(from_date, to_date, topic)
        except: 
            print("error in news api")
            pass
        else: loader += news_api_query,
    
    
    if use_gaurdian:
        try: guardian_query = guardian_loader(from_date=from_date, to_date=to_date, query=topic)
        except: pass
        else: loader += guardian_query,
    
    
    if use_wikipedia:
        try: wikipedia_query = wikipedia_loader(keywords)
        except: pass
        else: loader += wikipedia_query,
            
        
    return loader




    
    
    

    

    
