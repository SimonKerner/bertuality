import wikipediaapi
from newsapi import NewsApiClient
import pandas as pd
import re
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import random
      
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
        """
        if element.endswith("…"):           # sometimes a word ends ends in the middle with ... ("He did this and th...") --> remove incomplete word at the end
            words = element.split()         # but most of the time the word is still complete, so the function is not used now, but might be used later
            words.pop(len(words) -1)
            element = ""
            for word in words:
                element += (word + " ")   
        """        
        element = element.replace('…', '') 
        element = element.strip()
        if (not element.endswith(".") and not element.endswith("!") and not element.endswith("?")):     #add "." at end of every sentence
            element += "."
        content_cleaned.append(element)
        
    return content_cleaned

"""
    Data Preparation
        - Split Sentences 
        - Filter Sentences with custom criteria
"""

# https://stackoverflow.com/a/31505798 - and edited
def split_into_sentences(text):
    # prefixes for split function
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"
    digits = "([0-9])"
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    
    return sentences

# Maxi        
def filterForOneWord(sent_list, term):  
    result = []
    for i in range(len(sent_list)):
        words = sent_list[i].split()
        words = [x.lower().strip() for x in words]  #lowercase and strip
        words = [x.replace(".", "") for x in words] #remove "." because otherwise "." belongs to the word
        words = [x.replace(",", "") for x in words] #remove "," because otherwise "," belongs to the word
        words = [x.replace(":", "") for x in words]
        words = [x.replace(";", "") for x in words]
        words = [x.replace(")", "") for x in words]
        words = [x.replace("(", "") for x in words]
        words = [x.replace("\"", "") for x in words]
        term = term.lower().strip()
        for word in words:
            if term in word:    #term muss nur in Wort sein: "ama" steckt in "Obamas"
                result.append(sent_list[i])
    return result  


def remove_duplicates(sent_list):   # durch Iterieren wird die originale Reihenfolge beibehalten!
    result = []
    for i in range(len(sent_list)):
        a = sent_list[i]
        if a not in result:
            result.append(a)
    return result


def filter_list(sent_list, terms):
    result = sent_list
    for term in terms:
        result = filterForOneWord(result, term)
    result = remove_duplicates(result)
    return result


def filter_list_final(sent_list, twod_list):    #twod_list = 2 dimensional list, but can also be a one-dimensional list!
    result = [] 
    
    if isinstance(twod_list[0], list):      # if the list is 2d
        for lst in twod_list:
            result += filter_list(sent_list, lst)
    
    if isinstance(twod_list[0], str):                   # so that the given list can also be one-dimensional!
        result += filter_list(sent_list, twod_list)
        
    result = remove_duplicates(result)
    return result
        
   
def remove_too_long_sentences(sent_list):
    for sent in sent_list:
        encoding = tokenizer.encode(sent)
        if len((tokenizer.convert_ids_to_tokens(encoding))) > 102:    #512/5 = 102; max token length: 512, each sentence is taken 5 times
            sent_list.remove(sent)
            
    return sent_list
    

# overall text_filter for page loader  
def sentence_converter(*page_loader):
    # list holds many pages
    
    filtered_information = []
    for page in page_loader:
        filtered_information.append(split_into_sentences(page))

    return filtered_information


# used to merge the list of pages into one list with all information
def merge_pages(filtered_pages):
    merged = []
    for page in filtered_pages:
        for text in page:
            merged.append(text)
   
    merged = remove_too_long_sentences(merged)    
    
    return merged


# create keywordlist with deletion criteria: delete random word, from left/right, longest, shortest
def keyword_creator(masked_sentence, word_deletion=True, criteria="random", min_key_words=3):
    #prep sentence
    masked_sentence = masked_sentence.replace("[MASK]", "")
    masked_sentence = masked_sentence.replace(".", "")
    masked_sentence = masked_sentence.split()

    key_list = [masked_sentence]
    
    #further sentence preperation
    if not word_deletion:
        return key_list
    
    else:
        temp_key_list = list(key_list[0])
        
        for i in range(min_key_words, len(temp_key_list)): 
    
            # delete a word from masked_sentence chosen one of many criterias
            if criteria == "random":
                
                random_word = temp_key_list[random.randint(0, len(temp_key_list)-1)]
                
                #make copy from temporary list to prevent collision and remove rand word from both lists
                temp_temp_key_list = temp_key_list.copy()
                
                temp_key_list.remove(random_word)
                temp_temp_key_list.remove(random_word)
                
                #append temp_temp to key_list
                key_list.append(temp_temp_key_list)
            
            elif criteria == "left":
                temp_temp_key_list = temp_key_list.copy()
                
                # remove first item in list
                temp_key_list.pop(0)
                temp_temp_key_list.pop(0)
                
                #append temp_temp to key_list
                key_list.append(temp_temp_key_list)
            
            elif criteria == "right":
                temp_temp_key_list = temp_key_list.copy()
                
                # remove first item in list
                temp_key_list.pop()
                temp_temp_key_list.pop()
                
                #append temp_temp to key_list
                key_list.append(temp_temp_key_list)
                
            elif criteria == "shortest":
                shortest = min(filter(None, temp_key_list), key=len)
                
                temp_temp_key_list = temp_key_list.copy()
                
                # remove first item in list
                temp_key_list.remove(shortest)
                temp_temp_key_list.remove(shortest)
                
                #append temp_temp to key_list
                key_list.append(temp_temp_key_list)
            
            elif criteria == "longest":
                longest = max(filter(None, temp_key_list), key=len)
                
                temp_temp_key_list = temp_key_list.copy()
                
                # remove first item in list
                temp_key_list.remove(longest)
                temp_temp_key_list.remove(longest)
                
                #append temp_temp to key_list
                key_list.append(temp_temp_key_list)
                
            else:
                pass
            
        return key_list


"""
    BERT :
        
"""

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# BERT pre-trained
model_pre = BertForMaskedLM.from_pretrained('bert-large-uncased')


# make predictions
def make_predictions(masked_sentence, sent_list):
    
    # pipeline pre-trained
    fill_mask_pipeline_pre = pipeline("fill-mask", model = model_pre, tokenizer = tokenizer)
    
    #create df
    columns = ['masked', 'input', 'input + masked', 'token1', 'score1', 'token2', 'score2', 'token3', 'score3']
    pred = pd.DataFrame(columns = columns)
    
    #fill df
    pred['input'] = sent_list
    pred['masked'] = masked_sentence

    #make predictions
    token1 = []
    score1 = []
    token2 = []
    score2 = []
    token3 = []
    score3 = []
    sentences = []
    
    for i in range(len(pred)):
        sent = (pred.iloc[i].iloc[1] + " " + pred.iloc[i].iloc[1] + " " + pred.iloc[i].iloc[0] + " " + pred.iloc[i].iloc[1] + " " + pred.iloc[i].iloc[1])
        sentences.append(sent)
        
        preds_i = fill_mask_pipeline_pre(sent)[:3]
    
        token1.append(preds_i[0]['token_str'].replace(" ", ""))
        score1.append(preds_i[0]['score'])
        token2.append(preds_i[1]['token_str'].replace(" ", ""))
        score2.append(preds_i[1]['score'])
        token3.append(preds_i[2]['token_str'].replace(" ", ""))
        score3.append(preds_i[2]['score'])
        
    #fill df with scores, predictions    
    pred['input + masked'] = sentences
    pred['token1'] = token1
    pred['score1'] = score1
    pred['token2'] = token2
    pred['score2'] = score2
    pred['token3'] = token3
    pred['score3'] = score3
    
    return pred


"""
    TODO:
        Delete WikiToList()
        instead give text_filtered or merged_text
        
        potential main function for data preperation 
        (load page/s, filter page/s, (optional merge pages after filter))
        
        --Create Pipeline
"""

"""
    MAIN

def main(*pages):
    try:
        # wikipedia_loader() -- insert pages -- load api call -- 
        filtered_texts = text_filter(*pages)
        merged_texts = merge_pages(filtered_texts)
    except:
        print("error occured")
    
    
    
        

if __name__ == "__main__":
    main("barack_obama", "joe_biden")
"""





