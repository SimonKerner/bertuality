import wikipediaapi
import pandas as pd
import re
from transformers import BertTokenizer, BertForMaskedLM, pipeline
      
""" 
     API/Scraper Calls for Websites
        - Wikipedia     -- working
        - BBC           -- 
        - CNN           -- 
        - Guardian      --
        - NYT           --
        - overall News  --
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

# Maxi
def filter_arr(sent_list, terms):
    result = sent_list
    for term in terms:
        result = filterForOneWord(result, term)
    result = remove_duplicates(result)
    return result

# Maxi
def filter_arr_or(sent_list, twod_list):    #twod_list = 2 dimensional list, but can also be a one-dimensional list!
    result = [] 
    
    if isinstance(twod_list[0], list):      # if the list is 2d
        for lst in twod_list:
            result += filter_arr(sent_list, lst)
    
    if isinstance(twod_list[0], str):                   # so that the given list can also be one-dimensional!
        result += filter_arr(sent_list, twod_list)
        
    result = remove_duplicates(result)
    return result

# Maxi
def remove_duplicates(sent_list):   # durch Iterieren wird die originale Reihenfolge beibehalten!
    result = []
    for i in range(len(sent_list)):
        a = sent_list[i]
        if a not in result:
            result.append(a)
    return result

           
def remove_too_long_sentences(sent_list):
    for sent in sent_list:
        encoding = tokenizer.encode(sent)
        if len((tokenizer.convert_ids_to_tokens(encoding))) > 102:    #512/5 = 102; max token length: 512, each sentence is taken 5 times
            sent_list.remove(sent)
            
    return sent_list
    

  

# overall text_filter for page loader  
def text_filter(*page_loader):
    # list holds many pages
    """
        Insert Maxi Functions WIP_v.1
    """
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

"""
    BERT Input:
        
"""
# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# BERT pre-trained
model_pre = BertForMaskedLM.from_pretrained('bert-large-uncased')


# make predictions
def make_predictions(masked_sentence, sent_sent_list):
    
    # pipeline pre-trained
    fill_mask_pipeline_pre = pipeline("fill-mask", model = model_pre, tokenizer = tokenizer)
    
    #create df
    columns = ['masked', 'input', 'input + masked', 'token1', 'score1', 'token2', 'score2', 'token3', 'score3']
    pred = pd.DataFrame(columns = columns)
    
    #fill df
    pred['input'] = sent_sent_list
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

# load pages

page_1 = wikipedia_loader("Coronavirus", "text")
page_2 = wikipedia_loader("SARS-CoV-2", "text")
page_3 = wikipedia_loader("COVID-19", "text")
page_4 = wikipedia_loader("COVID-19_pandemic", "text")
page_5 = wikipedia_loader("COVID-19_pandemic_in_Europe", "text")

# filter pages
filtered_pages = text_filter(page_1, page_2, page_3, page_4, page_5)
merged_pages = merge_pages(filtered_pages)


# put pages to dataframe 
# df_arr = pd.DataFrame(merged_pages, columns = ["sentence"])  #just to display the sent_list better
# test = pd.DataFrame(filter_arr(arr, ["Hawai"]))


# predict
pred = make_predictions("Covid is a [MASK]", filter_arr_or(merged_pages, ["Covid", "Virus", "China"]))




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





