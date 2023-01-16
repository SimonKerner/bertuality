from transformers import BertTokenizer, BertForMaskedLM, pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import random
import re
import nltk

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
        
   
def remove_too_long_sentences(sent_list, tokenizer):
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
        if type(page) == list:
            for text in page:
                filtered_information.append(split_into_sentences(text))
        else:
            filtered_information.append(split_into_sentences(page))

    return filtered_information


# used to merge the list of pages into one list with all information
def merge_pages(filtered_pages, tokenizer):
    merged = []
    for page in filtered_pages:
        for text in page:
            merged.append(text)
   
    merged = remove_too_long_sentences(merged, tokenizer)    
    
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
    
            # delete a word from masked_sentence chosen one of many criteria
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


# further input sentence preparation
def spec_char_deletion(input_sentences):
    clean_sentences = []
    for i in input_sentences:
        clean = re.sub("r'\W+'",' ', i)
        clean = clean.replace(".", "")
        clean = clean.replace(",", "")
        clean = clean.replace(":", "")
        clean_sentences.append(clean)
    return clean_sentences


def keyword_focus(input_sentences, key_words, padding=0):
    filtered_input = []
    
    if len(input_sentences) == 0:
        return filtered_input
    
    clean_input = spec_char_deletion(input_sentences)  #relevance?
    
    for sentence in clean_input:
        sentence = sentence.lower()
        tokens = sentence.split()
        
        #find positions of key_words
        positions = []
        for word in key_words: 
            word = word.lower()
            
            error = True
            case = 0
            while(error):
                try:
                    positions.append(tokens.index(word))          # find exact same word
                except ValueError:
                    if case == 0:           #word too long obama is part of obama-era
                        found = False
                        for i in tokens:  
                            if word in i:
                                found = True
                                tokens[tokens.index(i)] = word          # obama-era =set_to> obama
                                break
                        if not found:
                            case = 1
                    
                    elif case == 1:                                           # word is too short =shorten_keyword>
                        if len(word) > 0:      
                            word = word[:-1]
                        else:
                            error = False # word is not in sentence
                else:
                    #print(word) # word found in list
                    error = False
        #print()
        
        #add padding to positions
        left_pos, right_pos = min(positions), max(positions)
        left_pad, right_pad = left_pos - padding, right_pos + padding
        
        # check if pad is out of bound and if set to max available value
        if left_pad < 0:
            left_pad = 0
        if right_pad > len(tokens):
            right_pad = len(tokens)
            
        focus = tokens[left_pad : right_pad + 1]
        filtered_input.append(" ".join(focus))
        filtered_input[-1] = filtered_input[-1] + "."
    return filtered_input


"""
    NER :
        - output with keywords from sample sentence
"""

def ner_keywords(sample, ner_model="bert-base-NER"):
    tokenizer = AutoTokenizer.from_pretrained("dslim/" + ner_model)
    model = AutoModelForTokenClassification.from_pretrained("dslim/" + ner_model)
    nlp = pipeline("ner", model = model, tokenizer=tokenizer)
    sample[0] = sample[0].replace(".", "")

    learn_new_token(sample, model, tokenizer)
    ner_results = nlp(sample)
    
    return [i.get("word") for i in ner_results[0]]


# create keywords by POS tagging (= Part of speech)
def pos_keywords(sample):
    sample = sample[0].replace("[MASK]", "")
    # create token and pos-tags
    token = nltk.word_tokenize(sample)
    pos_tags = nltk.pos_tag(token) 
    # filter words: only take nouns  (NN, NNP, NNS, NNPS)
    key_pos_tags = [x for x in pos_tags if x[1] == "NN" or x[1] == "NNP" or x[1] == "NNS" or x[1] == "NNPS"]
    
    key_words = [x[0] for x in key_pos_tags]
    return key_words

"""
    BERT :
        
"""
"""
# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# model: BERT pre-trained
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
"""


# make predictions
def make_predictions(masked_sentence, sent_list, model, tokenizer):
    
    # pipeline pre-trained
    fill_mask_pipeline_pre = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    
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


def load_actuality_dataset():
    # load dataset
    actuality_dataset = pd.read_excel (r'DS_Aktualitätsprüfung.xlsx', sheet_name = 'Gesamt Daten')
    # filter datset
    actuality_dataset = actuality_dataset[(actuality_dataset['Quelle'] == 'Eigenkreation') & 
                                          ((actuality_dataset['Akt.-Ind./Wortart'] == 'Subjekt') |
                                          (actuality_dataset['Akt.-Ind./Wortart'] == 'Objekt') |
                                          (actuality_dataset['Akt.-Ind./Wortart'] == 'Zahl/Objekt'))]
    # reset index, delete unnecessary columns
    actuality_dataset = actuality_dataset.reset_index()
    del actuality_dataset['index']
    del actuality_dataset['Unnamed: 7']
    del actuality_dataset['Unnamed: 8']
    del actuality_dataset['Unnamed: 9']
    del actuality_dataset['Unnamed: 10']
    
    return actuality_dataset


# learn all gold token in given dataset
def learn_all_new_gold_token(dataset, model, tokenizer):
    # create list of gold token from given dataset
    gold_token = list(dataset['Gold'])
    
    # add new token to tokenizer
    num_added_toks = tokenizer.add_tokens(gold_token)
    model.resize_token_embeddings(len(tokenizer)) #resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer
  
    
# learn one new gold token in sample, sample has to look like this: ['sentence', 'gold token']    
def learn_new_token(sample, model, tokenizer):
    # create list of all token form given sample ['sentence', 'gold token']
    gold_token = [sample[1]]
    other_token = sample[0].split()
    new_token = gold_token + other_token #1st new token gets id 30522
    
    # add new token to tokenizer
    num_added_toks = tokenizer.add_tokens(new_token)
    model.resize_token_embeddings(len(tokenizer)) #resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer
    







