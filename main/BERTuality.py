from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from sentence_transformers import SentenceTransformer, util
from BERTuality_loader import news_loader
from collections import Counter
from nltk import tokenize
import pandas as pd
import collections
import itertools
import time
import nltk
import re




"""
    POS & NER keyword creation:
        - prefered POS -> faster and better usability
"""


def ner_keywords(mask_sentence, ner_model="bert-base-NER"):
    tokenizer = AutoTokenizer.from_pretrained("dslim/" + ner_model)
    model = AutoModelForTokenClassification.from_pretrained("dslim/" + ner_model)
    nlp = pipeline("ner", model = model, tokenizer=tokenizer)
    mask_sentence = mask_sentence.replace(".", "")

    #learn_new_token(mask_sentence, model, tokenizer)
    ner_results = nlp(mask_sentence)
    
    return [i.get("word") for i in ner_results[0]]




# function removes duplicate tuples; ("Russias", "NNP") is removed, when there is also ("russia", "NNP")
def remove_longer_tuples(tuples_list):
    to_remove = []
    for i, t1 in enumerate(tuples_list):
        for j, t2 in enumerate(tuples_list):
            if i != j and str(t1[0]).lower() in str(t2[0]).lower():
                to_remove.append(t2)
                
    for t in to_remove:
        tuples_list.remove(t)
    return tuples_list




# create keywords by POS tagging
def pos_keywords(mask_sentence):
    sent = mask_sentence.replace("[MASK]", "")
    
    # create acronyms for better keyword creation
    sent = sent.replace("chief executive officer", "CEO")
    
    # create token and pos-tags
    token = nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(token) 
    
    # filter words: only take nouns  (NN, NNS, NNP, NNPS)
    key_pos_tags = [x for x in pos_tags if x[1] == "NN" or x[1] == "NNP" or x[1] == "NNS" or x[1] == "NNPS"]
    
    # remove duclicates while keeping the order (not with set)
    key_pos_tags_unique = []
    [key_pos_tags_unique.append(x) for x in key_pos_tags if x not in key_pos_tags_unique]
    key_pos_tags = key_pos_tags_unique
    
    # remove duplicates: "Russias" and "russia"!
    remove_longer_tuples(key_pos_tags)
    
    # filter word "Inc", because not important for meaning of the sentence!
    key_pos_tags = [x for x in key_pos_tags if x[0] != "Inc" and x[0] != "Inc."]
    key_pos_tags = [x for x in key_pos_tags if x[0] != "profession" ]
    key_pos_tags = [x for x in key_pos_tags if x[0] != "leader" ]
    
    # if key_pos_tags_filtered contains more than 4 keywords, remove some!
    if len(key_pos_tags) > 4:
        
        # remove the keywords 'United' and 'States', because they are not important
        key_pos_tags = [x for x in key_pos_tags if x[0] != 'United' and x[0] != 'States']
           
        # if still longer than 4, remove all excess NN and NNS     
        if len(key_pos_tags) > 4:
            key_pos_tags_shortened = key_pos_tags[0:4]
            key_pos_tags_excess = key_pos_tags[4:]
            
            # after keyword 4, only allow keywords with POS Tags NNP or NNPS; these are the names and names are very inportant!
            key_pos_tags_excess_shortened = [x for x in key_pos_tags_excess if x[1] == 'NNP' or x[1] == 'NNPS']
            key_pos_tags_shortened += key_pos_tags_excess_shortened
            
            return [x[0] for x in key_pos_tags_shortened]
        
        else:
            return [x[0] for x in key_pos_tags]
        
    #key_words = [x[0] for x in key_pos_tags]           
    else:
        return [x[0] for x in key_pos_tags]




"""
    DATA_LOADER_Query
        - load pages from NewsApi, Guardian and Wikipedia
"""


# main query for input data --> returns dict, but changeable to whatever
def loader_query(mask_sentence, from_date, to_date, use_NewsAPI, use_guardian, use_wikipedia):
    
    key_words = pos_keywords(mask_sentence)
    loader_query = news_loader(from_date, to_date, key_words, use_NewsAPI)
    
    loader = {"01_mask_sentence": mask_sentence,
              "02_keywords": key_words,
              "03_from_date": from_date,
              "04_to_date": to_date,
              "05_use_NewsAPI": use_NewsAPI,
              "06_use_guardian": use_guardian,
              "07_use_wikipedia": use_wikipedia,
              "08_loader_query": loader_query}              
    
    return loader




"""
    SPLIT_Query
        - NLTK splitter, converts textual blocks into sentences
"""  


def nltk_sentence_split(page_loader):
    filtered_information = []
    for page in page_loader:
        if type(page) == list:
            for text in page:
                filtered_information.append(tokenize.sent_tokenize(text))
        else:
            filtered_information.append(tokenize.sent_tokenize(page))

    return filtered_information




"""
    MERGE_Query
        - merge all Loader Sentences into a flat 1-d List
"""


# used to merge the list of pages into one list with all information
def merge_pages(filtered_pages):
    merged = []
    for page in filtered_pages:
        for text in page:
            if len(text) > 15:                                                      
                merged.append(text)
    return merged
    



"""
    EXTRACTION_Query 
        - search keywords in sentences and append to extractiton
        - also able to create subsets out of keywords
        -> may create duplicate sentences when True
"""


def remove_duplicates(input_sentences):   
    result = []
    for i in range(len(input_sentences)):
        a = input_sentences[i]
        if a not in result:
            result.append(a)
    return result




def remove_too_long_sentences(input_sentences, tokenizer):
    short_input_sentences = []
    for i in range(len(input_sentences)):
        
        try:
            encoding = tokenizer.encode(input_sentences[i])
        except:
            continue
        else:
            #512/5 = 102; max token length: 512, each sentence is taken 5 times
            if len(encoding) <= 102:    
                short_input_sentences.append(input_sentences[i])
            
    return short_input_sentences




def filterForOneWord(input_sentences, term):  
    result = []
    for i in range(len(input_sentences)):
        words = input_sentences[i].split()
        words = [x.lower().strip() for x in words] 
        words = [x.replace(".", "") for x in words]
        words = [x.replace(",", "") for x in words] 
        words = [x.replace(":", "") for x in words]
        words = [x.replace(";", "") for x in words]
        words = [x.replace(")", "") for x in words]
        words = [x.replace("(", "") for x in words]
        words = [x.replace("\"", "") for x in words]
        term = term.lower().strip()
        for word in words:
            if term in word:   
                result.append(input_sentences[i])
    return result  




def filter_list(input_sentences, terms):
    result = input_sentences
    for term in terms:
        result = filterForOneWord(result, term)
    result = remove_duplicates(result)
    return result




def filter_list_final(input_sentences, twod_list, tokenizer, duplicates):   
    result = [] 
    
    if isinstance(twod_list[0], list):     
        for lst in twod_list:
            result += filter_list(input_sentences, lst)
    
    if isinstance(twod_list[0], str):                  
        result += filter_list(input_sentences, twod_list)
        
    if duplicates==False: result = remove_duplicates(result)
    
    result = remove_too_long_sentences(result, tokenizer)
    
    return result




# function to create subsets from keywords, used in filter_for_keyword_subsets
def create_subsets(mylist, subset_size):
    if subset_size > len(mylist):
        return mylist
    subsets = [list(comb) for comb in itertools.combinations(mylist, subset_size)]
    
    return subsets




def filter_for_keyword_subsets(input_sentences, keywords, tokenizer, subset_size, duplicates):
    keyword_subsets = create_subsets(keywords, subset_size)
    
    extraction = filter_list_final(input_sentences, keyword_subsets, tokenizer, duplicates)
    extraction_sorted = [item for items, c in Counter(extraction).most_common() for item in [items] * c]
    
    return extraction_sorted
    



"""
    SIMILARITY_Query
        - find most similar sentences in camparison to mask_sentence
"""


# used to extract better sentences out of other querys --> score depends on similarity of sentence to mask_sentence
def similarity_filter(mask_sentence, query, sim_score, return_tuples=False):
    
    if sim_score == 0: return query
    if len(query) == 0: return query
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    #Compute embedding for both lists
    embedding_1 = model.encode([mask_sentence], convert_to_tensor=True)
    embedding_2 = model.encode(query, convert_to_tensor=True)
    
    if return_tuples==False:
        # calculate cos_sim score and return tuple with testet sentence
        cos_sim = [(util.pytorch_cos_sim(embedding_1[0], embedding_2[i]), query[i]) for i in range(len(embedding_2))]
        
        # delete [1] to get tuple with score and sentence --> returnes sentence if higher than sim_score
        results = [cos_sim[i][1] for i in range(len(cos_sim)) if float(cos_sim[i][0]) >= sim_score]
    else:
        cos_sim = [(float(util.pytorch_cos_sim(embedding_1[0], embedding_2[i])), query[i]) for i in range(len(embedding_2))]
        results = [cos_sim[i] for i in range(len(cos_sim)) if float(cos_sim[i][0]) >= sim_score]
    
    return results




"""
    FOCUS_Query
        - search first an last Keyword in Sentence and remove rest of Words
        - Option to add padding value to focus_query
        
        - a little deprecated: possible improvements --> use better_tokenizer 
"""


def keyword_focus(input_sentences, key_words, padding):
    filtered_input = []
    
    if len(input_sentences) == 0:
        return filtered_input
    
    for sentence in input_sentences:
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
                    positions.append(tokens.index(word))            # find exact same word
                except ValueError:
                    if case == 0:                                   # word too long obama is part of obama-era
                        found = False
                        for i in tokens:  
                            if word in i:
                                found = True
                                tokens[tokens.index(i)] = word      # obama-era =set_to> obama
                                break
                        if not found:
                            case = 1
                    
                    elif case == 1:                                 # word is too short =shorten_keyword>
                        if len(word) > 0:      
                            word = word[:-1]
                        else:
                            error = False                           # word is not in sentence
                else:
                    error = False

        if len(positions) != 0:
            
            #add padding to positions
            left_pos, right_pos = min(positions), max(positions)
            left_pad, right_pad = left_pos - padding, right_pos + padding
            
            # check if pad is out of bound and if set to max available value
            if left_pad < 0: left_pad = 0
            if right_pad > len(tokens): right_pad = len(tokens)
                
            focus = tokens[left_pad : right_pad + 1]
            focus_input = " ".join(focus)
            
            if focus_input[-1] != ".":
                focus_input = focus_input + "."
            
            filtered_input.append(focus_input)
            
        else:
            filtered_input.append(sentence)
    
    return filtered_input




"""
    DATAPREP_Query
"""


# main query for input data --> returns dict, returns completed list with query
def dataprep_query(mask_sentence, loader_query, tokenizer, 
                   subset_size, sim_score, focus_padding, 
                   extraction, similarity, focus, duplicates):
    
    key_words = pos_keywords(mask_sentence)
    
    split_query = nltk_sentence_split(loader_query)
    
    # is equal to merged_query
    dataprep = merge_pages(split_query)
    
    # extraction_query 
    if extraction == True:
        dataprep = filter_for_keyword_subsets(dataprep, key_words, tokenizer, subset_size, duplicates)
    
    # similarity_query
    if similarity == True and extraction == True:
        dataprep = similarity_filter(mask_sentence, dataprep, sim_score)
    elif similarity == True and extraction == False:
        dataprep = similarity_filter(mask_sentence, dataprep, sim_score)
    
    # focus_query
    if focus == True:
        dataprep = keyword_focus(dataprep, key_words, focus_padding)
    
    return dataprep




"""
    better tokenizer + detokenizer 
        - better detectability for word piece tokens
        - able to find hidden ones (wp inside of wp)
"""


# flatten multidimensional list to 1-d
def flatten(l):
    for sub in l:
        if isinstance(sub, collections.abc.Iterable) and not isinstance(sub, (str, bytes)):
            yield from flatten(sub)
        else:
            yield sub




# create lists of further tokenized token
def find_hidden_word_pieces(word_piece, tokenizer):
    
    hidden_wp = []
    token = tokenizer.tokenize(word_piece)
    
    for tpos in range(len(token)):
        if token[tpos] == "#": pass
        elif token[tpos][:2] == "##":
            hidden_wp += find_hidden_word_pieces(token[tpos], tokenizer),
        else: hidden_wp += token[tpos],
        
    return hidden_wp
        



# new tokenizer - able to find hidden word_pieces 
def better_tokenizer(sentence, tokenizer):
    
    better_tokens = []
    tokens = tokenizer.tokenize(sentence)
    
    #further tokenization
    for t in tokens:
        
        # check if wordpiece has further wp_tokens
        if t[:2] == "##":
            wp_tokens = find_hidden_word_pieces(t, tokenizer)
            wp_tokens = list(flatten(wp_tokens))
            wp_tokens = ["##" + wp for wp in wp_tokens]
            better_tokens += wp_tokens,
        else: better_tokens += t,
        
    return list(flatten(better_tokens))




def detokenizer(token_list, tokenizer):
    detoken = tokenizer.convert_tokens_to_string(token_list)
    detoken = re.sub(r'( )([\'\.])( ){0,}', r'\2', detoken)
    detoken = re.sub(r'( )([,])( )', r'\2 ', detoken)
    detoken = re.sub(r'([a-zA-Z])(\.)([a-zA-Z])', r"\1\2 \3", detoken)
    return detoken




"""
    OR_PRED_QUERY -- Original Projekt: BERT_Prediction Pipeline:  
        - uses fill_mask_pipeline
"""


def make_predictions(masked_sentence, input_sentences, model, tokenizer, max_input=None):
    
    if len(input_sentences) == 0: return
    if max_input != None: input_sentences = input_sentences[:max_input]
    
    # pipeline pre-trained
    fill_mask_pipeline_pre = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    
    #create df
    columns = ['masked', 'input', 'input + masked', 'token1', 'score1', 'token2', 'score2', 'token3', 'score3']
    pred = pd.DataFrame(columns = columns)
    
    #fill df
    pred['input'] = input_sentences
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




# main query for input data --> returns dict, but changeable to whatever
def query_pipeline(sample, from_date, to_date, tokenizer, subset_size, sim_score, focus_padding, use_NewsAPI = True):
    
    key_words = pos_keywords(sample)
    
    loader_query = news_loader(from_date, to_date, key_words, use_NewsAPI)
    split_query = nltk_sentence_split(loader_query)
    merged_query = merge_pages(split_query)
    extraction_query = filter_for_keyword_subsets(merged_query, key_words, tokenizer, subset_size)
    similarity_query = similarity_filter(sample, extraction_query, sim_score)
    focus_query = keyword_focus(similarity_query, key_words, focus_padding)
    
    query_dict = {"01_sample": sample,
                  "02_keywords": key_words,
                  "03_split_query": split_query,
                  "04_merged_query": merged_query,
                  "05_extraction_query": extraction_query,
                  "06_similarity_query": similarity_query,
                  "07_focus_query": focus_query}              
    
    return query_dict




"""
    WP_PRED_QUERY -- WordPiece-BERT_Prediction Pipeline:  
        - is a better prediciton pipeline than original one
        - able to predict every kind of word
"""

# THIS VERSTION IS DEPRECATED # THIS VERSTION IS DEPRECATED # THIS VERSTION IS DEPRECATED 


def create_expand_sentences(tokens, wp_position, tokenizer):
    word_piece_input = []
    wp_position.sort()
    
    sliced_out = tokens[0 : wp_position[0]] + tokens[wp_position[-1] + 1:]
    
    for p in wp_position:
        
        temp_sliced_out = sliced_out[:]
        temp_sliced_out.insert(wp_position[0], tokens[p].replace("##", ""))
        
        word_piece_input += detokenizer(temp_sliced_out, tokenizer), 
        
    return word_piece_input




def word_piece_temp_df_pred(mask_sentence, tokens, wp_position, model, tokenizer):
    
    ## create sentences - for a whole word piece word
    
    # 1. create new sentence
    word_piece_input = create_expand_sentences(tokens, wp_position, tokenizer)
        
    # 2. make prediciton to new sentences
    piece_pred = make_predictions(mask_sentence, word_piece_input, model, tokenizer)
    
    # 3. get word and score of word --append to list
    concat_word = ""
    if piece_pred["token1"].nunique() == 1:
        concat_word = piece_pred["token1"][0]
    else:
        for p in piece_pred["token1"]:
            concat_word += p
    
    # 4. get list - concat word and get average score
    temp_df = pd.DataFrame([{'masked': mask_sentence, 
                             'input': piece_pred["input"],
                             "input + masked": piece_pred["input + masked"],
                             'token1': concat_word, 
                             'score1': piece_pred["score1"].mean()}])
    return temp_df




def word_piece_prediction(mask_sentence, input_sentences, model, tokenizer, max_input=None, combine=True, threshold=None):
    start_time = time.perf_counter()
    # find index of ali ##ba ##ba
    # rule word bevore ## is always part of whole word
    # find whole word pieces
    if max_input != None: input_sentences = input_sentences[:max_input]
    
    print(mask_sentence)
    print("\n    Starting WordPiece Prediction:")
    print("    Input Size:", len(input_sentences))
    
    #output storage
    wp_results = pd.DataFrame(columns=["masked", "input", "input + masked", "token1", "score1"])
    
    status = 0
    sen_size = len(input_sentences)
    #get input sentences
    for sentence in input_sentences:
        
        #tokens = tokenizer.tokenize(sentence)
        tokens = better_tokenizer(sentence, tokenizer)
        
        # second path --> sentence has word pieces -- can be found with ## - wp-pred
        wp_position = []
        wp_temp_df = None
        wp_append_start = False
        
        for t_pos in range(len(tokens)):
            # find count of ##pieces
            #word = tokens[t_pos]
            
            if "##" in tokens[t_pos]:
                wp_position += t_pos,    # append first occurence  
                
                if wp_append_start == False:
                    wp_position += (t_pos - 1),    # append start of word piece word (==equal to one word) 
                    wp_append_start = True
                    
            elif wp_append_start == True and (("##" not in tokens[t_pos]) or (t_pos == len(tokens) - 1)): # word piece word is over --> make pred for it
                
                # get results of prediction in DataFrame
                temp_df = word_piece_temp_df_pred(mask_sentence, tokens, wp_position, model, tokenizer)
                
                # concat results of temp_df to results 
                # option True is able to use threshold speedup which breaks loop, when possible word is found
                # possible word = when first word exceeds threshold
                if combine == True:
                    wp_temp_df = pd.concat([wp_temp_df, temp_df], ignore_index=True)

                    if type(threshold) == float and temp_df["score1"][0] >= threshold: 
                        break
             
                # else is equal to combine False - no threshold --> append everything
                else: 
                    wp_results = pd.concat([wp_results, temp_df], ignore_index=True)
                
                # 5. reset counter and lists for next word piece
                wp_position.clear()
                wp_append_start = False
        
        # if size wp_results == 0 --> no wp_word was found (all tokens known) --> use normal make_pred pipe for prediction
        if type(wp_temp_df) == type(None):
            # get results of prediction in DataFrame
            temp_df = make_predictions(mask_sentence, [sentence], model, tokenizer)
            
            # concat results of
            wp_results = pd.concat([wp_results, temp_df], ignore_index=True)
            
        # when multiple word piece words are in one sentence,
        # this sentence will be predicted on multiple times
        # pred option True will take only the best token results of all sentences          
        elif combine == True:
                
            index = wp_temp_df["score1"].idxmax()
            best_result = wp_temp_df.iloc[index]
            wp_results = pd.concat([wp_results, best_result.to_frame().T], ignore_index=True)
        
        # status prints
        status+=1
        print("    Progress -->", round(status/sen_size*100, 2), "%")
    end_time = time.perf_counter()
    print(f"    WPP-Performance: {end_time - start_time:0.4f} seconds")
          
    #return list of sentences with  --> return predicted sentences from focus
    return wp_results


######################################################################################################################
######################################################################################################################
############################################ WORD PIECE PREDICTION v2 ################################################
######################################################################################################################
######################################################################################################################


def make_predictions_v2(masked_sentence, input_sentences, model, tokenizer, max_input=None, only_target_token=None, targets=None):
    
    if len(input_sentences) == 0: return
    if max_input != None: input_sentences = input_sentences[:max_input]
    
    # pipeline pre-trained
    fill_mask_pipeline_pre = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    
    #create df
    columns = ['masked', 'input', 'input + masked', 'token1', 'score1', 'token2', 'score2', 'token3', 'score3']
    pred = pd.DataFrame(columns = columns)
    
    #fill df
    pred['input'] = input_sentences
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
        
        if only_target_token == False or only_target_token == None:
            preds_i = fill_mask_pipeline_pre(sent)[:3] 
        elif only_target_token == True:
            preds_i = fill_mask_pipeline_pre(sent, targets=targets)[:3]
    
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




def wp_find_positions(tokens):
    wp_append_start = False
    every_wp_position = []
    wp_position = []
    
    for t_pos in range(len(tokens)):
        # find count of ##pieces
        #word = tokens[t_pos]
        
        if "##" in tokens[t_pos]:
            wp_position += t_pos,    # append first occurence  
            
            if wp_append_start == False:
                wp_position += (t_pos - 1),    # append start of word piece word (==equal to one word) 
                wp_append_start = True
                
        elif wp_append_start == True and (("##" not in tokens[t_pos]) or (t_pos == len(tokens) - 1)): # word piece word is over --> make pred for it
            temp_wp_position = wp_position.copy()
            temp_wp_position.sort()
            every_wp_position += temp_wp_position,
            wp_position.clear()
            wp_append_start = False
        
            
    return every_wp_position




def wp_input_sentence_creator(tokens, every_wp_position, tokenizer):
    int_wp_positions = [j for i in every_wp_position for j in i]
    int_basic_sentence = [j for j in range(len(tokens)) if j not in int_wp_positions]
    
    # append to wpw_input_sentences basic sentence on first place without WordPieces
    wpw_input_sentences = [] # [detokenizer([tokens[t] for t in int_basic_sentence], tokenizer)]
    for wpw in every_wp_position:   
        temp_wpw_input_sentences = []  
        for wp in wpw:
            sentence_int = int_basic_sentence + [wp]
            sentence_int.sort()
            
            detoken = detokenizer([tokens[t].replace("##", "") for t in sentence_int], tokenizer)
            temp_wpw_input_sentences += detoken,
        
        wpw_input_sentences += temp_wpw_input_sentences,
        
    return wpw_input_sentences




def wp_prediction_pred_v2(mask_sentence, inputs, model, tokenizer, max_input, only_target_token, target_token):
    
    if only_target_token == True:
        # delete duplicates
        target_token = list(dict.fromkeys(target_token))
        target_token = [t.replace("##", "") for t in target_token]
        target_token = [t for t in target_token if t != "."]
    
    
    piece_pred = make_predictions_v2(mask_sentence, inputs, model, tokenizer, max_input, only_target_token, targets=target_token)
    
    concat_word = ""
    if piece_pred["token1"].nunique() == 1:
        concat_word = piece_pred["token1"][0]
    else:
        for p in piece_pred["token1"]:
            concat_word += p
    
    temp_df = pd.DataFrame([{'masked': mask_sentence, 
                             'input': piece_pred["input"],
                             "input + masked": piece_pred["input + masked"],
                             'token1': concat_word, 
                             'score1': piece_pred["score1"].mean()}])
    
    return temp_df




def word_piece_prediction_v2(mask_sentence, input_sentences, model, tokenizer, max_input=None, combine=True, threshold=None, only_target_token=None):
    start_time = time.perf_counter()
    # find index of ali ##ba ##ba
    # rule word bevore ## is always part of whole word
    # find whole word pieces
    if max_input != None: input_sentences = input_sentences[:max_input]
    
    print(mask_sentence)
    print("\n    Starting WordPiece Prediction:")
    print("    Input Size:", len(input_sentences))
    
    #output storage
    wp_results = pd.DataFrame(columns=["masked", "input", "input + masked", "token1", "score1"])
    
    status = 0
    sen_size = len(input_sentences)
    #get input sentences
    for sentence in input_sentences:
        
        wp_temp_df = None
        
        #tokens = tokenizer.tokenize(sentence)
        tokens = better_tokenizer(sentence, tokenizer)
        
        """
               new second WP-PRED-Module
        """
        # find word-piece-positions
        every_wp_position = wp_find_positions(tokens)
         
        if len(every_wp_position) != 0:
            wpw_input_sentences = wp_input_sentence_creator(tokens, every_wp_position, tokenizer)      
            
            for inputs in wpw_input_sentences:
                
                target_token = None
                if only_target_token == True:
                    target_token = [better_tokenizer(i, tokenizer) for i in inputs]
                    target_token = [j for i in target_token for j in i]
                
                temp_df = wp_prediction_pred_v2(mask_sentence, inputs, model, tokenizer, max_input, only_target_token, target_token)
                
                if combine == True:
                    wp_temp_df = pd.concat([wp_temp_df, temp_df], ignore_index=True)
    
                    if type(threshold) == float and temp_df["score1"][0] >= threshold: 
                        break
             
                # else is equal to combine False - no threshold --> append everything
                else: 
                    wp_results = pd.concat([wp_results, temp_df], ignore_index=True)
            
        # if size wp_results == 0 --> no wp_word was found (all tokens known) --> use normal make_pred pipe for prediction
        if type(wp_temp_df) == type(None):
            
            target_token = None
            if only_target_token == True:
                target_token = list(dict.fromkeys(tokens))
                target_token = [t.replace("##", "") for t in tokens]
            
            # get results of prediction in DataFrame
            temp_df = make_predictions_v2(mask_sentence, [sentence], model, tokenizer, max_input, only_target_token, target_token)
            
            # concat results of
            wp_results = pd.concat([wp_results, temp_df], ignore_index=True)
            
        # when multiple word piece words are in one sentence,
        # this sentence will be predicted on multiple times
        # pred option True will take only the best token results of all sentences          
        elif combine == True:
                
            index = wp_temp_df["score1"].idxmax()
            best_result = wp_temp_df.iloc[index]
            wp_results = pd.concat([wp_results, best_result.to_frame().T], ignore_index=True)
        
        # status prints
        status+=1
        print("    Progress -->", round(status/sen_size*100, 2), "%")
    end_time = time.perf_counter()
    print(f"    WPP-Performance: {end_time - start_time:0.4f} seconds")
          
    #return list of sentences with  --> return predicted sentences from focus
    return wp_results

######################################################################################################################
######################################################################################################################
############################################ WORD PIECE PREDICTION v2 ################################################
######################################################################################################################
######################################################################################################################




"""
    get prediction result summary 
"""


def simple_pred_results(pred_query):
    
    if pred_query is None: return 
    
    results = pd.DataFrame(columns=["Token", "Frequency", "max_score", "min_score", "mean_score", "sum_up_score", "?new_ranking?"])
    results["Token"] = pred_query["token1"].unique()
    results["Frequency"] = results["Token"].map(pred_query["token1"].value_counts())
    results["max_score"] = results["Token"].map(pred_query.groupby("token1")["score1"].max())
    results["min_score"] = results["Token"].map(pred_query.groupby("token1")["score1"].min())
    results["mean_score"] = results["Token"].map(pred_query.groupby("token1")["score1"].mean())
    results["sum_up_score"] = results["Token"].map(pred_query.groupby("token1")["score1"].sum())
    results["?new_ranking?"] = results.index.map(((results.max_score - results.min_score) * results.mean_score)/(results.max_score - results.min_score))
    results = results.sort_values(by=["sum_up_score"], ascending=False, ignore_index=True) # was sum_up_score
    
    return results




"""
    bertuality_main_func
"""


def load_default_config():
    default_config = {
        'model': r'model',
        'tokenizer': r'tokenizer',
        'from_date': '2023-03-01',
        'to_date': '',
        'use_NewsAPI': True, 
        'use_guardian': False, 
        'use_wikipedia': True, 
        'subset_size': 2,
        'sim_score': 0.3,
        'focus_padding': 6,
        'duplicates': False,
        'extraction': True,
        'similarity': True,
        'focus': True,
        'max_input': 30,
        'threshold': 0.9,
        'only_target_token': True
        }
    return default_config




def bertuality(mask_sentence, config=None, return_values=False):
    
    try:
        if type(config) == type(None):
            config = load_default_config()
        elif type(config) == type(dict):
            config = config
        else: raise ValueError
        
    except ValueError: 
        print("No valid config found!")
        print("Set config to 'None' for default values")
        
    else:
        model = BertForMaskedLM.from_pretrained(config["model"])
        tokenizer = BertTokenizer.from_pretrained(config["tokenizer"])
        
        print()
        print("Step 1: Load config --> Done")
        
        data = loader_query(mask_sentence, 
                            config["from_date"], 
                            config["to_date"], 
                            config["use_NewsAPI"], 
                            config["use_guardian"], 
                            config["use_wikipedia"])
        
        print("Step 2: Load latest data --> Done")
        
        dataprep = dataprep_query(mask_sentence, 
                                  data['08_loader_query'], 
                                  tokenizer, 
                                  config['subset_size'], 
                                  config['sim_score'], 
                                  config['focus_padding'], 
                                  config['extraction'], 
                                  config['similarity'], 
                                  config['focus'], 
                                  config['duplicates'])
        
        print("Step 3: Prepare pata --> Done")
        print("Step 4: Start Prediction:\n")
        
        prediction = word_piece_prediction_v2(mask_sentence, 
                                           dataprep, 
                                           model, 
                                           tokenizer, 
                                           max_input=config['max_input'],
                                           threshold=config['threshold'],
                                           only_target_token=config['only_target_token'])
        
        simple_pred = simple_pred_results(prediction)

        pred_sentence = mask_sentence.replace("[MASK]", simple_pred["Token"][0].capitalize())
        print("\n" + pred_sentence)
        
        if return_values == True:
            return [mask_sentence, config, data, dataprep, prediction, simple_pred, pred_sentence]
                
    
  








