from transformers import BertTokenizer, BertForMaskedLM, pipeline
from BERTuality_loader import wikipedia_loader
from BERTuality_loader import NewsAPI_loader
from BERTuality_loader import guardian_loader
from BERTuality_loader import news_loader

from BERTuality import ner_keywords
from BERTuality import pos_keywords
from BERTuality import nltk_sentence_split
from BERTuality import merge_pages
from BERTuality import filter_list_final
from BERTuality import filter_for_keyword_subsets
from BERTuality import keyword_focus
from BERTuality import query_pipeline

from BERTuality import make_predictions
from BERTuality import word_piece_prediction
from BERTuality import simple_pred_results

from BERTuality import load_actuality_dataset
from BERTuality import learn_new_token
from BERTuality import learn_all_new_gold_token

from BERTuality import automatic_dataset_pred
from BERTuality import automatic_dataset_pred_v2
from BERTuality import load_actuality_dataset
from BERTuality import scoring
import pickle


# Test learn_all_new_token (for gold token in dataset)
"""
actuality_dataset = load_actuality_dataset()
learn_all_new_gold_token(actuality_dataset, model, tokenizer)

sample = 'macron trudeau scholz meloni kishida sunak diaz alibaba AT&T uber servicenow'
encoding_1 = tokenizer.encode(sample)
token_1 = tokenizer.convert_ids_to_tokens(encoding_1)

# Test learn_new_token

sample = ["abc def ghi.", "jkl"]
learn_new_token(sample, model, tokenizer)

encoding_sample = tokenizer.encode(sample[0])
encoding_gold = tokenizer.encode(sample[1])
token_sample = tokenizer.convert_ids_to_tokens(encoding_sample)
token_gold = tokenizer.convert_ids_to_tokens(encoding_gold)

# Test 1

# load actuality_dataset
#actuality_dataset = load_actuality_dataset()

# create sample and learn new token from sample
sample = ["Ukraine is in a war against [MASK].", "Russia"]
learn_new_token(sample, model, tokenizer)

# test encoling
#encoding = tokenizer.encode('Olaf Scholz')
#token = tokenizer.convert_ids_to_tokens(encoding)

# create key words
key_words = ["ukraine", "war"]

# load news from guardian and news_api
news_api_query, guardian_query, guardian_query_df = news_loader('2022-12-05', key_words)
split_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(split_query, tokenizer)

#filter information out of full article list
extraction_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
#focus_query = keyword_focus(extraction_query, key_words, 5)

# make prediction
pred_query = make_predictions(sample[0], extraction_query, model, tokenizer)

# Test 2

# create sample and learn new token from sample
sample = ["Barack Obama [MASK] the president of the United States of America.", "was"]
learn_new_token(sample, model, tokenizer)

# create key words
key_words = ["obama", "president"]

# get NER keywords
ner_keywords = ner_keywords(sample)
# get POS keywords
pos_keywords = pos_keywords(sample)

# load news from guardian and news_api
news_api_query, guardian_query, guardian_query_df = news_loader('2022-12-15', 'key_words')
split_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(split_query, tokenizer)

#filter information out of full article list
extraction_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
#focus_query = keyword_focus(extraction_query, key_words, 5)

# make prediction
pred_query = make_predictions(sample[0], extraction_query, model, tokenizer)
"""



"""
                ----------------Möglicherweise als gliederung in ungefährer From nutzbar
"""


# Gesamt Test verschiederer Möglichkeiten 


"""
    - schlechtes Beispiel da Token alibaba unbekannt ist
    Alibaba kann für [MASK] nie predictet werden
    Problem nur lösbar wenn BERT Tokens beigebracht werden können
    
    -->> Zeigt Limitation der Projektarbeit
"""
"""
# Test 3 

# create sample and learn new token from sample
sample = ["Daniel Zhang is the chief executive officer of [MASK] group.", "alibaba"]
#learn_new_token(sample, model, tokenizer)

# create key words
key_words = ['zhang', 'alibaba']

# get NER keywords
#ner_keywords = ner_keywords(sample)
# get POS keywords
pos_keywords = pos_keywords(sample)

# load news from guardian and news_api
news_api_query, guardian_query, guardian_query_df = news_loader('2022-09-01', key_words)    #using key_words weil pos_keywords prime enthält, was einen error verursacht
split_query = nltk_sentence_split(news_api_query, guardian_query) 
merged_query = merge_pages(split_query, tokenizer)

#filter information out of full article list
extraction_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
focus_query = keyword_focus(extraction_query, key_words, 5)

# make prediction
pred_query = make_predictions(sample[0], focus_query, model, tokenizer)
"""



"""
    Working BERTuality funktioniert aber mit Limitationen der TOKENS
    Gezeigt am Beispiel von Tim Cook 
"""

"""
# 1. Alle Tokens sind unter BERT bekannt

# load dataset with known gold token
actuality_dataset = load_actuality_dataset(tokenizer, delete_unknown_token=False)

# create sample and learn new token from sample
sample = ["Tim Cook is the CEO of [MASK].", "Apple"]
sample2 = ["Andy Jassy is the current CEO of [MASK].", "Amazon"]

# create key words
key_words = pos_keywords(sample)


# 2. Teste Vorwissen von BERT indem kein Input gegeben wird
pretrained_knowledge = make_predictions(sample[0], [""], model, tokenizer)
simple_pre_know = simple_pred_results(pretrained_knowledge)
# ERGEBNIS: BERT kennt keinen Zusammenhang zu Tim Cook und Apple und gibt als Word "Amazon"


# 4. Test mit unserem Verfahren um BERT "umzustimmen" und dem Model das richtige Ergebnis beizubringen

# load news from guardian and news_api
news_api_query, guardian_query, guardian_query_df = news_loader('2023-01-05', key_words)
split_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(split_query)

#filter information out of full article list
#extraction_query = filter_list_final(merged_query, key_words, tokenizer)
extraction_query = filter_for_keyword_subsets(merged_query, key_words, tokenizer, 2)

# focus on relevant part of sentence
focus_query = keyword_focus(extraction_query, key_words, 5)

# make prediction
pred_query = make_predictions(sample[0], focus_query, model, tokenizer)
#pred_query_info = make_predictions(sample[0], extraction_query, model, tokenizer)

simple_results = simple_pred_results(pred_query)
#simple_results_info = simple_pred_results(pred_query_info)

# ERGEBNIS NEU: Es wurde aufgezeigt, das BERT unter gelerntem Input aus dem Internet andere Predictions 
# für das MASK Word abgibt und Tim Cook mit großem Score nun als CEO von Apple vorhersagt
"""

"""
    PROBLEM Fehlende Tokens
    
    Probierte Möglichkeiten:
        - func. learn_new_tokens
        - ...
        
    Folgerung- BERT kann keine Tokens lernen
        
"""

"""
    Aktualitätsprüfung mit dem fill_mask model funktioniert --> solange die TOKENS in BERT bekannt sind
"""

"""
                MAIN TEST 2 - WP PREDICTION   
"""

"""
# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

# model: BERT pre-trained
model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')

# 1. Alle Tokens sind unter BERT bekannt

# load dataset with known gold token
#actuality_dataset = load_actuality_dataset(tokenizer, delete_unknown_token=False)

# create sample and learn new token from sample
#sample = ["Tim Cook is the CEO of [MASK].", "Apple"]
#sample2 = ["Andy Jassy is the current CEO of [MASK].", "Amazon"]
sample3 = "Daniel Zhang is the CEO of [MASK]."

# create key words
key_words = pos_keywords(sample3)


# 2. Teste Vorwissen von BERT indem kein Input gegeben wird
#pretrained_knowledge = make_predictions(sample2[0], [""], model, tokenizer)
#simple_pre_know = simple_pred_results(pretrained_knowledge)
# ERGEBNIS: BERT kennt keinen Zusammenhang zu Tim Cook und Apple und gibt als Word "Amazon"


# 4. Test mit unserem Verfahren um BERT "umzustimmen" und dem Model das richtige Ergebnis beizubringen

# load news from guardian and news_api
loader_query = news_loader('2022-05-05', key_words)
split_query = nltk_sentence_split(loader_query)
merged_query = merge_pages(split_query)

#filter information out of full article list
#extraction_query = filter_list_final(merged_query, key_words, tokenizer)
extraction_query = filter_for_keyword_subsets(merged_query, key_words, tokenizer, 2)

# focus on relevant part of sentence
focus_query = keyword_focus(extraction_query, key_words, 5)

# make prediction
pred_query = make_predictions(sample3, focus_query, model, tokenizer)               
wp_pred_query = word_piece_prediction(sample3, focus_query, model, tokenizer, threshold=0.9, max_input=10)   

# fancy results
simple_results = simple_pred_results(pred_query)                                              
wp_simple_results = simple_pred_results(wp_pred_query)                                 

"""




"""
Model & Tokenizer
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


"""
Load Dataset
"""
# full dataset 
actuality_dataset = load_actuality_dataset(tokenizer, delete_unknown_token=False)


"""
Load Articles for Dataset
"""

# load articles here to minimize api calls and gather data here --> also performance upgrade
#dataset_articles = []
#for index in range(len(actuality_dataset)):
#    loader = loader_query(actuality_dataset["MaskSatz"][index], "2022-01-01", "2023-01-30", use_NewsAPI=True, use_guardian=True, use_wikipedia=True)
#    dataset_articles.append(loader)
#
#with open('full_cleaned_data_22-01-01_to_23-01-30.pickle', 'wb') as f:
#    pickle.dump(dataset_articles, f)  




##################################################################################################################
                                                # TEST-SCRIPT #
##################################################################################################################




"""
Unpack Pickle to get Dataset -- no need for loader pipeline
"""
data=None
with open('full_cleaned_data_22-01-01_to_23-01-30.pickle', 'rb') as f:
    data = pickle.load(f) 

# Unpack Pickle Data to get query list
dataset_articles = [d["08_loader_query"] for d in data]



'''
##################################################################################################################
# WPW MODEL TEST - START -----> TESTING:     MODEL 1,      2,    and       3
##################################################################################################################
"""
Test 1 - base test for comparison

# bert standart --> prod version
base_results_v1 = automatic_dataset_pred(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.3, word_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False, 
                                      extraction=True, similarity=True, focus=True, duplicates=False)
base_scoring_v1 = scoring(base_results_v1)

with open('Tests\test_1.pickle', 'wb') as f:
    pickle.dump(base_results_v1, f)  
"""

# Test 2 deleted

"""
Test 3 - new_WORDPIECE PRED MODEL --> --------- WINNGING MODEL
"""
'''
base_results_v2 = automatic_dataset_pred_v2(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.3, word_padding=5, 
                                      threshold=0.9, max_input=10, query_test=False, 
                                      extraction=True, similarity=True, focus=True, duplicates=False)
base_scoring_v2 = scoring(base_results_v2)
'''
with open('Tests\test_3.pickle', 'wb') as f:
    pickle.dump(base_results_v2, f) 
##################################################################################################################
# WPW MODEL TEST - END
##################################################################################################################




##################################################################################################################
# QUERY - TEST - START  ----> EXTRACTION or SIMILARITY BETTER? ------> TEST ON model 2
##################################################################################################################
"""
Test 4 - model 2 -- only extraction query
"""
# übergebe tokens an bert + fasat
base_results_v2_extract_true = automatic_dataset_pred_v2(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.6, word_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False, 
                                      extraction=True, similarity=False, focus=False, duplicates=False)
base_scoring_v2_extract_true = scoring(base_results_v2_extract_true)

with open('Tests\test_4.pickle', 'wb') as f:
    pickle.dump(base_results_v2_extract_true, f) 




"""
Test 5 - model 2 -- only similarity query --> score 0.5
"""
base_results_v2_sim_true = automatic_dataset_pred_v2(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.5, word_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False, 
                                      extraction=False, similarity=True, focus=False, duplicates=False)
base_scoring_v2_sim = scoring(base_results_v2_sim_true)

with open('Tests\test_5.pickle', 'wb') as f:
    pickle.dump(base_results_v2_sim_true, f) 
    



"""
Test 6 - model 2 -- hybrid query (extr. and sim. --> score 0.5) 
"""
base_results_v2_hybrid = automatic_dataset_pred_v2(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.5, word_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False, 
                                      extraction=True, similarity=True, focus=False, duplicates=False)
base_scoring_v2_hybrid = scoring(base_results_v2_hybrid)

with open('Tests\test_6.pickle', 'wb') as f:
    pickle.dump(base_results_v2_hybrid, f) 
    
    
"""

    Important!!!: model 2 is now official Model 2

""" 
    
    

"""
Test 7 - model 2 -- hybrid query (extr. and sim. --> score 0.2) 
"""
base_results_v2_hybrid_low = automatic_dataset_pred_v2(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.2, word_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False, 
                                      extraction=True, similarity=True, focus=False, duplicates=False)
base_scoring_v2_hybrid_low = scoring(base_results_v2_hybrid_low)

with open('Tests\test_7.pickle', 'wb') as f:
    pickle.dump(base_results_v2_hybrid_low, f)
##################################################################################################################
# QUERY - TEST - END
##################################################################################################################
'''












