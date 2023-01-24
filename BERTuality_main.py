from transformers import BertTokenizer, BertForMaskedLM, pipeline
from BERTuality_loader import wikipedia_loader
from BERTuality_loader import NewsAPI_loader
from BERTuality_loader import guardian_loader
from BERTuality_loader import news_loader
from BERTuality import nltk_sentence_split
from BERTuality import split_into_sentences
from BERTuality import remove_too_long_sentences
from BERTuality import merge_pages
from BERTuality import make_predictions
from BERTuality import filter_list_final
from BERTuality import keyword_creator
from BERTuality import keyword_focus
from BERTuality import load_actuality_dataset
from BERTuality import learn_new_token
from BERTuality import learn_all_new_gold_token
from BERTuality import ner_keywords
from BERTuality import pos_keywords
from BERTuality import simple_pred_results
from BERTuality import filter_for_keyword_subsets

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# model: BERT pre-trained
model = BertForMaskedLM.from_pretrained('bert-large-uncased')

#Wikipedia Covid Test
"""
# load pages
page_1 = wikipedia_loader("Coronavirus", "text")
page_2 = wikipedia_loader("SARS-CoV-2", "text")
page_3 = wikipedia_loader("COVID-19", "text")
page_4 = wikipedia_loader("COVID-19_pandemic", "text")
page_5 = wikipedia_loader("COVID-19_pandemic_in_Europe", "text")

# filter pages
filtered_pages = nltk_sentence_split(page_1, page_2, page_3, page_4, page_5)
merged_pages = merge_pages(filtered_pages)

# predict
pred_1 = make_predictions("Covid is a [MASK]", filter_list_final(merged_pages, ["Covid", "Virus", "China"]))
"""

# Wikipedia and news api test
"""
page_6 = wikipedia_loader("Bob_Bennett", "text") 
page_7 = NewsAPI_loader("Bob Bennett AND Robert Foster Bennett")

filtered_pages = nltk_sentence_split(page_6)
filtered_pages.append(page_7)               # append, weil der NewsAPI_loader eigene Satzfilter verwendet; hier gibt es andere Suffixe und Präfixe als bei Wikipedia z.B.
merged_pages = merge_pages(filtered_pages)

masked_sentence = "Robert Foster Bennett is a [MASK] of the United States by profession."
keywords = keyword_creator(masked_sentence, word_deletion=True, criteria="shortest", min_key_words=2)

pred_2 = make_predictions(masked_sentence, filter_list_final(merged_pages, keywords))
"""

#wikipedia and news api test
"""
page_6 = wikipedia_loader("Niko_Kovač", "text") 
page_7 = NewsAPI_loader("2022-08-01", "Niko Kovac")
page_8, query_df = guardian_loader(from_date="2022-08-01", to_date="2022-12-15", query="Kovac")

filtered_pages = nltk_sentence_split(page_6, page_7, page_8)
merged_pages = merge_pages(filtered_pages)

masked_sentence = "Niko Kovač is a german football [MASK]."
keywords = keyword_creator(masked_sentence, word_deletion=True, criteria="shortest", min_key_words=2)

#pred_2_keyword_creator = make_predictions(masked_sentence, filter_list_final(merged_pages, keywords))
### bei Keywords: Kovač problematisch, weil manche News Seiten c und andere č schreiben; deswegen Sonderzeichen weglassen bei Keyword Suche; Kova ist in Kovač und funktioniert deswegen
pred_2_own_keywords = make_predictions(masked_sentence, filter_list_final(merged_pages, [['Niko','Kova','german','football'],['Kova','german','football'],['Kova','football']]))
"""

# guardian test
"""
query, query_df = guardian_loader(from_date="2022-08-01", to_date="2022-12-15", query="Scholz")
#path, path_df = guardian_loader(from_date="2022-08-01", to_date="2022-12-15", path="politics")

filtered_query = nltk_sentence_split(query)
#filtered_path = nltk_sentence_split(path)

merged_query = merge_pages(filtered_query, tokenizer)
#merged_path = merge_pages(filtered_path)

masked = "Chancellor Merkel is the [MASK] leader of Germany."
masked_2 = "[MASK] is the chancellor of germany"
key_words = ["Chancellor", "Merkel"]
key_words_2 = ["Chancellor", "scholz"]


query_pred = make_predictions(masked_2, filter_list_final(merged_query, key_words_2), model, tokenizer)
#path_pred = make_predictions(masked, filter_list_final(merged_path, key_words))
"""

# Test news_loader
"""
news_api_query, guardian_query, guardian_query_df = news_loader('2022-12-17', 'Ukraine')

filtered_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(filtered_query, tokenizer)

masked = "Ukraine is in a war against [MASK]."
key_words = ["Ukraine", "war"]

query_pred = make_predictions(masked, filter_list_final(merged_query, key_words), model, tokenizer)
"""

"""
Error in Fkt. guardian_call (wenn from_date zu weit in der Vergangenheit):  
news_api_query, guardian_query, guardian_query_df = news_loader('2021-12-10', 'Olaf Scholz')

Key Error in guardian_loader: (query="Niko Kovač" funktioniert)
page_8, query_df = guardian_loader(from_date="2022-08-15", to_date="2022-12-15", query="Kovač") 
"""


# Test learn_all_new_token (for gold token in dataset)
"""
actuality_dataset = load_actuality_dataset()
learn_all_new_gold_token(actuality_dataset, model, tokenizer)

sample = 'macron trudeau scholz meloni kishida sunak diaz alibaba AT&T uber servicenow'
encoding_1 = tokenizer.encode(sample)
token_1 = tokenizer.convert_ids_to_tokens(encoding_1)
"""


# Test learn_new_token
"""
sample = ["abc def ghi.", "jkl"]
learn_new_token(sample, model, tokenizer)

encoding_sample = tokenizer.encode(sample[0])
encoding_gold = tokenizer.encode(sample[1])
token_sample = tokenizer.convert_ids_to_tokens(encoding_sample)
token_gold = tokenizer.convert_ids_to_tokens(encoding_gold)
"""


# Test 1
"""
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
filtered_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(filtered_query, tokenizer)

#filter information out of full article list
info_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
#focus_query = keyword_focus(info_query, key_words, 5)

# make prediction
query_pred = make_predictions(sample[0], info_query, model, tokenizer)
"""

# Test 2
"""
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
filtered_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(filtered_query, tokenizer)

#filter information out of full article list
info_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
#focus_query = keyword_focus(info_query, key_words, 5)

# make prediction
query_pred = make_predictions(sample[0], info_query, model, tokenizer)
"""











# TODO
"""


            Weitete TODOs
                
                - Main Pipeline zusammenfügen ? ==> erste gesamt Pipeline fertig, wird aber noch geändert
                    - Keyworderstellung hinzufügen und in main formulieren
                    
                - Datensatz verwerfen ? 
                    - Gedanken: Viele Tokens sind unbekannt und lassen sich von BERT nicht predicten
                    - Idee: Herausfinden welche Sätze aus dem Datensatz eine vollständige Tokenisierung haben und diese benutzen
                            - Rest verwerfen
                    
                    
                
                - Keyword erstellung --> (NNP, NN) usw. entfernen -- Maxi
                - 
                -
                
                
                
                - Tests verschiedene Modelle erstmal unnätig
                    - kann just for fun getestet werden und vllt. in arbeit aufgenommen werden, aber geringe Prio
                    


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
filtered_query = nltk_sentence_split(news_api_query, guardian_query) 
merged_query = merge_pages(filtered_query, tokenizer)

#filter information out of full article list
info_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
focus_query = keyword_focus(info_query, key_words, 5)

# make prediction
query_pred = make_predictions(sample[0], focus_query, model, tokenizer)
"""



"""
    Working BERTuality funktioniert aber mit Limitationen der TOKENS
    Gezeigt am Beispiel von Tim Cook 
"""

# 1. Alle Tokens sind unter BERT bekannt

# load dataset with known gold token
actuality_dataset = load_actuality_dataset(tokenizer, delete_unknown_token=False)

# create sample and learn new token from sample
sample = ["Tim Cook is the CEO of [MASK].", "Apple"]
sample2 = ["Andy Jassy is the current CEO of [MASK]", "Amazon"]

# create key words
key_words = pos_keywords(sample2)


# 2. Teste Vorwissen von BERT indem kein Input gegeben wird
pretrained_knowledge = make_predictions(sample2[0], [""], model, tokenizer)
simple_pre_know = simple_pred_results(pretrained_knowledge)
# ERGEBNIS: BERT kennt keinen Zusammenhang zu Tim Cook und Apple und gibt als Word "Amazon"


# 4. Test mit unserem Verfahren um BERT "umzustimmen" und dem Model das richtige Ergebnis beizubringen

# load news from guardian and news_api
news_api_query, guardian_query, guardian_query_df = news_loader('2023-01-05', key_words)
filtered_query = nltk_sentence_split(news_api_query, guardian_query)
merged_query = merge_pages(filtered_query)

#filter information out of full article list
#info_query = filter_list_final(merged_query, key_words, tokenizer)
info_query = filter_for_keyword_subsets(merged_query, key_words, tokenizer, 2)

# focus on relevant part of sentence
focus_query = keyword_focus(info_query, key_words, 5)

# make prediction
query_pred = make_predictions(sample2[0], focus_query, model, tokenizer)
#query_pred_info = make_predictions(sample[0], info_query, model, tokenizer)

simple_results = simple_pred_results(query_pred)
#simple_results_info = simple_pred_results(query_pred_info)

# ERGEBNIS NEU: Es wurde aufgezeigt, das BERT unter gelerntem Input aus dem Internet andere Predictions 
# für das MASK Word abgibt und Tim Cook mit großem Score nun als CEO von Apple vorhersagt


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


