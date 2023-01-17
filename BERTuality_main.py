from transformers import BertTokenizer, BertForMaskedLM, pipeline
from BERTuality_loader import wikipedia_loader
from BERTuality_loader import NewsAPI_loader
from BERTuality_loader import guardian_loader
from BERTuality_loader import news_loader
from BERTuality import nltk_sentence_split
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
news_api_query, guardian_query, guardian_query_df = news_loader('2022-12-25', key_words)    #using key_words weil pos_keywords prime enthält, was einen error verursacht
filtered_query = nltk_sentence_split(news_api_query, guardian_query) 
merged_query = merge_pages(filtered_query, tokenizer)

#filter information out of full article list
info_query = filter_list_final(merged_query, key_words)

# focus on relevant part of sentence
focus_query = keyword_focus(info_query, key_words, 5)

# make prediction
query_pred = make_predictions(sample[0], focus_query, model, tokenizer)



#actuality_dataset = load_actuality_dataset()






