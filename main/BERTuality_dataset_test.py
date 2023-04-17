from transformers import BertTokenizer, BertForMaskedLM, pipeline
from BERTuality_dataset_functions import load_actuality_dataset
from BERTuality_dataset_functions import automatic_dataset_pred
from BERTuality_dataset_functions import automatic_dataset_pred_v2
from BERTuality import loader_query
from BERTuality_dataset_functions import scoring

import pickle
import os




"""
Model, Tokenizer and actuality_datqaset
"""

# use this code block to download a new model/tokenizer from hugginface
# and save them into a file for faster loading
"""
tokenizer = BertTokenizer.from_pretrained(r'tokenizer')
model = BertForMaskedLM.from_pretrained(r'model')

tokenizer.save_pretrained(os.path.join(os.getcwd(), 'tokenizer'))
model.save_pretrained(os.path.join(os.getcwd(), 'model'))
"""


# load a pretrained and presaved model/tokenizer from files
tokenizer = BertTokenizer.from_pretrained(r'tokenizer')
model = BertForMaskedLM.from_pretrained(r'model')


actuality_dataset = load_actuality_dataset(tokenizer, delete_unknown_token=False)




"""
Load new articles for dataset
"""


# load articles here to minimize api calls and gather data here --> also performance upgrade
#dataset_articles = []
#for index in range(len(actuality_dataset)):
#    loader = loader_query(actuality_dataset["MaskSatz"][index], "2022-01-01", "2023-01-30", use_NewsAPI=True, use_guardian=False, use_wikipedia=True)
#    dataset_articles.append(loader)
#
#with open(r'data/full_cleaned_data_22-01-01_to_23-01-30.pickle', 'wb') as f:
#    pickle.dump(dataset_articles, f)  




"""
Unpack Pickle to get dataset -- no need for loader pipeline
"""


data=None
with open(r'data/full_cleaned_data_22-01-01_to_23-01-30.pickle', 'rb') as f:
    data = pickle.load(f) 


# Unpack Pickle Data to get query list
dataset_articles = [d["08_loader_query"] for d in data]




##################################################################################################################
# QUERY - DATASET-TEST - WP-Pred 1 vs. WP-Pred 2
##################################################################################################################


# BERTuality - WP-Pred 1 - Test
bertuality_test_v1 = automatic_dataset_pred(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.4, focus_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False,
                                      extraction=True, similarity=True, focus=True, duplicates=False)
bertuality_test_score = scoring(bertuality_test_v1)


with open(r'data/bertuality_test_v1.pickle', 'wb') as f:
    pickle.dump(bertuality_test_v1, f) 




# BERTuality - WP-Pred 2 - Test
bertuality_test_v2 = automatic_dataset_pred_v2(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.4, focus_padding=5, 
                                      threshold=0.9, max_input=40, query_test=False, only_target_token=True,
                                      extraction=True, similarity=True, focus=True, duplicates=False)
bertuality_test_score = scoring(bertuality_test_v2)

    
with open(r'data/bertuality_test_v2.pickle', 'wb') as f:
    pickle.dump(bertuality_test_v2, f) 


##################################################################################################################
# QUERY - DATASET-TEST
##################################################################################################################


"""
Unpack Pickle to get old tests
"""


with open(r'data/bertuality_test_v1.pickle', 'rb') as f:
    test = pickle.load(f) 

test_score = scoring(test)


with open(r'data/bertuality_test_v2.pickle', 'rb') as f:
    test = pickle.load(f) 

test_score = scoring(test)










