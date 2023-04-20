import os
import sys

path = os.path.join(os.path.abspath('..'))
sys.path.append(path)

from transformers import BertTokenizer, BertForMaskedLM, pipeline
from bertuality.BERTuality import loader_query

from bertuality_testing.BERTuality_dataset_functions import load_actuality_dataset
from bertuality_testing.BERTuality_dataset_functions import automatic_dataset_pred
from bertuality_testing.BERTuality_dataset_functions import scoring

import pickle



"""
Model, Tokenizer and actuality_datqaset
"""

# use this code block to download a new model/tokenizer from hugginface
# and save them into a file for faster loading
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

tokenizer.save_pretrained(os.path.join(path, 'bertuality/tokenizer'))
model.save_pretrained(os.path.join(path, 'bertuality/model'))
"""


# load a pretrained and presaved model/tokenizer from files
tokenizer = BertTokenizer.from_pretrained(os.path.join(path, 'bertuality/tokenizer'))
model = BertForMaskedLM.from_pretrained(os.path.join(path, 'bertuality/model'))


actuality_dataset = load_actuality_dataset(tokenizer, delete_unknown_token=False)




"""
Load new articles for dataset
"""


# load articles here to minimize api calls and gather data here --> also performance upgrade
#from_date = "2022-01-01"
#to_date = "2023-01-30"

#dataset_articles = []
#for index in range(len(actuality_dataset)):
#    loader = loader_query(actuality_dataset["MaskSatz"][index], from_date, to_date, use_NewsAPI=True, use_guardian=False, use_wikipedia=True)
#    dataset_articles.append(loader)
#
#with open(r'data/dataset_' + from_date + '_to_' + to_date + '.pickle', 'wb') as f:
#    pickle.dump(dataset_articles, f)  




"""
Unpack Pickle to get dataset -- no need for loader pipeline
"""


data=None
with open(r'data/dataset_2022-01-01_to_2023-01-30.pickle', 'rb') as f:
    data = pickle.load(f) 


# Unpack Pickle Data to get query list
dataset_articles = [d["08_loader_query"] for d in data]




##################################################################################################################
# QUERY - DATASET-TEST - WP-Pred 1 vs. WP-Pred 2
##################################################################################################################


# BERTuality - WP-Pred - Test
bertuality_test = automatic_dataset_pred(actuality_dataset[:], dataset_articles, tokenizer, model, 
                                      subset_size=2, sim_score=0.4, focus_padding=5, 
                                      threshold=0.9, max_input=20, query_test=False, only_target_token=True,
                                      extraction=True, similarity=True, focus=True, duplicates=False)
bertuality_test_score = scoring(bertuality_test)

    
with open(r'data/bertuality_test.pickle', 'wb') as f:
    pickle.dump(bertuality_test, f) 


##################################################################################################################
# QUERY - DATASET-TEST
##################################################################################################################


"""
Unpack Pickle to get old tests
"""


with open(r'data/bertuality_test.pickle', 'rb') as f:
    test = pickle.load(f) 

test_score = scoring(test)












