import os
import sys

path = os.path.join(os.path.abspath('..'))
sys.path.append(path)

from bertuality.BERTuality import word_piece_prediction
from bertuality.BERTuality import simple_pred_results
from bertuality.BERTuality import make_predictions
from bertuality.BERTuality import dataprep_query
from bertuality.BERTuality import pos_keywords

import pandas as pd




"""
    Learn New Tokens - failed
"""


# learn all gold token in given dataset
def learn_all_new_gold_token(dataset, model, tokenizer):
    # create list of gold token from given dataset
    gold_token = list(dataset['Gold'])
    
    # add new token to tokenizer
    num_added_toks = tokenizer.add_tokens(gold_token)
    model.resize_token_embeddings(len(tokenizer)) #resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer
  
    
  
    
# learn one new gold token in mask_sentence, mask_sentence has to look like this: ['sentence', 'gold token']    
def learn_new_token(mask_sentence, model, tokenizer):
    # create list of all token form given mask_sentence ['sentence', 'gold token']
    gold_token = [mask_sentence[1]]
    other_token = mask_sentence[0].split()
    new_token = gold_token + other_token #1st new token gets id 30522
    
    # add new token to tokenizer
    num_added_toks = tokenizer.add_tokens(new_token)
    model.resize_token_embeddings(len(tokenizer)) #resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer
    
    
    

"""
    Dataset functions for faster testing
"""


def load_actuality_dataset(tokenizer, delete_unknown_token = False):
    # load dataset
    actuality_dataset = pd.read_excel (r'data/DS_Aktualitätsprüfung.xlsx', sheet_name = 'Gesamt Daten')
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
    
    # rows that are not suitable for an actuality check
    unsuitable_rows = [619, 622, 644, 645, 648, 649, 655, 673, 674, 675, 676, 677, 707, 750, 751, 752, 756]
    
    # find rows with unknown Gold Token (= Gold token consists of multiple WordPieces)
    unknown_gold_token_rows = []
    if delete_unknown_token:
        for index, row in actuality_dataset.iterrows():
            gold_token = row['Gold']
            encoding = tokenizer.encode(str(gold_token))
            if (len(encoding) != 3):    # if encoding consists of more than CLS + Token + SEP
                unknown_gold_token_rows.append(int(row['Nummer']))
    
    # delete the rows with unknown gold tokens and the unsuitable rows
    rows_to_delete = unsuitable_rows + unknown_gold_token_rows
    for i in rows_to_delete:
        actuality_dataset = actuality_dataset[actuality_dataset.Nummer != i]
        
    # delete rows that the tokenizer does not know the vaule of
    #for i in actuality_dataset:
    actuality_dataset = actuality_dataset.reset_index(drop=True)
    return actuality_dataset




def automatic_dataset_pred(actuality_dataset, loader_query, tokenizer, model, 
                           subset_size=2, sim_score=0.0, focus_padding=0, 
                           threshold=None, max_input=None, query_test=False, 
                           extraction=True, similarity=True, focus=True, duplicates=False, only_target_token=False):
    
    #actuality_dataset = actuality_dataset.reset_index(drop=True)
    
    results = []
    error = []
    for index in range(len(actuality_dataset)):
        
        print(f"\nDataset Prediction Progress [{index+1}/{len(actuality_dataset)}]")
        
        
        query = dataprep_query(actuality_dataset["MaskSatz"][index], loader_query[index], tokenizer, 
                               subset_size=subset_size, sim_score=sim_score, focus_padding=focus_padding, 
                               extraction=extraction, similarity=similarity, focus=focus, duplicates=duplicates)
        
        # For testing the query
        if query_test == True: 
            results += query, 
            continue
        
        # safety mech
        if len(query)==0: 
            samp_results = {"01_Nummer": actuality_dataset["Nummer"][index],
                   "02_MaskSatz": actuality_dataset["MaskSatz"][index],
                   "03_Original": actuality_dataset["Original"][index],
                   "04_Gold":actuality_dataset["Gold"][index],
                   "05_query": query,
                   "06_Prediction": "Error"}
            results.append(samp_results)
            error += index,
            continue
        
        # Test Knowlede of BERT without input
        kn_pred_query = make_predictions(actuality_dataset["MaskSatz"][index], [""], model, tokenizer, max_input=max_input)              
        kn_simple_results = simple_pred_results(kn_pred_query)
        
        # Test Original BERT with input
        or_pred_query = make_predictions(actuality_dataset["MaskSatz"][index], query, model, tokenizer, max_input=max_input)              
        or_simple_results = simple_pred_results(or_pred_query)  
        
        # Test full Word Piece Prediction # TODO is v2!!!!!!!!!!!!!!!!!
        wp_pred_query = word_piece_prediction(actuality_dataset["MaskSatz"][index], query, model, tokenizer, 
                                              threshold=threshold, max_input=max_input, only_target_token=only_target_token)              
        wp_simple_results = simple_pred_results(wp_pred_query)                                  
        
        
        samp_results = {"01_Nummer": actuality_dataset["Nummer"][index],
                   "02_MaskSatz": actuality_dataset["MaskSatz"][index],
                   "03_Original": actuality_dataset["Original"][index],
                   "04_Gold":actuality_dataset["Gold"][index],
                   "05_query": query,
                   "06_keywords": pos_keywords(actuality_dataset["MaskSatz"][index]),
                   
                   "07_kn_pred_query": kn_pred_query,
                   "08_kn_simple_results": kn_simple_results,
                   "09_kn_word": kn_simple_results["Token"][0],
                   "10_kn_score": kn_simple_results["sum_up_score"][0],
                   
                   "11_or_pred_query": or_pred_query,
                   "12_or_simple_results": or_simple_results,
                   "13_or_word": or_simple_results["Token"][0],
                   "14_or_score": or_simple_results["sum_up_score"][0],
                   
                   "15_wp_pred_query": wp_pred_query,
                   "16_wp_simple_results": wp_simple_results,
                   "17_wp_word": wp_simple_results["Token"][0],
                   "18_wp_score": wp_simple_results["sum_up_score"][0],
                   }
        results.append(samp_results)
    
    print("\nActuality_Dataset Prediction Summary:")
    print("Length Dataset:", len(results))
    print("Predictions on:", round((len(results) - len(error))/len(results)*100, 2), "% of Dataset")
    print("No Predictions for Index:", error)
    
    return results




# create a dataFrame with all the important scores and informations about the tests
def scoring(results):
    # initialize scores
    corr_kn = 0
    corr_or = 0
    corr_persuaded_or = 0
    corr_wp = 0
    corr_persuaded_wp = 0
    num_query_empty = 0
    num_of_tests = len(results)
    
    # calculate scores
    for i in results:
        if ('06_Prediction' not in i):
            if (i['09_kn_word'] == i['04_Gold']): corr_kn += 1
            if (i['13_or_word'] == i['04_Gold']): corr_or += 1
            if (i['17_wp_word'] == i['04_Gold']): corr_wp += 1
            
            if (i['13_or_word'] == i['04_Gold'] and i['13_or_word'] != i['09_kn_word']): 
                corr_persuaded_or += 1
            if (i['17_wp_word'] == i['04_Gold'] and i['17_wp_word'] != i['09_kn_word']):
                corr_persuaded_wp += 1
        else:
            num_query_empty += 1
      
    # calculate average scores        
    avg_corr_kn = round(corr_kn / num_of_tests, 3)
    avg_corr_or = round(corr_or / num_of_tests, 3)
    avg_corr_persuaded_or = round(corr_persuaded_or / num_of_tests, 3)
    avg_corr_wp = round(corr_wp / num_of_tests, 3)
    avg_corr_persuaded_wp = round(corr_persuaded_wp /num_of_tests, 3)
    
    # create df
    columns = ["avg", "avg_pers", "#tests", "#empty"]
    rows = ["knowledge", "original", "word-peice"]
    results = pd.DataFrame(index = rows, columns = columns)
    
    #fill df
    avg_corr_values = [avg_corr_kn, avg_corr_or, avg_corr_wp]
    avg_corr_persuaded_values = [0, avg_corr_persuaded_or, avg_corr_persuaded_wp]
    num_query_empty_value = [num_query_empty, "/", "/"]
    num_of_tests_value = [num_of_tests, "/", "/"]
    
    results["avg"] = avg_corr_values
    results["avg_pers"] = avg_corr_persuaded_values
    results["#empty"] = num_query_empty_value
    results["#tests"] = num_of_tests_value
    
    return results