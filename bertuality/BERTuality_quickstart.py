from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertForMaskedLM, pipeline

from bertuality.BERTuality import loader_query
from bertuality.BERTuality import dataprep_query
from bertuality.BERTuality import word_piece_prediction 
from bertuality.BERTuality import simple_pred_results

from bertuality.BERTuality_loader import news_loader

import datetime



"""
    bertuality_main_func
"""


def load_default_config():
    
    currentday = datetime.date.today()
    deltaday = currentday - datetime.timedelta(90)
    
    default_config = {
        'model': r'bertuality/model',
        'tokenizer': r'bertuality/tokenizer',
        'from_date': str(deltaday),
        'to_date': str(currentday),
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
        
        print("Step 3: Prepare data --> Done")
        print("Step 4: Start Prediction:\n")
        
        prediction = word_piece_prediction(mask_sentence, 
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
        
        