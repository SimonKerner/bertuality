from BERTuality import bertuality


"""
BERTuality Config options 
    - only change if needed
    - else: default values 
"""


"""
example_config = {
    'model': 'bert-base-uncased',
    'tokenizer':'bert-base-uncased',
    'from_date': '2022-10-01',
    'to_date': '2023-01-30',
    'use_NewsAPI': True, 
    'use_guardian': False, 
    'use_wikipedia': True, 
    'subset_size': 2,
    'sim_score': 0.3,
    'focus_padding': 5,
    'duplicates': False,
    'extraction': True,
    'similarity': True,
    'focus': True,
    'max_input': 20,
    'threshold': 0.9,
    'only_target_token': True
    }
"""

#bertuality(mask_sentence)
values_1 = bertuality("Obama is the [MASK] president of America.", return_values=True)

values_2 = bertuality("Donald Trump is the [MASK] president of America.", return_values=True)

values_3 = bertuality("Tim Cook is the current CEO of [MASK].", return_values=True)



