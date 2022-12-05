from BERTuality import wikipedia_loader
from BERTuality import sentence_converter
from BERTuality import merge_pages
from BERTuality import make_predictions
from BERTuality import filter_list_final
from BERTuality import keyword_creator


"""
# load pages
page_1 = wikipedia_loader("Coronavirus", "text")
page_2 = wikipedia_loader("SARS-CoV-2", "text")
page_3 = wikipedia_loader("COVID-19", "text")
page_4 = wikipedia_loader("COVID-19_pandemic", "text")
page_5 = wikipedia_loader("COVID-19_pandemic_in_Europe", "text")

# filter pages
filtered_pages = sentence_converter(page_1, page_2, page_3, page_4, page_5)
merged_pages = merge_pages(filtered_pages)

# predict
pred_1 = make_predictions("Covid is a [MASK]", filter_list_final(merged_pages, ["Covid", "Virus", "China"]))
"""

page_6 = wikipedia_loader("Lionel_Messi", "text") 

filtered_pages = sentence_converter(page_6)
merged_pages = merge_pages(filtered_pages)

masked_sentence = "Lionel Messi was born in [MASK]."
keywords = keyword_creator(masked_sentence, word_deletion=True, criteria="random", min_key_words=3)

pred_2 = make_predictions(masked_sentence, filter_list_final(merged_pages, keywords))