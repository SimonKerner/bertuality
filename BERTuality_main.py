from BERTuality_loader import wikipedia_loader
from BERTuality_loader import NewsAPI_loader
from BERTuality_loader import guardian_loader
from BERTuality import sentence_converter
from BERTuality import merge_pages
from BERTuality import make_predictions
from BERTuality import filter_list_final
from BERTuality import keyword_creator


#Wikipedia Covid Test
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

# Wikipedia and news api test
"""
page_6 = wikipedia_loader("Bob_Bennett", "text") 
page_7 = NewsAPI_loader("Bob Bennett AND Robert Foster Bennett")

filtered_pages = sentence_converter(page_6)
filtered_pages.append(page_7)               # append, weil der NewsAPI_loader eigene Satzfilter verwendet; hier gibt es andere Suffixe und Präfixe als bei Wikipedia z.B.
merged_pages = merge_pages(filtered_pages)

masked_sentence = "Robert Foster Bennett is a [MASK] of the United States by profession."
keywords = keyword_creator(masked_sentence, word_deletion=True, criteria="shortest", min_key_words=2)

pred_2 = make_predictions(masked_sentence, filter_list_final(merged_pages, keywords))
"""

#wikipedia and news api test

page_6 = wikipedia_loader("Niko_Kovač", "text") 
page_7 = NewsAPI_loader("Niko Kovac")

filtered_pages = sentence_converter(page_6, page_7)
merged_pages = merge_pages(filtered_pages)

masked_sentence = "Niko Kovač is a german football [MASK]."
keywords = keyword_creator(masked_sentence, word_deletion=True, criteria="shortest", min_key_words=2)

pred_2_keyword_creator = make_predictions(masked_sentence, filter_list_final(merged_pages, keywords))
### bei Keywords: Kovač problematisch, weil manche News Seiten c und andere č schreiben; deswegen Sonderzeichen weglassen bei Keyword Suche; Kova ist in Kovač und funktioniert deswegen
pred_2_own_keywords = make_predictions(masked_sentence, filter_list_final(merged_pages, [['Niko','Kova','german','football'],['Kova','german','football'],['Kova','football']]))

"""
# guardian test
query, query_df = guardian_loader(from_date="2022-08-01", to_date="2022-12-15", query="Scholz")
#path, path_df = guardian_loader(from_date="2022-08-01", to_date="2022-12-15", path="politics")

filtered_query = sentence_converter(query)
#filtered_path = sentence_converter(path)

merged_query = merge_pages(filtered_query)
#merged_path = merge_pages(filtered_path)

masked = "Chancellor Merkel is the [MASK] leader of Germany."
masked_2 = "[MASK] is the chancellor of germany"
key_words = ["Chancellor", "Merkel"]
key_words_2 = ["Chancellor", "scholz"]


query_pred = make_predictions(masked_2, filter_list_final(merged_query, key_words_2))
#path_pred = make_predictions(masked, filter_list_final(merged_path, key_words))
"""





