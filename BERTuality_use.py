from BERTuality import wikipedia_loader
from BERTuality import text_filter
from BERTuality import merge_pages
from BERTuality import make_predictions
from BERTuality import filter_list_final


# load pages
page_1 = wikipedia_loader("Coronavirus", "text")
page_2 = wikipedia_loader("SARS-CoV-2", "text")
page_3 = wikipedia_loader("COVID-19", "text")
page_4 = wikipedia_loader("COVID-19_pandemic", "text")
page_5 = wikipedia_loader("COVID-19_pandemic_in_Europe", "text")

# filter pages
filtered_pages = text_filter(page_1, page_2, page_3, page_4, page_5)
merged_pages = merge_pages(filtered_pages)

# predict
pred_1 = make_predictions("Covid is a [MASK]", filter_list_final(merged_pages, ["Covid", "Virus", "China"]))




pred_2 = make_predictions("Lionel Messi was born in [MASK].", filter_list_final(merged_pages, [["Lionel","Messi","was","born","in"],["Messi","was","born","in"],["Messi","born","in"]]))