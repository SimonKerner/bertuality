import BERTuality as b

tokenizer = b.tokenizer
model_pre = b.model_pre


# load pages
page_1 = b.wikipedia_loader("Coronavirus", "text")
page_2 = b.wikipedia_loader("SARS-CoV-2", "text")
page_3 = b.wikipedia_loader("COVID-19", "text")
page_4 = b.wikipedia_loader("COVID-19_pandemic", "text")
page_5 = b.wikipedia_loader("COVID-19_pandemic_in_Europe", "text")

# filter pages
filtered_pages = b.text_filter(page_1, page_2, page_3, page_4, page_5)
merged_pages = b.merge_pages(filtered_pages)

# predict
pred = b.make_predictions("Covid is a [MASK]", b.filter_list_final(merged_pages, ["Covid", "Virus", "China"]))

