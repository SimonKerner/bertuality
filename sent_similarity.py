from sentence_similarity import sentence_similarity
import pandas as pd

#model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')
model_sim=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='sentence_embedding')



sents = ["Lionel Messi was born in.",
         "Lionel Messi was born in.",
         "LioneL MeSSi was boRn in.",
         "Messi was born in.",
         "Lionel Messi was born in",
         "Messi was born in Rosario.",
         "Lionel Messi was in Rosario.",
         "was born in Rosario.",
         "Lionel Messi was born.",
         "was in.",
         "Lionel Messi.",
         "born.",
         "Lionel Messi was born in France.",
         "Lionel Messi was not born in France.",
         "Lionel Messi was born in Rosario, Argentina.",
         "Lionel Messi was born in Rosario, Argentina, in 1987.",
         "Lionel Messi was born in Rosario, Argentina, in 1987, In foreign policy, Merkel has emphasised international cooperation, both in the context of the EU and NATO, and strengthening transatlantic economic relations.",
         "In foreign policy, Merkel has emphasised international cooperation, both in the context of the EU and NATO, and strengthening transatlantic economic relations.",
         "His favorite color is blue.",
         "x"]

sents2 = ["Lionel Messi was born in Rosario.",
          "Lionel Messi is from Rosario.",
          "Lionel Messi's hometown is Rosario.",
          "Lionel Messi grew up in Rosario.",
          "Lionel Messi spent his childhood in Rosario.",
          "Lionel Messi was ##not born in Rosario.",
          "Lionel Messi now lives in France.",
          "Lionel Messi's son was born in Madrid.",
          "Lionel Messi's son was born in Rosario.",
          "Lionel Messi's favorite color is yellow."]


columns = ['sentence', 'cosine_similarity']
sent_sim = pd.DataFrame(columns = columns)

sent_sim['sentence'] = sents2

sim = []
for sentence in sents2:
    sim_sc = model_sim.get_score(sentence,"Lionel Messi was born in Rosario.",metric="cosine")
    sim.append(sim_sc)
    
    
sent_sim['cosine_similarity'] = sim
