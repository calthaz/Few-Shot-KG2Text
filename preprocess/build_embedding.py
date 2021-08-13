import torch
import pickle
import numpy as np
#from transformers import AutoTokenizer, AutoModelForMaskedLM

from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("uer/bart-base-chinese-cluecorpussmall")

#model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
model = BartForConditionalGeneration.from_pretrained("uer/bart-base-chinese-cluecorpussmall")
#tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')#../../pretrained_model/bart-large
embedding = model.get_input_embeddings().weight

vocab = pickle.load(open("node.pkl", "rb"))

my_embedding = []
my_iddx = set()
for token, idx in vocab.items():
    iddx = tokenizer.convert_tokens_to_ids([token])[0]
    my_iddx.add(iddx)
    my_embedding.append(embedding[iddx])

my_embedding = torch.stack(my_embedding, dim=0).detach().numpy()
np.save("node_embeddings.npy", my_embedding)