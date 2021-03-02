import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ast 

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords 

import re
pattern = re.compile(r'(,){2,}')

stop_words = set(stopwords.words('english')) 

fw = open("abstracts_processed.txt","w")
f = open("abstracts.txt","r")

# loads the inverted abstracts and stores them as id-abstracts in a dictionary dic and in a folder fw
dim = 256
dic = {}
for l in f:
    if(l=="\n"):
        continue
    id = l.split("----")[0]
    inv = "".join(l.split("----")[1:])
    res = ast.literal_eval(inv) 
    abstract =[ "" for i in range(res["IndexLength"])]
    inv_indx=  res["InvertedIndex"]

    for i in inv_indx:
        if i.isalpha() and i not in stop_words:
            for j in inv_indx[i]:
                abstract[j] = i.lower()
    abstract = re.sub(pattern, ',', ",".join(abstract))
    fw.write(id+"----"+abstract+"\n")
    dic[id] = abstract
fw.close()

# cleans the abstracts from stopwords, numeric and non legible characters
doc = []
for i in dic:
    p = dic[i].split(",")
    dic[i] = [l for l in p if l.isalpha() and l not in stop_words]
    doc.append(dic[i])

# learns the embeddings of each abstract 
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(doc)]
del doc
model = Doc2Vec(tagged_data, vector_size = dim, window = 5, min_count = 2, epochs = 100, workers=10)

# store the embeddings in "paperID":array format
f = open("paper_embeddings.txt","w")
for tid in dic:
    sentence = dic[tid]
    f.write(str(tid)+":"+np.array2string(model.infer_vector(sentence), formatter={'float_kind':lambda x: "%.8f" % x})+"\n")    
f.close()