import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re

# read the file to create a dictionary with author key and paper list as value
f = open("author_papers.txt","r")
papers_set = set()
d = {}
for l in f:
    auth_paps = [paper_id.strip() for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    d[l.split(":")[0]] = auth_paps
f.close()

# read the embeddings of each paper
f = open("paper_embeddings.txt","r")
papers = {}
s = ""
pattern = re.compile(r'(\s){2,}')
for l in f:
    if(":" in l and s!=""):
        papers[s.split(":")[0]] = np.array(ast.literal_eval(re.sub(pattern, ',', s.split(":")[1]).replace(" ",",")))
        s = l.replace("\n","")
    else:
        s = s+" "+l.replace("\n","")
    
f.close()

# the author representation is set to be the average of its papers' representations
pattern = re.compile(r'(,){2,}')
df = open("author_embedding.csv","w")
for author in d:
    v = np.zeros(256)
    c = 0
    for paper in d[author]:
        try:
            v+=papers[paper]
            c+=1
        except:
            continue
    if(c==0):
        c=1
    df.write(author+","+",".join(map(lambda x:"{:.8f}".format(round(x, 8)), v/c))+"\n")
    
df.close()