"""
In this file, we generate word embeddings using gensim.models.Word2Vec,
that is, using a skip-gram model in this case.
"""

import json
import os
import re
import ast
import gensim
from gensim.models import word2vec
import numpy as np

from nltk.corpus import stopwords


PATH_TO_DATA = "ModelAuthorConvolution\\data"

pattern = re.compile(r'(,){2,}')
stop_words = set(stopwords.words('english'))


def write_sentences():
    """ extract sentences to feed Word2Vec model, and store them in txt file """
    
    f = open("abstracts.txt","r",encoding='utf8')
    fw = open(os.path.join(PATH_TO_DATA, "abstracts_sentences.txt"),"w",encoding='utf8')

    # load the inverted abstracts and store them as id-abstracts in a txt file
    dic = {}
    print('processing the abstracts...')
    for j, l in enumerate(f):
        if j % 100000 == 0:
            print('{}-th abstract being processed...'.format(j//2 + 1))
        if(l=="\n"):
            continue
        id = l.split("----")[0]
        dic[id] = []
        inv = "".join(l.split("----")[1:])
        res = ast.literal_eval(inv)
        abstract =[ "" for k in range(res["IndexLength"])]
        inv_indx=  res["InvertedIndex"]

        for i in inv_indx:
            try:
                if (i.isalpha() or i[-1] in (',','.',':')) and i not in stop_words:
                    for j in inv_indx[i]:
                        if i[-1] in (',',':'):
                            abstract[j] = i[:-1].lower()
                        else:
                            abstract[j] = i.lower()
            except IndexError:
                pass

        abstract = re.sub(pattern, ',', ",".join(abstract))
        sentences = abstract.split('.')
        for sentence in sentences:
            if sentence == '':
                continue
            if sentence[0] == ',':
                sentence = sentence[1:]
            fw.write(id+'----'+sentence+'\n')
    f.close()
    fw.close()
    print('done processing abstracts.')
    print()

def sentences2list():
    """ make a list of sentences list out of txt file """
    print('collecting sentences...')
    sentences = []
    with open(os.path.join(PATH_TO_DATA, "abstracts_sentences.txt"), "r", encoding='utf8') as f:
        for l in f:
            if l == '\n':
                continue
            sentence = ("".join(l.split('----')[1:])).replace("\n", "").split(',')
            if len(sentence) > 2:
                sentences.append(sentence)
    return sentences

# learn word embeddings
def Word2Vec(sentences, size=100, sg=1):
    """ train a gensim Word2Vec model with embedding dimension = size and return it """

    print('learning embeddings...')
    model = gensim.models.Word2Vec(window=6, min_count=3, size=size, workers=8, sg=sg)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=5)
    print('done learning.')
    return model

# store the embeddings
def store_embeddings_as_json(model):
    print('storing as json dictionnary...')
    f = open(os.path.join(PATH_TO_DATA, "word_embeddings.json"),"w")
    word_vectors = {}
    for word in model.wv.vocab.keys():
        try:
            word_vectors[word] = model.wv[word].tolist()
        except KeyError:
            pass
    json.dump(word_vectors, f)
    f.close()

def store_embeddings_as_txt(model):
    print('storing as text...')
    f = open(os.path.join(PATH_TO_DATA, "word_embeddings.txt"),"w",encoding='utf8')
    for word in model.wv.vocab.keys():
        try:
            f.write(word+":"+np.array2string(model.wv[word],
                        formatter={'float_kind':lambda x: "%.8f" % x})+"\n")
        except KeyError:
            pass
    f.close()

def store_embeddings(model):
    store_embeddings_as_json(model)
    store_embeddings_as_txt(model)

def check_if_exists():
    abspath = os.path.abspath(__file__).split("\\")[:-1]
    abspath.append("data")
    abspath = "\\".join(abspath) + "\\"
    files_to_check = [  
        "abstracts_sentences.txt",
        "word_embeddings.json",
        "word_embeddings.txt"
    ]
    return all([os.path.exists(abspath + f) for f in files_to_check])

def make_embeddings(embedding_dim=100, make_sentences=False):
    """ cascade the previous functions in order to create the embeddings data """

    if make_sentences: write_sentences()
    try:
        sentences = sentences2list()
    except ModuleNotFoundError:
        write_sentences()
        sentences = sentences2list()
    model = Word2Vec(sentences, size=embedding_dim)
    store_embeddings(model)
