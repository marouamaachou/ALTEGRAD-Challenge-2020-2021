"""
In this file, we generate word embeddings using gensim.models.Word2Vec,
that is, using a skip-gram model in this case.
"""

import json
import os
import re
import ast
import gensim
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

def make_vocab(sentences):
    """ store vocabulary in a list in a json file and return it """

    print("collecting vocab...")
    vocab = []
    for sentence in sentences:
        for word in sentence:
            if not word in vocab and word.isalpha():
                vocab.append(word)
    with open(os.path.join(PATH_TO_DATA, "vocab.json"), 'w') as f:
        json.dump(vocab, f)
    return vocab

# learn word embeddings
def Word2Vec(sentences, size=50):
    """ train a gensim Word2Vec model with embedding dimension = size and return it """

    print('learning embeddings...')
    model = gensim.models.Word2Vec(sentences, min_count=5, size=size)
    print('done learning.')
    return model

# store the embeddings
def store_embeddings_as_json(model, vocab):
    print('storing as json dictionnary...')
    f = open(os.path.join(PATH_TO_DATA, "word_embeddings.json"),"w")
    word_vectors = {}
    for word in vocab:
        try:
            word_vectors[word] = model.wv[word].tolist()
        except KeyError:
            pass
    json.dump(word_vectors, f)
    f.close()

def store_embeddings_as_txt(model, vocab):
    print('storing as text...')
    f = open(os.path.join(PATH_TO_DATA, "word_embeddings.txt"),"w",encoding='utf8')
    for word in vocab:
        try:
            f.write(word+":"+np.array2string(model.wv[word], formatter={'float_kind':lambda x: "%.8f" % x})+"\n")
        except KeyError:
            pass
    f.close()

def store_embeddings(model, vocab):
    store_embeddings_as_json(model, vocab)
    store_embeddings_as_txt(model, vocab)

def check_if_exists():
    abspath = os.path.abspath(__file__).split("\\")[:-1]
    abspath.append("data")
    abspath = "\\".join(abspath) + "\\"
    files_to_check = [
        "abstracts_sentences.txt",
        "vocab.json",
        "word_embeddings.json",
        "word_embeddings.txt"
    ]
    return all([os.path.exists(abspath + f) for f in files_to_check])

def make_embeddings(make_sentences=False, make_vocab=False, embedding_dim=50):
    """ cascade the previous functions in order to create the embeddings data """

    if make_sentences: write_sentences()
    try:
        sentences = sentences2list()
    except ModuleNotFoundError:
        write_sentences()
        sentences = sentences2list()
    if make_vocab: vocab = make_vocab(sentences)
    else:
        with open(os.path.join(PATH_TO_DATA, "vocab.json"), "r") as f:
            vocab = json.load(f)
    model = Word2Vec(sentences, size=embedding_dim)
    store_embeddings(model, vocab)
