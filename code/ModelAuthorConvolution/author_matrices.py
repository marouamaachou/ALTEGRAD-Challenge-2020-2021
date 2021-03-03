"""
In this file, we generate matrices representations of authors of the shape
(m,dim), where m is the total number of words in the author's abstracts and
dim is the word embedding dimension.
"""

import ast
import os
import numpy as np
import pickle as pkl

from os import path

from utils import exist_object



def abstracts_to_dict(path="abstracts_processed.txt", save=False):
    """ save a dictionnary containing the infos {abstract_id: [word 1, word 2, ...]}
        
        if save is set to True, save a pickle version of the dictionnary at 
        "objects\\abstracts_dict.pkl"
    """

    try:
        f = open(path, "r", encoding="utf8")
    except FileNotFoundError as e:
        print("the file \"{}\" does not exist yet (abstracts have not been processed) ; "
                "run paper_representations.py first")
        raise e
    num_lines = len(f.readlines())
    f.seek(0)
    dic = {}
    for i, l in enumerate(f):
        if l == "\n":
            continue
        if i % 50000 == 0 and i != 0:
            print("{} / {} have been processed".format(i, num_lines))
        id, content = l.split('----')
        id = int(id)                        # author id
        content = content.replace("\n", "")[:-1]
        content = content.split(',')
        dic[id] = content
    f.close()
    if save:
        try:
            with open("objects\\abstracts_dict.pkl", "wb") as f_out:
                pkl.dump(dic, f_out)
        except ModuleNotFoundError:
            os.mkdir("objects")
            with open("objects\\abstracts_dict.pkl", "wb") as f_out:
                pkl.dump(dic, f_out)
    return dic

def authors_to_dict(file="author_papers.txt", save=False):
    """ save a dictionnary containing the infos {author: [paper_id 1, paper_id 2, ...]}
        
        if save is set to True, save a pickle version of the dictionnary at 
        "objects\\author_papers_dict.pkl"
    """

    f = open(file, "r", encoding="utf8")
    num_lines = len(f.readlines())
    f.seek(0)
    dic = {}
    for i , l in enumerate(f):
        if l == "\n":
            continue
        if i % 20000 == 0 and i != 0:
            print("{} / {} have been processed".format(i, num_lines))
        id, content = l.split(':')
        id = int(id)                    # paper id
        content = content.replace("\n", "")
        content = ast.literal_eval(content)
        dic[id] = content
    f.close()
    if save:
        with open("objects\\author_papers_dict.pkl", "wb") as f_out:
            pkl.dump(dic, f_out)
    return dic

def concatenate_abstracts(
    author_paps_read="author_papers.txt",
    abstracts_read="abstracts_processed.txt",
    file_write="author_abstracts.txt"
):
    """ write a txt file containing concatenation of all abstracts per author
        the format is "Author_ID----[word 1, word 2, ...]"
    """
    
    print("create author sequences of abstracts...")
    if not exist_object(file_name="abstracts_dict.pkl"):
        print("abstracts : txt --> dic...")
        abs_dic = abstracts_to_dict(abstracts_read, save=True)
        print()
    else:
        print("loading abstracts...")
        print()
        with open("objects\\abstracts_dict.pkl", "rb") as f:
            abs_dic = pkl.load(f)
    
    if not exist_object(file_name="author_papers_dict.pkl"):
        print("authors paper list : txt --> dic....")
        author_dic = authors_to_dict(author_paps_read, save=True)
    else:
        print("loading authors...")
        with open("objects\\author_papers_dict.pkl", "rb") as f:
            author_dic = pkl.load(f)
    
    f_out = open(file_write, "w", encoding="utf8")
    n_auth = len(author_dic)
    for i, (auth, abs_list) in enumerate(author_dic.items()):
        full_abs = []
        if i % 20000 == 0:
            print("{} / {} authors processed".format(i, n_auth))
        for abs_id in abs_list:
            try:
                abstract = abs_dic[abs_id]
            except KeyError:
                continue
            full_abs = full_abs + abstract

        f_out.write(str(auth) + ":" + ",".join(full_abs) + "\n")
    print("done.")
    f_out.close()
