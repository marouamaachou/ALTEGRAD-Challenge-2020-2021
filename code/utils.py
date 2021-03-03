import os
import numpy as np

from random import randint
from gensim.models import Word2Vec
from os.path import splitext, exists




def check_running_file(correct_name='code'):
    """ check that the root directory from where python is running is the right one """

    run_path = os.getcwd()
    to_test = run_path.split("\\")[-1]
    return (to_test == correct_name)

def exist_object(file_name):
    """ check the existence of a .pkl pickle file in the objects folder """

    ext = splitext(file_name)[1]
    if ext != ".pkl":
        raise TypeError("only .pkl files are stored in the objects folder, not {}".format(ext))
    path_to_objects = __file__.split("\\")[:-2]
    path_to_objects.append("objects")
    path_to_objects = "\\".join(path_to_objects)
    return exists(path_to_objects + "\\" + file_name)






""" deep walk utils """
def random_walk(G, node, walk_length):
    
    walk = [node]
    for i in range(walk_length):
        neighbors = list(G.neighbors(walk[-1]))
        if neighbors:
            walk.append(neighbors[randint(0, len(neighbors)-1)])
    
    walk = [str(node) for node in walk]
    return walk

# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
 
    for i in range(num_walks):
        permuted_nodes = np.random.permutation(list(G.nodes()))
        for node in permuted_nodes:
            walks.append(random_walk(G, node, walk_length))
    
    return walks

# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
