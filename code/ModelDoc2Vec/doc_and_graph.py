import sys
import os
import json
import pandas as pd

path = "\\".join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.insert(0, path)
from models import GNN, DeepWalk
from utils import check_running_file




def merge_embeddings(auth_file="author_embeddings.csv", graph_file="node_embeddings.json"):
    """ read author embeddings and graph embeddings, concatenate them and return them as list """

    auth_df = pd.read_csv(auth_file)
    with open(graph_file, "r") as f:
        embeddings = json.load(f)


def make_predictions(model):
    pass