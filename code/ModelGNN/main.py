import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

path = "\\".join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.insert(0, path)
from models import MLP, DeepWalk
from utils import check_running_file



def make_predictions(
    model,
    input_file="test.csv",
    embeddings_file="node_embeddings.json",
    output_file="test_predictions.csv"
):
    """ create a csv file containing the author indexes and the h-index predictions associated """

    path_to_predictions = "ModelGNN\\data"
    try:
        os.mkdir(path_to_predictions)
    except FileExistsError:
        pass
    try:
        with open(embeddings_file, "r") as f:
            embeddings = json.load(f)
    except ModuleNotFoundError as e:
        print("the file {} does not exist yet ; run DeepWalk.deepwalk first".format(embeddings_file))
        raise e
    embeddings = {int(k):v for k, v in embeddings.items()}
    
    preds = []
    df_test = pd.read_csv(input_file)
    for i, auth in enumerate(df_test['authorID']):
        if i % 50000 == 0 and i > 0:
            print("{} nodes processed".format(i))
        emb = torch.tensor(embeddings[auth])
        preds.append(model.predict(emb))
    preds = np.array(preds)
    df_test['h_index_pred'].update(pd.Series(np.round_(preds, decimals=3)))
    df_test.loc[:,["authorID","h_index_pred"]].to_csv(
        os.path.join(path_to_predictions, output_file), index=False
    )






if __name__ == "__main__":

    # whether to train the model or not
    TRAIN = True

    # whether to make predictions or not
    MAKE_PREDICTIONS = True

    # should run from "altegrad-2020"
    TO_RUN_FROM = "code"
    if not check_running_file(TO_RUN_FROM):
        raise OSError("the file should run from \"{}/\" folder".format(TO_RUN_FROM))
    
    dw = DeepWalk()
    
    # check if node_embeddings.json file exists, else create it
    if not os.path.exists("node_embeddings.json"):
        dw.deepwalk()
    else:
        print("loading embeddings from deep walk...")
        dw.load_embeddings()

    embedding_dim = dw.embedding_dim

    # launch network
    model = MLP(embedding_dim=embedding_dim)
    if not model.existing_model():
        TRAIN = True

    # train model
    if TRAIN:
        inputs_list, targets_list = model.make_inputs_targets()
        print("fitting model")
        model.train()
        CHECKPOINT_FILE = "GNN_checkpoint_1.pt"
        model.fit(inputs_list, targets_list, n_epochs=15, batch_size=64, save_file=CHECKPOINT_FILE)
        
    else:
        TO_LOAD = "GNN_checkpoint_1.pt"
        model.load_model(load_file=TO_LOAD)
        print("successfully loaded {}".format(TO_LOAD))

    # build up the predictions and put it in a csv file
    CSV_OUTPUT = "test_predictions_1.csv"
    if MAKE_PREDICTIONS:
        model.eval()
        make_predictions(model, output_file=CSV_OUTPUT)

    FILES_TO_TEST = []