import sys
import os
import json
import torch
import numpy as np
import pandas as pd

path = "\\".join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.insert(0, path)
from ModelAuthorConvolution.author_matrices import concatenate_abstracts
from models import MLP, BERT
from utils import check_running_file


# change according to the actual node embedding dimension
NODE_EMBEDDING_DIM = 64



def make_inputs_targets_list(graph_file="node_embeddings.json", targets_file="train.csv"):
    """ read author BERT embeddings and graph embeddings, concatenate them and
        return them as list, meanwhile, fill a list with targets so that inputs
        and targets are correctly aligned

        also fill a dictionnary with the author who are not in the train list
    """

    bert = BERT()
    bert_sentences = bert.load_sentences()      # dictionnary ob author abstracts embeddings
    print("loading node embeddings...")
    with open(graph_file, "r") as f:
        node_embeddings = json.load(f)
    print("done.")
    node_embeddings = {int(k):v for k, v in node_embeddings.items()}
    target_df = pd.read_csv(targets_file).set_index("authorID")
    test_dic = {}
    
    full_embeddings = []
    targets_list = []
    print("pulling embeddings...")
    for i, auth in enumerate(bert_sentences.keys()):
        if i % 10 == 0 and i > 0:
            print("{} authors processed".format(i))
        auth = int(auth)
        bert_embedding = bert.to_embedding(bert_sentences[auth])
        try:
            h_index = target_df.loc[auth]['h_index']
        except KeyError:
            try:
                test_dic[auth] = torch.tensor(bert_embedding + node_embeddings[auth])
                continue
            except KeyError:
                continue
        try:
            node_embedding = node_embeddings[auth]
        except KeyError:
            continue
        full_embeddings.append(torch.tensor(bert_embedding + node_embedding))
        targets_list.append(h_index)

    return (full_embeddings, targets_list), test_dic


def make_predictions(
    model,
    test_dic,
    input_file="test.csv",
    output_file="test_predictions.csv"
):
    """ create a csv file containing the author indexes and the h-index predictions associated """
    
    path_to_predictions = "ModelBert\\data"
    try:
        os.mkdir(path_to_predictions)
    except FileExistsError:
        pass

    preds = []
    df_test = pd.read_csv(input_file)
    for i, auth in enumerate(df_test['authorID']):
        if i % 20000 == 0 and i > 0:
            print("{} authors processed".format(i))
        emb = test_dic[auth]
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

    # check if author_abstracts.txt file exists, else create it
    if not os.path.exists("author_abstracts.txt"):
        concatenate_abstracts()

    bert = BERT()

    node_embedding_dim = NODE_EMBEDDING_DIM
    bert_embedding_dim = bert.dimension
    embedding_dim = node_embedding_dim + bert_embedding_dim
    
    # launch network
    model = MLP(embedding_dim=embedding_dim, path_to_checkpoints="ModelBert\\checkpoints")
    if not model.existing_model():
        TRAIN = True

    # train model
    if TRAIN:
        (inputs_list, targets_list), test_dic = make_inputs_targets_list()
        print("fitting model")
        model.train()
        CHECKPOINT_FILE = "BERT_checkpoint_1.pt"
        model.fit(inputs_list, targets_list, n_epochs=40, batch_size=32, save_file=CHECKPOINT_FILE)
        
    else:
        TO_LOAD = "BERT_checkpoint_1.pt"
        model.load_model(load_file=TO_LOAD)
        print("successfully loaded {}".format(TO_LOAD))

    # build up the predictions and put it in a csv file
    CSV_OUTPUT = "test_predictions_1.csv"
    if MAKE_PREDICTIONS:
        model.eval()
        make_predictions(model, test_dic, output_file=CSV_OUTPUT)

    FILES_TO_TEST = [
        "test_predictions_1.csv"
    ]