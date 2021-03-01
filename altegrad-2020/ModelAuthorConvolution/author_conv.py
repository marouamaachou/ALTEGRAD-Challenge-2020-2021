import os
import sys
import csv
import torch
import pickle as pkl
import json
from random import shuffle

path = "\\".join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.insert(0, path)
from models import AuthorConvNet
from dummy.utils import check_running_file
from ModelAuthorConvolution.word_embeddings import *
from ModelAuthorConvolution.author_matrices import *


class AuthorConvData:
    """ a data class to store formatted inputs and targets for the author
        convolutional network

        attributes:
            max_input_size: number of lines in the input_matrix
            input_file: .txt file whose lines are of the format "author_id:word1,word2..."
            path_to_data: where to find relevant data
        where the words are the concatenation of all author's abstracts
            target_file: should be a csv with columns as (author_id, h-index) ;
        path to read the file containing targets
    """

    def __init__(
        self,
        max_input_size=500,
        input_file="author_abstracts.txt",
        target_file="train.csv",
        node_file="node_embeddings.txt",
        path_to_data = "ModelAuthorConvolution\\data"
    ):
        self.max_input_size = max_input_size
        self.input_file = input_file
        self.target_file = target_file
        self.path_to_data = path_to_data

    def make_embedding_matrix(self, full_abstract):
        """ turn a full abstract (a text concatenation of all abstracts from an author)
            into a matrix of word embeddings
        """

        try:
            embeddings = self.word_embeddings
        except AttributeError as e:
            print("no embeddings have been loaded yet ; run load_word_embeddings first")
            raise e
        embedding_dim = self.embedding_dim
        abstract_list = full_abstract.split(",")
        matrix = torch.zeros(size=(self.max_input_size, embedding_dim))
        for i, word in enumerate(abstract_list):
            if i >= self.max_input_size: break
            try:
                embedding = torch.tensor(embeddings[word])
            except KeyError:
                continue
            matrix[i] = embedding
        return matrix

    def load_input_target_list(self):
        """ load the input and target data so that they are correctly aligned """

        f_input = open(self.input_file, "r", newline='', encoding="utf8")
        target_dic = self.load_target_file()
    
        list_input = []
        list_target = []
        for l in f_input:
            auth, full_abstract = l.split(":")
            auth = int(auth)
            try:
                list_target.append(target_dic[auth])
            except KeyError:
                continue
            full_abstract = full_abstract[:-1]      # skip '\n' char
            list_input.append(full_abstract)
        f_input.close()
        self.list_input = list_input
        self.list_target = list_target
        return list_input, list_target

    def train_test_split(self, ratio=0.8):
        try:
            list_input, list_target = self.list_input, self.list_target
        except AttributeError:
            list_input, list_target = self.load_input_target_list()
        random_indexes = list(range(len(list_target)))
        shuffle(random_indexes)
        split = int(ratio * len(random_indexes))
        list_input_train = []
        list_input_test = []
        list_target_train = []
        list_target_test = []
        for index in random_indexes[:split]:
            list_input_train.append(list_input[index])
            list_target_train.append(list_target[index])
        for index in random_indexes[split:]:
            list_input_test.append(list_input[index])
            list_target_test.append(list_target[index])
        return (list_input_train, list_target_train,
                list_input_test, list_target_test)

    def make_input_batch(self, batch_indexes):
        """ return an array of embbedding matrices with the batch indexes """
        
        to_return = torch.empty(size=(len(batch_indexes), self.max_input_size, self.embedding_dim))
        for i, index in enumerate(batch_indexes):
            full_abstract = self.list_input[index]
            to_return[i] = self.make_embedding_matrix(full_abstract)
        return to_return

    def load_word_embeddings(self, file="word_embeddings.json"):
        with open(self.path_to_data + "\\" + file, "r") as f:
            dic = json.load(f)
        self.word_embeddings = dic
        dim = len(list(dic.values())[0])
        self.word_embedding_dim = dim

    def load_node_embeddings(self, file="node_embeddings.json"):
        with open(file, "r") as f:
            dic = json.load(f)
        self.node_embeddings = dic
        dim = len(list(dic.values())[0])
        self.node_embedding_dim = dim

    def load_target_file(self):
        " turn target file into a dictionnary and return it "

        dic = {}
        with open(self.target_file, "r", newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)      # skip the header
            for row in reader:
                auth, h_index = row
                auth, h_index = int(auth), int(h_index)
                dic[auth] = h_index
        return dic




def make_predictions(
    model,                      # an AuthorConvNet instance
    data,                    # an AuthorConvData instance
    input_file="test.csv", 
    abstracts_file="author_abstracts.txt",
    output_file="test_predictions.csv"
):
    """ create a csv file containing the author indexes and the h-index predictions associated """

    path_to_predictions = "ModelAuthorConvolution\\data"
    f_in = open(input_file, "r", newline='')
    f_abs = open(abstracts_file, "r", encoding="utf8")
    f_out = open(os.path.join(path_to_predictions, output_file), "w", newline='')
    list_abstracts = f_abs.readlines()
    n_auths = len(f_in.readlines())
    f_in.seek(0)
    dic_abstracts = {}

    for l in list_abstracts:
        auth_id, abs = l.split(":")
        auth_id = int(auth_id)
        abs = abs[:-1]              # skip the \n char
        dic_abstracts[auth_id] = abs
    
    reader = csv.reader(f_in)
    fieldnames = ['authorID', 'h_index_pred']
    dic_writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    next(reader, None)      # skip the header
    dic_writer.writeheader()

    print("making predictions...")
    for i, row in enumerate(reader):
        auth_id = row[0]
        auth_id = int(auth_id)
        if i % 20000 == 0 and i > 0:
            print("{} / {} authors processed".format(i, n_auths))
        try:
            full_abstract = dic_abstracts[auth_id]
        except KeyError:
            dic_writer.writerow({'authorID':auth_id, 'h_index_pred':"NAN"})
        matrix = data.make_embedding_matrix(full_abstract)
        prediction = model.predict(matrix)
        dic_writer.writerow({'authorID':auth_id, 'h_index_pred':round(prediction.item(), 3)})
        
    f_in.close()
    f_abs.close()
    f_out.close()





if __name__ == "__main__":
    
    # whether to train the model or not
    TRAIN = True

    # should run from "altegrad-2020"
    TO_RUN_FROM = "altegrad-2020"
    if not check_running_file(TO_RUN_FROM):
        raise OSError("the file should run from \"{}\"".format(TO_RUN_FROM))

    # check if embeddings have been created, else create them
    if not check_if_exists():
        make_embeddings(make_vocab=True)

    # check if author_abstracts.txt file exists, else create it
    if not os.path.exists("author_abstracts.txt"):
        concatenate_abstracts()

    # prepare data
    data = AuthorConvData()
    print("loading word embeddings...")
    data.load_word_embeddings()
    if TRAIN:
        print("making inputs and targets lists...")
        data.load_input_target_list()
        print("done.")

    embedding_dim = data.word_embedding_dim

    # launch network
    model = AuthorConvNet(embedding_dim=embedding_dim, min_conv_size=1, max_conv_size=4, hidden_dim=200)
    if not model.existing_model():
        TRAIN = True
    
    # train model
    if TRAIN:
        print("fitting model")
        model.train()
        CHECKPOINT_FILE = "author_conv_checkpoint_8.pt"
        model.fit(data=data, n_epochs=15, batch_size=64,
                    save_file=CHECKPOINT_FILE)
        
    else:
        TO_LOAD = "author_conv_checkpoint_8.pt"
        model.load_model(load_file=TO_LOAD)

    # build up the predictions and put it in a csv file
    CSV_OUTPUT = "test_predictions_8.csv"
    model.eval()
    make_predictions(model, data, output_file=CSV_OUTPUT)

    FILES_TO_TEST = [
        "test_predictions_8.csv"
    ]