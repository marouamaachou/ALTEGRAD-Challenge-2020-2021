""" In this file, we develop the base classes associated to the models
    we experiments.

    Models currently available:
        Author Convolutions ; AuthorConvNet
        PLSA ; PLSA (obsolete)
        Deep Walk ; DeepWalk
        Graph MLP ; MLP
        BERT ; BERT
        Transformer ; transformer_encoder
        Encoder-decoder
"""


import os
import csv
import plsa
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf

from tensorflow.keras.layers import Attention, LSTM , Dense, GRU, Conv1D, Dropout, Attention, Concatenate
from tensorflow.keras import Sequential
from tqdm import tqdm
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel, file_utils
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AuthorConvNet(nn.Module):
    """ Class to implement the network architecture as proposed in
        "Non Linear Text Regression With Deep Convolutional Neural Network",
        Bitvai and Cohn (https://www.aclweb.org/anthology/P15-2030.pdf)

        After training, checkpoints may be found at ModelAuthorConvolution\\checkpoints ;
        these can be loaded with the load_model method
    """

    def __init__(
        self,
        word_embedding_dim,
        hidden_dim=200,
        dropout_rate=0.3,
        min_conv_size=1,
        max_conv_size=4,
        criterion=nn.L1Loss(),
        optimizer = optim.Adadelta,
        path_to_checkpoints="ModelAuthorConvolution\\checkpoints",
        use_graph=False,      # use node embeddings (True) or not
        node_embedding_dim=None     # should be provided if use_graph=True,
                                    # otherwise an error will be raised
    ):
        """ This architecture performs 2D convolutions using 1D kernels with 
            size min_conv_size, min_conv_size+1, ..., max_conv_size
        """
        
        super(AuthorConvNet, self).__init__()
        if use_graph:
            assert node_embedding_dim is not None, (
                "if use_graph is True, a valid embedding_dim integer should be provided"
            )
        if not node_embedding_dim: node_embedding_dim = 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.dropout = nn.Dropout(dropout_rate)
        self.min_conv_size = min_conv_size
        self.max_conv_size = max_conv_size
        self.max_pool = lambda x: torch.max(x, dim=len(x.size())-2)[0]    # max_pool here is the max over the input raws

        self.word_embedding_dim = word_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hidden_dim = hidden_dim
        # min_conv_size,...,max_conv_size 1D convolutional filters of size 1,...,max_conv_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,1,(i,1)) for i in range(min_conv_size, self.max_conv_size+1)]
        )
        self.relu = nn.ReLU()
            
        self.linear1 = nn.Linear(
            word_embedding_dim * (max_conv_size-min_conv_size+1) + node_embedding_dim,
            hidden_dim
        )
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.path_to_checkpoints = path_to_checkpoints
        self.use_graph = use_graph

    def forward(self, x, x_node=None):
        if self.use_graph:
            assert x_node is not None, (
                "x_node argument should be provided as a batch of node embeddings"
            )
        z = x.unsqueeze(1)   # second dimension corresponds to in_channels, 
                             # so it should be 1 (1st is batch size)
        C = []
        for convolution in self.convs:       # apply convolution for each kernel size between min and max
            conv = convolution(z).squeeze()
            C.append(conv)
        H = [self.relu(conv) for conv in C]
        p = [self.max_pool(h) for h in H]
        p = torch.cat(p, dim=-1)
        # if use_graph, concatenate author convolution embedding with author node embedding
        if self.use_graph:
            p = torch.cat((p, x_node), dim=-1)

        # multi-layer net
        out = self.linear1(p)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.linear3(out)
        return out

    def set_optimizer(self):
        params = self.parameters()
        self.optimizer = self.optimizer(params)
        return self.optimizer

    def save_model(
        self,
        path=None,
        save_file="author_conv_checkpoint.pt",
    ):
        if path:
            try:
                torch.save(self.state_dict(), path + "\\" + save_file)
            except FileNotFoundError:
                os.mkdir(path)
                torch.save(self.state_dict(), path + "\\" + save_file)
        else:
            try:
                torch.save(self.state_dict(), self.path_to_checkpoints + "\\" + save_file)
            except FileNotFoundError:
                os.mkdir(self.path_to_checkpoints)
                torch.save(self.state_dict(), self.path_to_checkpoints + "\\" + save_file)

    def load_model(
        self,
        path=None,
        load_file="author_conv_checkpoint.pt",
    ):
        if path:
            self.load_state_dict(torch.load(path + "\\" + load_file))
        else:
            self.load_state_dict(torch.load(self.path_to_checkpoints + "\\" + load_file))
            
    def fit(
        self,
        data,           # an AuthorConvData instance
        n_epochs=10,
        batch_size=64,
        save_checkpoint_every=5,
        save_file="author_conv_checkpoint.pt",
    ):
        """ train the network on the training set [inputs, targets]

            Args:
                data: object of the kind AuthorConvData, correctly initialized
                n_epochs: number of epochs to go through for the training
                batch_size: size of the batches to use (batch optimisation)
                save_checkpoint_every: save model.state_dict() checkpoint after
            this number of epochs (should be smaller than n_epochs)
        """

        optimizer = self.set_optimizer()
        criterion = self.criterion

        targets = torch.LongTensor(data.list_target).float()

        n = len(targets)
        n_batch = n // batch_size + 1 * (n % batch_size != 0)
        tqdm_dict = {"loss": 0.0}

        for epoch in range(n_epochs):
            
            with tqdm(total=n_batch, unit_scale=True, postfix={'loss':0.0},#,'test loss':0.0},
                    desc="Epoch : %i/%i" % (epoch+1, n_epochs), ncols=100) as pbar:
                
                batch_indexes = torch.randperm(n)
                total_loss = 0.0
                
                for i in range(0, n, batch_size):
                    index_batch = batch_indexes[i:(i+batch_size)]
                    abstract_batch, node_batch = data.make_input_batch(index_batch)
                    target_batch = targets[index_batch]
                    # use the node_batch
                    if self.use_graph:
                        output_batch = self.forward(abstract_batch, node_batch)
                    else:
                        output_batch = self.forward(abstract_batch)
                    loss = criterion(output_batch.flatten(), target_batch)
                    total_loss += loss.item()
                    tqdm_dict['loss'] = total_loss / (i+1)
                    pbar.set_postfix(tqdm_dict)

                    # compute gradients and take an optimisation step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
            if (epoch + 1) % save_checkpoint_every == 0:
                self.save_model(save_file=save_file)
                print("model checkpoint saved")
        self.save_model(save_file=save_file)

    def predict(self, author_matrix, node_embedding=None):
        """ predict for a single author embedding matrix """

        if self.use_graph:
            assert node_embedding is not None, (
                "node_embedding argument should be provided as an embedding array"
            )
        with torch.no_grad():
            mat = author_matrix.unsqueeze(dim=0)    # make it a batch of size 1
            prediction = self.forward(mat, node_embedding)
        return prediction
    
    def existing_model(self):
        """ return True if a checkpoint exists in path_to_checkpoint, False otherwise """

        abspath = "\\".join(os.path.abspath(__file__).split("\\")[:-1])
        try:
            if os.listdir(abspath + "\\" + self.path_to_checkpoints): return True
        except FileNotFoundError:
            return False
        return False






class PLSA:
    """ Class to implement "Probabilistic Latent Semantic Analysis" """

    def __init__(self, input_file="author_abstracts.txt"):
        self.txt_input_file = input_file

    def make_input_file(self, out_file="author_abstracts.csv"):
        """ the plsa module needs a csv file to work ; so we make a csv file out of an abstracts file """
        
        f_in = open(self.txt_input_file, "r", encoding="utf8")
        f_out = open(out_file, 'w', newline='')
        writer = csv.writer(f_out)
        
        for l in f_in:
            _, abs = l.split(":")
            abs = abs[:-1].replace(",", " ").encode('utf8')
            writer.writerow([abs])
        self.input_file = out_file

        f_in.close()
        f_out.close()

    def make_corpus(self):
        print("building corpus up...")
        try:
            corpus_ = plsa.corpus.Corpus.from_csv(
                self.input_file, pipeline=plsa.Pipeline(plsa.preprocessors.to_lower)
            )
        except AttributeError:
            self.make_input_file()
            corpus_ = plsa.corpus.Corpus.from_csv(
                self.input_file, pipeline=plsa.Pipeline(plsa.preprocessors.to_lower)
            )
        self.corpus_ = corpus_
        return corpus_

    def launch_plsa(self, n_topics=50):
        try:
            corpus_ = self.corpus_
        except AttributeError:
            corpus_ = self.make_corpus()
        plsa_ = plsa.algorithms.plsa.PLSA(corpus=corpus_, n_topics=n_topics, tf_idf=True)
        print("computing PLSA for the corpus...")
        plsa_.fit()
        print("done.")
        topic_embeddings = plsa_.topic_given_doc.T
        self.topic_embeddings = topic_embeddings
        self.save_embeddings(topic_embeddings)
        return topic_embeddings

    def save_embeddings(self, embeddings, output_file="objects\\topic_embeddings.pkl"):
        """ store the array of "topic given docs" distributions in a pickle .pkl file """

        try:
            with open(output_file, "wb") as f:
                pkl.dump(embeddings, f)
        except FileNotFoundError:
            os.mkdir("objects")
            with open(output_file, "wb") as f:
                pkl.dump(embeddings, f)






class DeepWalk:
    """ Class to implement the deep walk algorithm """

    def __init__(
        self,
        n_walks=10,
        walk_length=30,
        window_size=6,
        embedding_dim=100,
        input_file="collaboration_network.edgelist",
        output_file="node_embeddings.json"
    ):
        self.graph = nx.read_edgelist(
            input_file,
            delimiter=' ',
            nodetype=int,
            create_using=nx.Graph()
        )
        self.n_nodes = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        
        self.walk_length = walk_length
        self.n_walks = n_walks

        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.input_file = input_file
        self.output_file = output_file

    def random_walk(self, node):
        """ simulate a random walk of length "walk_length" starting from node "node" """
        
        walk = [node]
        for _ in range(self.walk_length):
            neighbours = list(self.graph.neighbors(walk[-1]))
            try:
                neighbour_choice = np.random.randint(0, len(neighbours) - 1)
            except ValueError:
                neighbour_choice = 0
            walk.append(neighbours[neighbour_choice])
        walk = [str(node) for node in walk]
        return walk

    def generate_walks(self):
        """ run "n_walks" random walks from each node ; a walk is a list of walks,
            and each walk is itself a list of nodes
        """

        walks = []
        for i in range(self.n_walks):
            permuted_nodes = np.random.permutation(self.graph.nodes())
            for node in tqdm(permuted_nodes):
                walks.append(self.random_walk(node))
            print("{} / {} walks generated".format(i+1, self.n_walks))
        self.walks = walks
        return walks

    def deepwalk(self):
        """ launch the deep walk algorithms, i.e, skip-gram over walks """

        print("Generating walks...")
        try:
            walks = self.walks
        except AttributeError:
            walks = self.generate_walks()
        print("Training word2vec...")
        model = Word2Vec(
            size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=8
        )
        model.build_vocab(walks)
        model.train(walks, total_examples=model.corpus_count, epochs=5)
        dic = {}
        for id in model.wv.vocab.keys():
            dic[id] = model.wv[id].tolist()
        self.embeddings_dic = dic.copy()
        self.save_model()
        return model

    def save_model(self):
        """ store node embeddings as json """

        try:
            dic_to_save = self.embeddings_dic.copy()
        except AttributeError as e:
            print("no embeddings dictionnary to save ; launch deepwalk first")
            raise e
        with open(self.output_file, "w") as f:
            json.dump(dic_to_save, f)

    def load_embeddings(self, load_file=None):
        """ load an embeddings json file """

        dic = {}
        if load_file:
            f = open(load_file, "r")
        else:
            f = open(self.output_file, "r")
        self.embeddings_dic = json.load(f)
        self.embedding_dim = len(list(dic.values())[0])
        f.close()
        return dic






class MLP(nn.Module):
    """ Fully connected neural net with node embedding input
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim=200,
        criterion=nn.L1Loss(),
        optimizer=optim.Adam,
        dropout_rate=0.5,
        path_to_checkpoints="ModelGNN\\checkpoints"
    ):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.criterion = criterion
        self.optimizer = optimizer

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        self.path_to_checkpoints = path_to_checkpoints

    def forward(self, x):
        out = self.linear1(x)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.linear3(out)
        return out

    def set_optimizer(self):
        params = self.parameters()
        self.optimizer = self.optimizer(params)
        return self.optimizer

    def save_model(
        self,
        path=None,
        save_file="GNN_checkpoint.pt",
    ):
        if path:
            try:
                torch.save(self.state_dict(), path + "\\" + save_file)
            except FileNotFoundError:
                os.mkdir(path)
                torch.save(self.state_dict(), path + "\\" + save_file)
        else:
            try:
                torch.save(self.state_dict(), self.path_to_checkpoints + "\\" + save_file)
            except:
                os.mkdir(self.path_to_checkpoints)
                torch.save(self.state_dict(), self.path_to_checkpoints + "\\" + save_file)

    def load_model(
        self,
        path=None,
        load_file="GNN_checkpoint.pt",
    ):
        if path:
            self.load_state_dict(torch.load(path + "\\" + load_file))
        else:
            self.load_state_dict(torch.load(self.path_to_checkpoints + "\\" + load_file))

    def make_inputs_targets(self, input_file="node_embeddings.json", target_file="train.csv"):
        """ make up inputs and targets list from input and target files so that
            both are correctly aligned
        """

        inputs_list, targets_list = [], []
        with open(input_file, "r") as f:
            emb_dic = json.load(f)
        emb_dic = {int(k):v for k,v in emb_dic.items()}
        target_df = pd.read_csv(target_file).set_index("authorID")
        for auth, emb in emb_dic.items():
            try:
                h_index = target_df.loc[auth]['h_index']
            except KeyError:
                continue
            inputs_list.append(torch.tensor(emb))
            targets_list.append(h_index)
        return inputs_list, targets_list
        
    def make_input_batch(self, batch_indexes, inputs_list):
        batch = torch.empty(size=(len(batch_indexes), inputs_list[0].size()[0]))
        for i, ind in enumerate(batch_indexes):
            batch[i] = inputs_list[ind]
        return batch

    def fit(
        self,
        inputs_list,
        targets_list,
        n_epochs=15,
        batch_size=64,
        save_checkpoint_every=5,
        save_file="author_conv_checkpoint.pt",
    ):
        """ train the network on the training set [inputs, targets]

            Args:
                n_epochs: number of epochs to go through for the training
                batch_size: size of the batches to use (batch optimisation)
                save_checkpoint_every: save model.state_dict() checkpoint after
            this number of epochs (should be smaller than n_epochs)
                inputs_list: a list of node embedding arrays
                targets_list: a list of h-index integers
        """

        optimizer = self.set_optimizer()
        criterion = self.criterion

        targets = torch.LongTensor(targets_list).float()

        n = len(targets)
        n_batch = n // batch_size + 1 * (n % batch_size != 0)
        tqdm_dict = {"loss": 0.0}

        for epoch in range(n_epochs):
            
            with tqdm(total=n_batch, unit_scale=True, postfix={'loss':0.0},#,'test loss':0.0},
                    desc="Epoch : %i/%i" % (epoch+1, n_epochs), ncols=100) as pbar:
                
                batch_indexes = torch.randperm(n)
                total_loss = 0.0
                
                for i in range(0, n, batch_size):
                    index_batch = batch_indexes[i:(i+batch_size)]
                    input_batch = self.make_input_batch(index_batch, inputs_list)
                    target_batch = targets[index_batch]
                    
                    output_batch = self.forward(input_batch)
                    loss = criterion(output_batch.flatten(), target_batch)
                    total_loss += loss.item()
                    tqdm_dict['loss'] = total_loss / (i+1)
                    pbar.set_postfix(tqdm_dict)

                    # compute gradients and take an optimisation step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)
            if (epoch + 1) % save_checkpoint_every == 0:
                self.save_model(save_file=save_file)
                print("model checkpoint saved")
        self.save_model(save_file=save_file)

    def predict(self, node_embedding):
        """ predict for a single node embedding vector """

        with torch.no_grad():
            prediction = self.forward(node_embedding)
        return prediction

    def existing_model(self):
        """ return True if a checkpoint exists in path_to_checkpoint, False otherwise """

        abspath = "\\".join(os.path.abspath(__file__).split("\\")[:-1])
        try:
            if os.listdir(abspath + "\\" + self.path_to_checkpoints): return True
        except FileNotFoundError:
            return False
        return False




class BERT:
    """ Class to implement BERT model
        
        for the purpose of this implementation, we use Bert directly on 
        full abstracts concatenation from authors (although it is supposed
        to be used on single or couple of sentences)
    """

    bert_dimension = 7

    def __init__(
        self,
        max_len=500,
        text_file="author_abstracts.txt",
        path_to_data="ModelBert\\data"
    ):
        self.model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states = True, # Whether the model returns all hidden-states.
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_file = text_file
        self.max_len = max_len
        self.dimension = self.model.config.hidden_size
        self.path_to_data = path_to_data

    def load_sentences(self, n_sentences=None):
        """ return a dictionnary out of the sentences file """
        
        print("loading abstracts...")
        dic = {}
        f = open(self.text_file, "r", encoding="utf8")
        for i, l in enumerate(f):
            if n_sentences and i > n_sentences: break
            try:
                pap, sentence = l.split(":")
            except ValueError:
                continue
            pap = int(pap)
            sentence = sentence[:-1].split(",")
            dic[pap] = sentence
        print("done.")
        f.close
        self.sentences = dic
        return dic

    def to_embedding(self, sentence):
        input_ids = self.tokenizer.encode(
            sentence,
            add_special_tokens = True,
            max_length = self.max_len,
            truncation=True
        )
        results = pad_sequences([input_ids], maxlen=self.max_len, dtype="long",
                                    value=0, truncating="post", padding="post")
        
        input_ids = results[0]
        attention_mask = [int(i > 0) for i in input_ids]
        
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)

        # make it a batch of size 1
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids = input_ids,
                token_type_ids = None,
                attention_mask = attention_mask
            )
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0).tolist()
        return sentence_embedding

    def embeddings_to_csv(self):
        print("pulling embeddings...")
        try:
            df = open(self.path_to_data + "\\" + "bert_embeddings.csv","w")
        except FileNotFoundError:
            os.path.mkdir(self.path_to_data)
            df = open(self.path_to_data + "\\" + "bert_embeddings.csv","w")
        for i, auth in enumerate(self.sentences.keys()):
            if i % 2000 == 0 and i > 0:
                print("{} authors processed".format(i))
            auth = int(auth)
            bert_embedding = self.to_embedding(self.sentences[auth])

            df.write(str(auth)+","+",".join(map(str, bert_embedding))+"\n")
        df.close()




class Net(nn.Module):
    """ a simple MLP """

    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 128)
        self.drop = nn.Dropout(0.5)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, X):
   

        x = self.layer1(X).relu()
        x = self.drop(x)
        x = self.layer2(x).relu()
        x = self.drop(x)
        x = self.layer3(x)

        return x






def Transformer_encoder(input_size, d_model, dff):
    input = tf.keras.layers.Input(shape=(input_size,))
    x = Dense(d_model)(input)
    ## self-attention
    query = tf.keras.layers.Dense(d_model)(input)
    value = tf.keras.layers.Dense(d_model)(input)
    key = tf.keras.layers.Dense(d_model)(input)
    attention = tf.keras.layers.Attention()([query, value, key])
    attention = tf.keras.layers.Dense(d_model)(attention)
    x = tf.keras.layers.Add()([x , attention]) # residual connection
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ## Feed Forward
    dense = tf.keras.layers.Dense(dff, activation='relu')(x)
    dense = tf.keras.layers.Dense(d_model)(dense)
    x = tf.keras.layers.Add()([x , dense])      # residual connection
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ######################################################

    x = tf.keras.layers.Dense(1)(x)
    base_model = tf.keras.models.Model(inputs=input, outputs=x)
    return base_model






class Encoder(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Linear(self.input_size, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
      
    def forward(self, A_hat, H):
        x = self.layer1(H)
        x = torch.mm(A_hat, x)
        x = self.relu(x)
        x = self.dropout(x)
        #---------------
        x = self.layer2(x)
        x = torch.mm(A_hat, x)
        #-----------------
        Z = x

        return Z

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Decoder, self).__init__()
    
        self.input_size = input_size #graph_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.layer1 = nn.Linear(self.input_size, self.hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_size) 

    def forward(self, x):
        
        x = self.layer1(x) 
        x = self.dropout(x)
        output = self.output_layer(x)

        return output

class Ensemble(nn.Module):
    
    def __init__(self, en_input_size, en_hidden_dim,
                    de_hidden_dim, de_output_size):
        super(Ensemble, self).__init__()

        self.encoder = Encoder(en_input_size, en_hidden_dim)
        self.decoder = Decoder(en_hidden_dim, de_hidden_dim, de_output_size)
        
    def forward(self, A_hat, X):
        Z = self.encoder(A_hat, X)
        output = self.decoder(Z)
        
        return output
