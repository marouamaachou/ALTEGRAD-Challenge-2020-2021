import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf 
from tensorflow.keras.layers import Attention, Dense, Dropout,Attention, Concatenate
from tensorflow.keras import Sequential

from tqdm import tqdm

path = "\\".join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.insert(0, path)
from ModelTransformer.data import *
from models import Transformer_encoder
from utils import *


# Set to True train_deepwalk variable 
# if you want to train the deepwalk algorithm
# otherwise it, you will retrieve the embeddings trained

train_deepwalk = False

# Set to true if you want to train X_train

train_model = True

# Set to true if you want to predict X_test

predict = True

TO_RUN_FROM = "code"
if not check_running_file(TO_RUN_FROM):
    raise OSError("the file should run from \"{}/\" folder".format(TO_RUN_FROM))

PATH_TO_DATA = "ModelGraphFeatures\\data"
if not os.path.exists(PATH_TO_DATA):
    tokens = PATH_TO_DATA.split("\\")
    for i in range(len(tokens)):
        os.mkdir("\\".join(tokens[:(i+1)]))


####################################################
####################################################
###                                              ###
###                 PREPARE DATA                 ###
###                                              ###
####################################################
####################################################

train_file = 'train.csv'
test_file = 'test.csv'
embeddings_file = 'author_embeddings.csv'
graph_file = 'collaboration_network.edgelist'
author_file = 'author_ranks.csv'


# Init data to get the dataset
print("\nReading the data...")
try:
    data = Data(train_file, test_file, embeddings_file, graph_file, author_file, val_size=0.3)
except FileNotFoundError:
    print("the current file requires \"author_embeddings.csv\" to work ; "
                "run author_representations.py first")

#retrive the graph
G = data.get_graph()


#Retrieve  train nodes, validation nodes, test nodes
train_nodes = data.get_train_nodes()
val_nodes = data.get_val_nodes()
test_nodes = data.get_test_nodes()


# Train set, val set and test set
X_train, y_train = data.X_y_train()
X_val, y_val = data.X_y_val()
X_test, _ = data.X_y_test()

print("\nDataset created.")




####################################################
####################################################
###                                              ###
###                 MODEL                        ###
###                                              ###
####################################################
####################################################

#model configurations
if not all([
    os.path.exists(PATH_TO_DATA + '\\' + 'deep_walk_emb_train.txt'),
    os.path.exists(PATH_TO_DATA + '\\' + 'deep_walk_emb_val.txt'),
    os.path.exists(PATH_TO_DATA + '\\' + 'deep_walk_emb_test.txt')
]):
    train_deepwalk = True

if train_deepwalk:
    n_dim = 128
    n_walks = 30
    walk_length = 45

    model = deepwalk(G, n_walks, walk_length, n_dim)

    embeddings_train = np.zeros((len(train_nodes), n_dim))
    for i, node in enumerate(train_nodes):
        embeddings_train[i,:] = model.wv[str(node)]

    embeddings_val = np.zeros((len(val_nodes), n_dim))
    for i, node in enumerate(val_nodes):
        embeddings_val[i,:] = model.wv[str(node)]

    embeddings_test = np.zeros((len(test_nodes), n_dim))
    for i, node in enumerate(test_nodes):
        embeddings_test[i,:] = model.wv[str(node)]


    #save the embeddings
    np.savetxt(PATH_TO_DATA + '\\' + 'deep_walk_emb_train.txt', embeddings_train,fmt='%.5f')
    np.savetxt(PATH_TO_DATA + '\\' + 'deep_walk_emb_val.txt', embeddings_val,fmt='%.5f')
    np.savetxt(PATH_TO_DATA + '\\' + 'deep_walk_emb_test.txt', embeddings_test,fmt='%.5f')

else:
    print("load node embeddings...")
    embeddings_train = np.loadtxt(PATH_TO_DATA + '\\' + 'deep_walk_emb_train.txt')
    embeddings_val = np.loadtxt(PATH_TO_DATA + '\\' + 'deep_walk_emb_val.txt')
    embeddings_test = np.loadtxt(PATH_TO_DATA + '\\' + 'deep_walk_emb_test.txt')




#Concatenate the nodes embeddings with the dataset created above
x_train = np.concatenate((X_train, embeddings_train), axis=1)
x_val = np.concatenate((X_val, embeddings_val), axis=1)
x_test = np.concatenate((X_test, embeddings_test), axis=1)

X_train = np.concatenate((x_train, x_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)



# Hyperparameters
d_model = 64
dff=128

# Size of input vocab plus start and end tokens
input_size = X_train.shape[1]

## Run model 
base_model = Transformer_encoder(input_size, d_model, dff)
base_model.summary() 
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.1, patience=5)
base_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
base_model.fit(X_train, y_train, epochs = 50, batch_size=32, callbacks=[callback])

if predict:

    #base_model= tf.keras.models.load_model('/Users/marouamaachou/ALTEGRAD-Challenge-2020-2021/code_samia/transformer')
    #predict y_test 
    print("\nPredicting...")
    y_pred = np.squeeze(base_model.predict(x_test),1)
    print("\nPrediction finished.")


    # write the predictions to file
    print("\nSaving prediction...")
    df_test = pd.read_csv(test_file, dtype={'authorID': np.int64})
    df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
    df_test.loc[:,["authorID","h_index_pred"]].to_csv(
        PATH_TO_DATA + "\\" + "test_predictions_transformer.csv", index=False
    )