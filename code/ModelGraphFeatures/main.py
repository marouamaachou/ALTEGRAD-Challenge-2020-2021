import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn

from tqdm import tqdm

path = "\\".join(os.path.abspath(__file__).split("\\")[:-2])
sys.path.insert(0, path)
from ModelGraphFeatures.data import *
from utils import *
from models import *



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set to True train_deepwalk variable 
# if you want to train the deepwalk algorithm
# otherwise it, you will retrieve the embeddings trained

train_deepwalk = False

# Set to true if you want to train X_train

train_model = True

# Set to true if you want to predict X_test

predict = True

####################################################
####################################################
###                                              ###
###                 PREPARE DATA                 ###
###                                              ###
####################################################
####################################################

PATH_TO_DATA = "ModelGraphFeatures\\data"
if not os.path.exists(PATH_TO_DATA):
    os.mkdir(PATH_TO_DATA)

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
                "run paper_representations.py first")

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
X_train = np.concatenate((X_train, embeddings_train), axis=1)
X_val = np.concatenate((X_val, embeddings_val), axis=1)
X_test = np.concatenate((X_test, embeddings_test), axis=1)

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)



##########################
#### save model

def save_model(model, path_to_checkpoints="ModelGraphFeatures\\checkpoints",
                save_file="graph_features_checkpoint.pt"):
    try:
        torch.save(model.state_dict(), path_to_checkpoints + "\\" + save_file)
    except FileNotFoundError:
        os.mkdir(path_to_checkpoints)
        torch.save(model.state_dict(), path_to_checkpoints + "\\" + save_file)    






#############################
## Neural network

# convert to the numpy matrices to tensors

X_train = torch.FloatTensor(X_train).to(device)
#X_val = torch.FloatTensor(X_val).to(device)
y_train = torch.FloatTensor(y_train).to(device)#.unsqueeze(1).to(device)
#y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X_test).to(device)


print("X_train shape :", X_train.size())
print("y_train shape", y_train.size())

#configurations of the neural network
en_input_size = X_train.size(1)
en_hidden_dim = 256



def make_batch(X_train, index_batch):
    batch = X_train[index_batch]
    return batch




net = Net(en_input_size, en_hidden_dim).to(device)

if train_model:

    # Init criterion
    criterion = nn.L1Loss() # mae 

    # Init optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # training settings

    n_epochs = 60
    batch_size = 64
    n = len(X_train)
    n_batch = n // batch_size + 1 * (n % batch_size != 0)
    tqdm_dict = {"loss": 0.0}

    net.train()

    print("training...")
    for epoch in range(n_epochs):
        
        with tqdm(total=n_batch, unit_scale=True, postfix={'loss':0.0},#,'test loss':0.0},
                        desc="Epoch : %i/%i" % (epoch+1, n_epochs), ncols=100) as pbar:

            batch_indexes = torch.randperm(n)
            total_loss = 0.0

            for i in range(0, n, batch_size):
                
                index_batch = batch_indexes[i:(i+batch_size)]
                input_batch = make_batch(X_train, index_batch)
                target_batch = make_batch(y_train, index_batch)
                output_batch = net.forward(input_batch)
                
                loss = criterion(output_batch.flatten(), target_batch)
                total_loss += loss.item()
                tqdm_dict['loss'] = total_loss / (i+1)
                pbar.set_postfix(tqdm_dict)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

    try:
        torch.save(net.state_dict(), "ModelGraphFeatures\\checkpoints\\graph_features_checkpoint.pt")
    except FileNotFoundError:
        os.mkdir("ModelGraphFeatures\\checkpoints")
        torch.save(net.state_dict(), "ModelGraphFeatures\\checkpoints\\graph_features_checkpoint.pt")
    print("model checkpoint saved")

####################################################
####################################################
###                                              ###
###                   RESULTS                    ###
###                                              ###
####################################################
####################################################

if predict:

    #predict y_test 
    print("\nPredicting...")
    net.eval()
    y_pred = net(X_test).squeeze(1).detach().numpy()
    print("\nPrediction finished.")


    # write the predictions to file
    print("\nSaving prediction...")
    df_test = pd.read_csv(test_file, dtype={'authorID': np.int64})
    df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
    df_test.loc[:,["authorID","h_index_pred"]].to_csv(
        PATH_TO_DATA+ '\\' + 'test_predictions_what.csv', index=False
    )
