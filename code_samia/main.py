import numpy as np
import pandas as pd
import networkx as nx
from data import *
from utils import *
from models import *
import torch
import torch.nn as nn



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set to True train_deepwalk variable 
# if you want to train the deepwalk algorithm
# otherwise it, you will retrieve the embeddings trained

train_deepwalk = False

# Set to true if you want to predict X_test

predict = False

####################################################
####################################################
###                                              ###
###                 PREPARE DATA                 ###
###                                              ###
####################################################
####################################################

train_file = '../data/train.csv'
test_file = '../data/test.csv'
embeddings_file = '../data/author_embedding.csv'
graph_file = '../data/collaboration_network.edgelist'
author_file = '../data/author_ranks.csv'


# Init data to get the dataset
print("\nReading the data...")
data = Data(train_file, test_file, embeddings_file, graph_file, author_file, val_size=0.3)

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
    np.savetxt('../data/deep_walk_emb_train.txt', embeddings_train,fmt='%.5f')
    np.savetxt('../data/deep_walk_emb_val.txt', embeddings_val,fmt='%.5f')
    np.savetxt('../data/deep_walk_emb_test.txt', embeddings_test,fmt='%.5f')

else:

    embeddings_train = np.loadtxt('../data/deep_walk_emb_train.txt')
    embeddings_val = np.loadtxt('../data/deep_walk_emb_val.txt')
    embeddings_test = np.loadtxt('../data/deep_walk_emb_test.txt')




#Concatenate the nodes embeddings with the dataset created above
x_train = np.concatenate((X_train, embeddings_train), axis=1)
x_val = np.concatenate((X_val, embeddings_val), axis=1)
x_test = np.concatenate((X_test, embeddings_test), axis=1)







#############################
## Neural network

# convert to the numpy matrices to tensors

x_train = torch.FloatTensor(x_train).to(device)
x_val = torch.FloatTensor(x_val).to(device)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
x_test = torch.FloatTensor(x_test).to(device)


#configurations of the neural network
en_input_size = x_train.size(1)
en_hidden_dim = 256

net = Net(en_input_size, en_hidden_dim).to(device)


# Init criterion
criterion = nn.L1Loss() # mae 

# Init optimizer 
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# Number epochs

n_epochs = 250


for epoch in range(n_epochs + 1):
    net.train()
    optimizer.zero_grad()
    output = net(x_train) 
    loss = criterion(output, y_train) 
    loss.backward()
    optimizer.step()

    net.eval()
    output_val = net(x_val)
    loss_val =  nn.L1Loss()(output_val, y_val)
    if epoch%10 == 0:
        print('Epoch: {:01d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss.item()),
          'loss_val: {:.4f}'.format(loss_val.item()))



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
    y_pred = net(x_test).squeeze(1).detach().numpy()
    print("\nPrediction finished.")


    # write the predictions to file
    print("\nSaving prediction...")
    df_test = pd.read_csv(test_file, dtype={'authorID': np.int64})
    df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
    df_test.loc[:,["authorID","h_index_pred"]].to_csv('../data/test_predictions_what.csv', index=False)
