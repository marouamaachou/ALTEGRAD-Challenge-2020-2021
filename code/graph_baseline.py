import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import Lasso

# read training data
df_train = pd.read_csv('train.csv', dtype={'authorID': np.int64, 'h_index': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv('test.csv', dtype={'authorID': np.int64})
n_test = df_test.shape[0]

# load the graph    
G = nx.read_edgelist('collaboration_network.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# computes structural features for each node
core_number = nx.core_number(G)
avg_neighbor_degree = nx.average_neighbor_degree(G)

# create the training matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number and (3) the average degree of its neighbors
X_train = np.zeros((n_train, 3))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = row['authorID']
    X_train[i,0] = G.degree(node)
    X_train[i,1] = core_number[node]
    X_train[i,2] = avg_neighbor_degree[node]
    y_train[i] = row['h_index']

# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number and (3) the average degree of its neighbors
X_test = np.zeros((n_test, 3))
for i,row in df_test.iterrows():
    node = row['authorID']
    X_test[i,0] = G.degree(node)
    X_test[i,1] = core_number[node]
    X_test[i,2] = avg_neighbor_degree[node]
    
# train a regression model and make predictions
reg = Lasso(alpha=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# write the predictions to file
df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
df_test.loc[:,["authorID","h_index_pred"]].to_csv('test_predictions.csv', index=False)
