import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

# read training data
df_train = pd.read_csv('train.csv', dtype={'authorID': np.int64, 'h_index': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv('test.csv', dtype={'authorID': np.int64})
n_test = df_test.shape[0]

# read embeddings of abstracts   
embeddings = pd.read_csv("author_embeddings.csv", header=None)
embeddings = embeddings.rename(columns={0: "authorID"})

# create the training matrix. each author is represented by the average of
# the embeddings of the abstracts of his/her top-cited papers
df_train = df_train.merge(embeddings, on="authorID")
X_train = df_train.iloc[:,2:].values
y_train = df_train.iloc[:,1].values

# create the test matrix. each author is represented by the average of
# the embeddings of the abstracts of his/her top-cited papers
df_test = df_test.merge(embeddings, on="authorID")
X_test = df_test.iloc[:,2:].values

# train a regression model and make predictions
reg = Lasso(alpha=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# write the predictions to file
df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
df_test.loc[:,["authorID","h_index_pred"]].to_csv('test_predictions.csv', index=False)
