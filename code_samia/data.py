import pandas as pd
import numpy as np
import networkx as nx
import time

class Data():

    def __init__(self, train_file, test_file, embeddings_file, graph_file, author_file, val_size):

        # train set
        df = pd.read_csv(train_file, dtype={'authorID': np.int64, 'h_index': np.float32})
        n = df.shape[0]
        train_size = 1 - val_size
        self.df_train = df[:int(train_size*n)]
        self.df_val = df[int(train_size*n):]
        print("Train done")

        
        # test set
        self.df_test = pd.read_csv(test_file, dtype={'authorID': np.int64})
        print("Test done")

        # read embeddings of abstracts
        start = time.time()   
        self.embeddings = pd.read_csv(embeddings_file, header=None)
        self.embeddings = self.embeddings.rename(columns={0: "authorID"})
        end = time.time()

        print("Embeddings done", end - start)

        
        # load author papers info
        self.author_data = pd.read_csv(author_file, delimiter=',' ,dtype={'authorID': np.int64, 'rank': np.float64})
        self.author_data_dict = dict(zip(self.author_data["authorID"], self.author_data["rank"]))


        # load the graph  
        start = time.time()  
        self.G = nx.read_edgelist(graph_file, delimiter=' ', nodetype=int)
        nx.set_node_attributes(self.G, self.author_data_dict, "n_papers")
        self.core_number = nx.core_number(self.G)
        print("B")
        self.b_centrality = nx.betweenness_centrality(self.G, k = 40)
        print("b finish")
        self.centrality = nx.degree_centrality(self.G)
        self.e_centrality = nx.eigenvector_centrality(self.G)
        self.p_rank = nx.pagerank(self.G)
        self.avg_neighbor_degree = nx.average_neighbor_degree(self.G)
        end = time.time()
        print("Graph done", end - start)


    
        

    def __preprocess(self, df):
        
        cols_to_norm = ["rank", "Degrees", "Core_numbers", "Avg_neighbor_degrees", "Mean_number_papers","Centrality", "Eign_centrality", "Pagerank", "B_centrality"]
        df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        return df

    def structural_features(self, df):


        degrees = []
        core_numbers = []
        avg_neighbor_degrees = []
        mean_papers = []
        centralities = []
        b_centralities = []
        #c_centralities = []
        e_centralities = []
        p_ranks = []
        #second_centralities = []
        #std_papers = []

        for i,row in df.iterrows():
            node = int(row['authorID'])
            degrees.append(self.G.degree(node))
            core_numbers.append(self.core_number[node])
            avg_neighbor_degrees.append(self.avg_neighbor_degree[node])
            centralities.append(self.centrality[node])
            b_centralities.append(self.b_centrality[node])
            #c_centralities.append(self.c_centrality[node])
            e_centralities.append(self.e_centrality[node])
            p_ranks.append(self.p_rank[node])
            #second_centralities.append(self.second_centrality[node])
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                mean = 0
                for n in neighbors:
                    mean += self.G.nodes[n]['n_papers']
                mean_papers.append(mean/len(neighbors))
            else:
                mean_papers.append(0.0)
 
        df["Degrees"] = degrees
        df["Core_numbers"] = core_numbers
        df["Avg_neighbor_degrees"] = avg_neighbor_degrees
        df["Mean_number_papers"] = mean_papers
        df["Centrality"] = centralities
        df["B_centrality"] = b_centralities
        #df["C_centrality"] = c_centralities
        #df["S_centrality"] = second_centralities
        df["Eign_centrality"] = e_centralities
        df["Pagerank"] = p_ranks


        
        print("Checking....")
        print(df["Mean_number_papers"].isnull().sum())
  
        return df

 
    def X_y_train(self):
        
        self.df_train = self.df_train.merge(self.embeddings, on="authorID")
        self.df_train = self.df_train.merge(self.author_data, on="authorID")
        self.df_train = self.structural_features(self.df_train)
        self.df_train = self.__preprocess(self.df_train)


        X_train = self.df_train.iloc[:,2:].values
        y_train = self.df_train.iloc[:,1].values


        return X_train, y_train

    def X_y_val(self):
        
        self.df_val = self.df_val.merge(self.embeddings, on="authorID")
        self.df_val = self.df_val.merge(self.author_data, on="authorID")
        self.df_val = self.structural_features(self.df_val)
        self.df_val = self.__preprocess(self.df_val)

        X_val = self.df_val.iloc[:,2:].values
        y_val = self.df_val.iloc[:,1].values


        return X_val, y_val


    def X_y_test(self):
        
        self.df_test = self.df_test.merge(self.embeddings, on="authorID")
        self.df_test = self.df_test.merge(self.author_data, on="authorID")
        self.df_test = self.structural_features(self.df_test)
        self.df_test = self.__preprocess(self.df_test)


        X_test = self.df_test.iloc[:,2:].values
        y_test = self.df_test.iloc[:,1].values


        return X_test, y_test
 

    def get_graph(self):

        return self.G
    
    def get_train_nodes(self):

        return self.df_train["authorID"].values.tolist()

    def get_val_nodes(self):

        return self.df_val["authorID"].values.tolist()

    def get_test_nodes(self):

        return self.df_test["authorID"].values.tolist()