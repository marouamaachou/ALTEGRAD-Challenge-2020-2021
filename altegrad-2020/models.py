import os
import csv
import plsa
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl

from tqdm import tqdm



class AuthorConvNet(nn.Module):
    """ Class to implement the network architecture as proposed in
        "Non Linear Text Regression With Deep Convolutional Neural Network",
        Bitvai and Cohn (https://www.aclweb.org/anthology/P15-2030.pdf)
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim=300,
        dropout_rate=0.3,
        max_conv_size=3,
        criterion=nn.L1Loss(),
        optimizer = optim.Adadelta,
        path_to_checkpoints="ModelAuthorConvolution\\checkpoints"
    ):
        """ This architecture performs 2D convolutions using 1D kernels with size 1,...,max_conv_size """
        super(AuthorConvNet, self).__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.dropout = nn.Dropout(dropout_rate)
        self.max_conv_size = max_conv_size
        self.max_pool = lambda x: torch.max(x, dim=len(x.size())-2)[0]    # max_pool here is the max over the input raws

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # max_conv_size 1D convolutional filters of size 1,...,max_conv_size
        self.convs = [nn.Conv2d(1,1,(i,1)) for i in range(1, self.max_conv_size+1)]
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(embedding_dim*max_conv_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.path_to_checkpoints=path_to_checkpoints

    def forward(self, x):
        z = x.unsqueeze(1)   # second dimension corresponds to in_channels, 
                             # so it should be 1 (1st is batch  size)
        C = []
        for convolution in self.convs:          # apply convolution for each kernel size between 1 and q
            conv = convolution(z).squeeze()
            C.append(conv)
        H = [self.relu(conv) for conv in C]
        p = [self.max_pool(h) for h in H]
        p = torch.cat(p, dim=len(p[0].size())-1)
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

    def save_model(self, path=None, file_name="author_conv_checkpoint.pt"):
        if path:
            torch.save(self.state_dict(), path + "\\" + file_name)
        else:
            torch.save(self.state_dict(), self.path_to_checkpoints + "\\" + file_name)

    def load_model(self, path=None, file_name="author_conv_checkpoint.pt"):
        if path:
            self.load_state_dict(torch.load(path + "\\" + file_name))
        else:
            self.load_state_dict(torch.load(self.path_to_checkpoints + "\\" + file_name))
            
    def fit(
        self,
        data,           # an AuthorConvData instance
        n_epochs=10,
        batch_size=64,
        save_checkpoint_every=5,
        save_file="author_conv_checkpoint.pt"
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
                    input_batch = data.make_input_batch(index_batch)
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
                self.save_model()
                print("model checkpoint saved")
        self.save_model(file_name=save_file)

    def predict(self, author_matrix):
        """ predict for a single author embedding matrix """

        with torch.no_grad():
            mat = author_matrix.unsqueeze(dim=0)    # make it a batch of size 1
            prediction = self.forward(mat)
        return prediction
    
    def existing_model(self):
        """ return True if a checkpoint exists in path_to_checkpoint, False otherwise """

        abspath = "\\".join(os.path.abspath(__file__).split("\\")[:-1])
        if os.listdir(abspath + "\\" + self.path_to_checkpoints): return True
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
        print("computing PLSA for the corpus...")
        plsa_ = plsa.PLSA(corpus=corpus_, n_topics=n_topics, tf_idf=True)
        print("done.")
        topic_embeddings = plsa_.topic_given_doc.T
        self.topic_embeddings = topic_embeddings
        self.save_embeddings(topic_embeddings)
        return topic_embeddings

    def save_embeddings(self, embeddings, output_file="objects\\topic_embeddings.pkl"):
        """ store the array of "topic given docs" distributions in a pickle .pkl file """

        with open(output_file, "wb") as f:
            pkl.dump(embeddings, f)