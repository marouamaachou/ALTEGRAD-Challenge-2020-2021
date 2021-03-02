import torch
import torch.nn as nn




class Net(nn.Module):

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
