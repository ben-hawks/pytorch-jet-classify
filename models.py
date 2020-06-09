import torch.nn as nn
import numpy as np



class three_layer_model(nn.Module):
        def __init__(self):
            #Model with <16,64,32,32,5> Behavior
            super(three_layer_model,self).__init__()
            self.input_shape = 16 #(16,)
            self.fc1 = nn.Linear(self.input_shape,64)
            self.fc2 = nn.Linear(64,32)
            self.fc3 = nn.Linear(32,32)
            self.fc4 = nn.Linear(32,5)
            self.act = nn.ReLU()
            self.softmax = nn.Softmax(0)

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = self.act(self.fc4(x))
            softmax_out = self.softmax(x)

            return softmax_out

