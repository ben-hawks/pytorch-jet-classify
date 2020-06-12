import torch
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
            softmax_out = self.softmax(self.fc4(x))
            #softmax_out = self.softmax(x)

            return softmax_out

def three_layer_model_seq(Inputs, nclasses, l1Reg=0):
    """
    3 hidden layer model
    """
    # x = Dense(32, activation='relu', kernel_initializer='lecun_uniform',
              # name='fc1_relu', W_regularizer=l1(l1Reg))(Inputs)
    # predictions = Dense(nclasses, activation='sigmoid', kernel_initializer='lecun_uniform',
                        # name = 'output_sigmoid', W_regularizer=l1(l1Reg))(x)
    # model = Model(inputs=Inputs, outputs=predictions)
    print("Getting sequential 3layer model")
    model = nn.Sequential(
        torch.nn.Linear(Inputs, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, nclasses),
        torch.nn.ReLU(),
        torch.nn.Softmax(0)
        #torch.nn.Sigmoid(),
    )
    return model
