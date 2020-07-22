import torch
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from brevitas.core.quant import QuantType


class three_layer_model_masked(nn.Module):
        def __init__(self,masks):
            #Model with <16,64,32,32,5> Behavior
            self.m1 = masks['fc1']
            self.m2 = masks['fc2']
            self.m3 = masks['fc3']
            super(three_layer_model_masked,self).__init__()
            self.quantized_model = False
            self.input_shape = 16 #(16,)
            self.fc1 = nn.Linear(self.input_shape,64)
            self.fc2 = nn.Linear(64,32)
            self.fc3 = nn.Linear(32,32)
            self.fc4 = nn.Linear(32,5)
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            self.softmax = nn.Softmax(0)

        def update_masks(self, masks):
            self.m1 = masks['fc1']
            self.m2 = masks['fc2']
            self.m3 = masks['fc3']

        def forward(self, x):
            x = self.act1(self.fc1(x))
            self.fc1.weight.data.mul_(self.m1)
            x = self.act2(self.fc2(x))
            self.fc2.weight.data.mul_(self.m2)
            x = self.act3(self.fc3(x))
            self.fc3.weight.data.mul_(self.m3)
            softmax_out = self.softmax(self.fc4(x))

            return softmax_out


class three_layer_model(nn.Module):
    def __init__(self):
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        softmax_out = self.softmax(self.fc4(x))
        return softmax_out

class three_layer_model_bv(nn.Module):
    def __init__(self):
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_bv, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
        self.weight_precision = 8
        self.fc1 = qnn.QuantLinear(self.input_shape, int(64),
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(64, 32,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(32, 32,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(32, 5,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6) #TODO Check/Change this away from 6, do we have to set a max value here? Can we not?
        self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(test)
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        softmax_out = self.softmax(self.fc4(x))

        return softmax_out



class three_layer_model_bv_masked(nn.Module):
    def __init__(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_bv_masked, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
        self.weight_precision = 8
        self.fc1 = qnn.QuantLinear(self.input_shape, int(64),
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(64, 32,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(32, 32,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(32, 5,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6) #TODO Check/Change this away from 6, do we have to set a max value here? Can we not?
        self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.softmax = nn.Softmax(0)

    def update_masks(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(test)
        self.fc1.weight.data.mul_(self.m1)
        x = self.act2(self.fc2(x))
        self.fc2.weight.data.mul_(self.m2)
        x = self.act3(self.fc3(x))
        self.fc3.weight.data.mul_(self.m3)
        softmax_out = self.softmax(self.fc4(x))

        return softmax_out
