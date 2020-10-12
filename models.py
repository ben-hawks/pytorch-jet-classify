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


class three_layer_model_batnorm_masked(nn.Module):
    def __init__(self, masks):
        # Model with <16,64,32,32,5> Behavior
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        super(three_layer_model_batnorm_masked, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.softmax = nn.Softmax(0)

    def update_masks(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']

    def mask_to_device(self, device):
        self.m1 = self.m1.to(device)
        self.m2 = self.m2.to(device)
        self.m3 = self.m3.to(device)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(self.bn1(test))
        self.fc1.weight.data.mul_(self.m1)
        x = self.act2(self.bn2(self.fc2(x)))
        self.fc2.weight.data.mul_(self.m2)
        x = self.act3(self.bn3(self.fc3(x)))
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

    def mask_to_device(self, device):
        self.m1 = self.m1.to(device)
        self.m2 = self.m2.to(device)
        self.m3 = self.m3.to(device)

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


class three_layer_model_bv_batnorm_masked(nn.Module):
    def __init__(self, masks, precision = 8):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        self.weight_precision = precision
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_bv_batnorm_masked, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
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
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.softmax = nn.Softmax(0)

    def update_masks(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']

    def mask_to_device(self, device):
        self.m1 = self.m1.to(device)
        self.m2 = self.m2.to(device)
        self.m3 = self.m3.to(device)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(self.bn1(test))
        self.fc1.weight.data.mul_(self.m1)
        x = self.act2(self.bn2(self.fc2(x)))
        self.fc2.weight.data.mul_(self.m2)
        x = self.act3(self.bn3(self.fc3(x)))
        self.fc3.weight.data.mul_(self.m3)
        softmax_out = self.softmax(self.fc4(x))

        return softmax_out


class three_layer_model_bv_tunable(nn.Module):
    def __init__(self, masks, dims = [64,32,32], precision = 8):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        self.dims = dims
        self.weight_precision = precision
        # Model with variable behavior
        super(three_layer_model_bv_tunable, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
        self.fc1 = qnn.QuantLinear(self.input_shape, self.dims[0],
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(self.dims[0], self.dims[1],
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(self.dims[1], self.dims[2],
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(self.dims[2], 5,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6) #TODO Check/Change this away from 6, do we have to set a max value here? Can we not?
        self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.bn1 = nn.BatchNorm1d(self.dims[0])
        self.bn2 = nn.BatchNorm1d(self.dims[1])
        self.bn3 = nn.BatchNorm1d(self.dims[2])
        self.softmax = nn.Softmax(0)

    def update_masks(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']

    def mask_to_device(self, device):
        self.m1 = self.m1.to(device)
        self.m2 = self.m2.to(device)
        self.m3 = self.m3.to(device)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(self.bn1(test))
        self.fc1.weight.data.mul_(self.m1)
        x = self.act2(self.bn2(self.fc2(x)))
        self.fc2.weight.data.mul_(self.m2)
        x = self.act3(self.bn3(self.fc3(x)))
        self.fc3.weight.data.mul_(self.m3)
        softmax_out = self.softmax(self.fc4(x))

        return softmax_out

class three_layer_model_bv_masked_quad(nn.Module):
    def __init__(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        # Model with <16,64,32,32,5> x 1/4 Behavior
        super(three_layer_model_bv_masked_quad, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
        self.weight_precision = 8
        self.fc1 = qnn.QuantLinear(self.input_shape, int(256),
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(256, 128,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(128, 128,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(128, 5,
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

    def mask_to_device(self, device):
        self.m1 = self.m1.to(device)
        self.m2 = self.m2.to(device)
        self.m3 = self.m3.to(device)

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


class three_layer_model_bv_masked_quarter(nn.Module):
    def __init__(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']
        # Model with <16,64,32,32,5> x 1/4 Behavior
        super(three_layer_model_bv_masked_quarter, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True  # variable to inform some of our plotting functions this is quantized
        self.weight_precision = 8
        self.fc1 = qnn.QuantLinear(self.input_shape, int(16),
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(16, 8,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(8, 8,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(8, 5,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision,
                                  max_val=6)  # TODO Check/Change this away from 6, do we have to set a max value here? Can we not?
        self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.softmax = nn.Softmax(0)

    def update_masks(self, masks):
        self.m1 = masks['fc1']
        self.m2 = masks['fc2']
        self.m3 = masks['fc3']

    def mask_to_device(self, device):
        self.m1 = self.m1.to(device)
        self.m2 = self.m2.to(device)
        self.m3 = self.m3.to(device)

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


class t2_autoencoder_masked(nn.Module):
    def __init__(self,masks):

        self.e1 = masks['enc1']
        self.e2 = masks['enc2']
        self.e3 = masks['enc3']
        self.e4 = masks['enc4']

        self.d1 = masks['dec1']
        self.d2 = masks['dec2']
        self.d3 = masks['dec3']
        self.d4 = masks['dec4']

        super(t2_autoencoder_masked, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(128, 128)
        self.ebn1 = nn.BatchNorm1d(128)
        self.eact1 = nn.ReLU(True)
        self.enc2= nn.Linear(128, 128)
        self.ebn2 = nn.BatchNorm1d(128)
        self.eact2 = nn.ReLU(True)
        self.enc3 = nn.Linear(128, 128)
        self.ebn3 = nn.BatchNorm1d(128)
        self.eact3 = nn.ReLU(True)
        self.enc4 = nn.Linear(128, 8)
        self.ebn4 = nn.BatchNorm1d(128)
        self.eact4 = nn.ReLU(True)

        # Decoder
        self.dec1 = nn.Linear(8, 128)
        self.dbn1 = nn.BatchNorm1d(128)
        self.dact1 = nn.ReLU(True)
        self.dec2= nn.Linear(128, 128)
        self.dbn2 = nn.BatchNorm1d(128)
        self.dact2 = nn.ReLU(True)
        self.dec3 = nn.Linear(128, 128)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dact3 = nn.ReLU(True)
        self.dec4 = nn.Linear(128, 128)
        self.dbn4 = nn.BatchNorm1d(128)
        self.dact4 = nn.ReLU(True)

    def update_masks(self, masks):
        self.e1 = masks['enc1']
        self.e2 = masks['enc2']
        self.e3 = masks['enc3']
        self.e4 = masks['enc4']

        self.d1 = masks['dec1']
        self.d2 = masks['dec2']
        self.d3 = masks['dec3']
        self.d4 = masks['dec4']

    def mask_to_device(self, device):
        self.e1 = self.e1.to(device)
        self.e2 = self.e2.to(device)
        self.e3 = self.e3.to(device)
        self.e4 = self.e4.to(device)

        self.d1 = self.d1.to(device)
        self.d2 = self.d2.to(device)
        self.d3 = self.d3.to(device)
        self.d4 = self.d4.to(device)

    def forward(self, x):

        # Encoder Pass
        x = self.eact1(self.ebn1(self.enc1(x)))
        self.enc1.weight.data.mul_(self.e1)
        x = self.eact2(self.ebn2(self.enc2(x)))
        self.enc2.weight.data.mul_(self.e2)
        x = self.eact3(self.ebn3(self.enc3(x)))
        self.enc3.weight.data.mul_(self.e3)
        x = self.eact4(self.ebn4(self.enc4(x)))
        self.enc4.weight.data.mul_(self.e4)

        # Decoder Pass
        x = self.dact1(self.dbn1(self.dec1(x)))
        self.dec1.weight.data.mul_(self.d1)
        x = self.dact2(self.dbn2(self.dec2(x)))
        self.dec2.weight.data.mul_(self.d2)
        x = self.dact3(self.dbn3(self.dec3(x)))
        self.dec3.weight.data.mul_(self.d3)
        x = self.dact4(self.dbn4(self.dec4(x)))
        self.dec4.weight.data.mul_(self.d4)
        return x