# pytorch-jet-classify

Requirements:
Python 3.7

`pip install numpy pandas torch torchsummaryX matplotlib sklearn h5py seaborn`

`pip install git+https://github.com/Xilinx/brevitas.git`

To Run training: 

`python3 train.py -i <input H5 File/Directory of Files to train on> -o <output directory> -t <H5 File/Directory of files to use as test set> -e <num epoch to train for>`

Upon training completion, graphs for the ROC AUC vs Epoch, Loss vs Epoch, Precision vs Epoch, ROC for each tagger, and Confusion Matrix are saved to the output directory, along with a .pt saved model file. 

The Float/Unquantized 3 Layer model is `models.three_layer_model()`, with the Quantized/Brevitas 3 Layer Model is `models.three_layer_model_bv()`. Either can be chosen to train by setting `current_model` to one of the two. 

At the moment, the precision of `models.three_layer_model_bv()` is set by `self.weight_precision` within the class in `models.py`, though this is likely to change in the future
