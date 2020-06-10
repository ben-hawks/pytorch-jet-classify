# pytorch-jet-classify

Requirements:
Python 3.7

`pip install numpy pandas torch torchsummaryX matplotlib sklearn h5py seaborn`

To Run training: 

`python3 train.py -i <input H5 File> -o <output directory> -e <num epoch to train for>`

Note that currently nothing is saved to the output directory, and there is no early stopping, 
so it will train for however many epochs is specified. 
