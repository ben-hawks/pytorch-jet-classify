import os, queue, subprocess, time, argparse, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import tensorflow as tf
#import tensorflow.keras as keras
#import TensorState
#import LeNet
import numpy as np
#from tqdm import tqdm
from pathlib import Path

def sort_microstates(unsorted_microstates,is_bool=False):
    if not is_bool:
        microstates = unsorted_microstates>0
    else:
        microstates = unsorted_microstates
    sorted_index = np.lexsort(tuple(column for column in microstates.T))
    microstates = microstates[sorted_index].squeeze()
    index = np.argwhere(np.any(np.logical_xor(microstates[0:-1,:],microstates[1:,:]),axis=-1)).squeeze()
    if index.size>1:
        index = [i+1 for i in index]
    elif index.size==1:
        index = [index+1]
    else:
        index = []
    index.append(len(sorted_index))
    return microstates,index

def accumulate_ensemble(ensemble=None,microstates=None,index=None):
    if index==None:
        for state in microstates:
            microstate = ''.join(['1' if s>=0 else '0' for s in state])
            if microstate in ensemble.keys():
                ensemble[microstate] += 1
            else:
                ensemble[microstate] = 1
    else:
        count = 0
        for ind in index:
            microstate = ''.join(['1' if s else '0' for s in microstates[count,:]])
            if microstate in ensemble.keys():
                ensemble[microstate] += ind - count
            else:
                ensemble[microstate] = ind - count
            count = ind
                
    return ensemble

def layer_efficiency(ensemble):
    num_microstates = sum(v for v in ensemble.values())
    frequencies = {}
    for microstate,count in ensemble.items():
        frequencies[microstate] = count/num_microstates
    entropy = sum(-p * np.log2(p) for p in frequencies.values())
    max_entropy = len(list(ensemble)[0])
    efficiency = entropy/max_entropy
    return efficiency,entropy,max_entropy

def network_efficiency(efficiencies):
    net_efficiency = np.exp(sum(np.log(eff) for eff in efficiencies)/len(efficiencies)) # geometric mean
    return net_efficiency

def aIQ(net_efficiency,accuracy,weight):
    if weight <= 0:
        raise ValueError('aIQ weight must be greater than 0.')
    aIQ = np.power(accuracy**weight * net_efficiency,1/(weight+1))
    return aIQ
