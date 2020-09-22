import torch
import pandas as pd
from torch.utils.data import Dataset
import h5py
import io
import os
import numpy as np
from sklearn import preprocessing
import torchaudio


class aeWavDataset(Dataset):
    """TinyML Autoencoder .WAV File dataset"""

    def __init__(self, dataPath):
        data_path = dataPath

        # List of features to use

        self.wav_list = []

        if os.path.isdir(data_path):
            print("Directory of data files found!")
            first = True
            for file in os.listdir(data_path):
                if file.endswith(".wav"):
                    try:
                        audio = torchaudio.load_wav(file)
                        self.wav_list.append(audio)
                    except:
                        print("Error! Failed to load wav " + file)
        elif os.path.isfile(data_path):
            print("Single data file found!")
            if data_path.endswith(".wav"):
                try:
                    audio = torchaudio.load_wav(data_path)
                    self.wav_list.append(audio)
                except:
                    print("Error! Failed to load wav " + data_path)
        else:
            print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")


    def __getitem__(self, index):
        return self.wav_list[index], self.wav_list[index]

    def __len__(self):
        return len(self.wav_list)

    def close(self):
        self.wav_list.clear()
