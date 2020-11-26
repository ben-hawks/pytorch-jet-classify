import torch
import pandas as pd
from torch.utils.data import Dataset
import io
import os
import numpy as np
from sklearn import preprocessing
import librosa
import librosa.core
import librosa.feature
import sys

class aeWavDataset(Dataset):
    """TinyML Autoencoder .WAV File dataset"""

    def __init__(self, dataPath):
        data_path = dataPath

        # List of features to use

        self.wav_list = None

        if os.path.isdir(data_path):
            print("Directory of data files found at: {}".format(data_path))
            first = True
            num_files =len([name for name in os.listdir(data_path) if name.endswith(".wav")])
            for idx, file in enumerate(os.listdir(data_path)):
                if file.endswith(".wav"):
                    vector_array = self.file_to_vector_array(os.path.join(data_path,file))
                    if first == True:
                        dataset = np.zeros((vector_array.shape[0] * num_files, 640), float)
                        first = False
                    dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (int(idx) + 1), :] = vector_array
            self.wav_list = dataset
        else:
            print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")

    def file_to_vector_array(self,
                             file_name,
                             n_mels=128,
                             frames=5,
                             n_fft=1024,
                             hop_length=512,
                             power=2.0):
        """
        convert file_name to a vector array.
        file_name : str
            target .wav file
        return : numpy.array( numpy.array( float ) )
            vector array
            * dataset.shape = (dataset_size, feature_vector_length)
        """
        # 01 calculate the number of dimensions
        dims = n_mels * frames

        # 02 generate melspectrogram using librosa
        y, sr = self.file_load(file_name)
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)

        # 03 convert melspectrogram to log mel energy
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

        # 04 calculate total vector size
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

        # 05 skip too short clips
        if vector_array_size < 1:
            return np.empty((0, dims))

        # 06 generate feature vectors by concatenating multiframes
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        return vector_array

    def file_load(self, wav_name, mono=False):
        """
        load .wav file.
        wav_name : str
            target .wav file
        sampling_rate : int
            audio file sampling_rate
        mono : boolean
            When load a multi channels file and this param True, the returned data will be merged for mono data
        return : numpy.array( float )
        """
        try:
            return librosa.load(wav_name, sr=None, mono=mono)
        except Exception as e:
            print("Error! Failed to load wav {}, Reason: {}".format(wav_name, e))

    def __getitem__(self, index):
        #since this is for an autoencoder, there's no "label", but return the same thing twice for convienence
        return self.wav_list[index], self.wav_list[index]

    def __len__(self):
        return len(self.wav_list)

    def close(self):
        self.wav_list.clear()
