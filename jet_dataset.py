import torch
import pandas as pd
from torch.utils.data import Dataset
import h5py
import io
import os
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import auc, roc_curve, accuracy_score

class ParticleJetDataset(Dataset):
    """CMS Particle Jet dataset."""

    def __init__(self, options, yamlConfig, normalize=True):

        # To use one data file:
        self.h5File = h5py.File(options.inputFile, 'r', libver='latest', swmr=True)


        # List of features to use
        features = yamlConfig['Inputs']
        self.features_list = features
        #print(features)
        # List of labels to use
        labels = yamlConfig['Labels']
        self.labels_list = labels
       # print(labels)
        # Convert to dataframe
        columns_arr = np.array(self.h5File['jetFeatureNames'][:]).astype(str) #slicing h5 data because otherwise it's a reference to the actual file?
        features_labels_df = pd.DataFrame(self.h5File["jets"][:], columns=columns_arr)
        #print(features_labels_df.columns) #H5 File doesn't have column names, ugh. this has caused me a bunch of headaches
        #features_labels_df = features_labels_df[features]
        features_labels_df = features_labels_df.drop_duplicates()

        features_df = features_labels_df[features]
        labels_df = features_labels_df[labels]
        # Convert to numpy array
        self.features_val = features_df.values
        self.labels_val = labels_df.values

        if 'j_index' in features:
            self.features_val = self.features_val[:, :-1]  # drop the j_index feature
        if 'j_index' in labels:
            self.labels_val = self.labels_val[:, :-1]  # drop the j_index label
            labels = labels[:-1]

        if normalize:
            # Normalize inputs
            if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' \
                    and yamlConfig['InputType'] != 'Conv2D':
                scaler = preprocessing.StandardScaler().fit(self.features_val)
                self.features_val = scaler.transform(self.features_val)

            # Normalize inputs (w/ MinMax for squared hinge)
            if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' \
                    and yamlConfig['InputType'] != 'Conv2D' \
                    and  yamlConfig['KerasLoss'] == 'squared_hinge':
                scaler = preprocessing.MinMaxScaler().fit(self.features_val)
                self.features_val = scaler.transform(self.features_val)

            # Normalize conv inputs
            if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] == 'Conv1D':
                reshape_X_train_val = self.features_val.reshape(self.features_val.shape[0] * self.features_val.shape[1],
                                                          self.features_val.shape[2])
                scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
                for p in range(self.features_val.shape[1]):
                    self.features_val[:, p, :] = scaler.transform(self.features_val[:, p, :])

    def __getitem__(self, index):
        return self.features_val[index], self.labels_val[index]

    def __len__(self):
        return len(self.features_val)

    def close(self):
        self.h5File.close()