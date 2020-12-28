import torch
import pandas as pd
from torch.utils.data import Dataset
import h5pickle as h5py
import io
import os
import numpy as np
from sklearn import preprocessing

class ParticleJetDataset(Dataset):
    """CMS Particle Jet dataset."""

    def __init__(self, dataPath, yamlConfig, normalize=True, filenames=None):
        data_path = dataPath

        # List of features to use
        features = yamlConfig['Inputs']
        self.features_list = features
        # List of labels to use
        labels = yamlConfig['Labels']
        self.labels_list = labels

        columns_arr = np.array([])
        features_labels_df = pd.DataFrame()
        loaded_files = 0
        #Check/Handle directory of files vs 1 file
        if filenames is not None: #Using dataset of .h5 files split into k folds by utilities/k_fold_split.py
            if os.path.isdir(data_path):
                print("Directory of data files found!")
                first = True
                for file in os.listdir(data_path):
                    if (file.endswith(".h5") or file.endswith(".h5df")) and (file in filenames):
                        try:
                            print("Loading " + str(file))
                            self.h5File = h5py.File(os.path.join(data_path,file), 'r', libver='latest', swmr=True)
                            if first:
                                columns_arr = np.array(self.h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
                                first = False
                            this_file = pd.DataFrame(self.h5File["jets"][:], columns=columns_arr)
                            features_labels_df = pd.concat([features_labels_df,this_file],axis=0) #concat along axis 0 if doesn't work?
                            self.h5File.close()
                            loaded_files +=1
                        except Exception as e:
                            print("Error! Failed to load jet file " + file)
                            print(e)
            elif os.path.isfile(data_path):
                print("Single data file found!")
                self.h5File = h5py.File(dataPath, 'r', libver='latest', swmr=True)
                # Convert to dataframe
                columns_arr = np.array(self.h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
                features_labels_df = pd.DataFrame(self.h5File["jets"][:], columns=columns_arr)
            else:
                print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")
                print("Given Path: {}".format(data_path))
        else: #Using a directory full of .h5 files
            if os.path.isdir(data_path):
                print("Directory of data files found!")
                first = True
                for file in os.listdir(data_path):
                    if file.endswith(".h5") or file.endswith(".h5df"):
                        try:
                            print("Loading " + str(file))
                            self.h5File = h5py.File(os.path.join(data_path,file), 'r', libver='latest', swmr=True)
                            if first:
                                columns_arr = np.array(self.h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
                                first = False
                            this_file = pd.DataFrame(self.h5File["jets"][:], columns=columns_arr)
                            features_labels_df = pd.concat([features_labels_df,this_file],axis=0) #concat along axis 0 if doesn't work?
                            self.h5File.close()
                            loaded_files +=1
                        except Exception as e:
                            print("Error! Failed to load jet file " + file)
                            print(e)
            elif os.path.isfile(data_path):
                print("Single data file found!")
                self.h5File = h5py.File(dataPath, 'r', libver='latest', swmr=True)
                # Convert to dataframe
                columns_arr = np.array(self.h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
                features_labels_df = pd.DataFrame(self.h5File["jets"][:], columns=columns_arr)
            else:
                print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")
                print("Given Path: {}".format(data_path))

        print("Loaded {} files successfully!".format(loaded_files))
        features_labels_df = features_labels_df.drop_duplicates()
        features_df = features_labels_df[features]
        labels_df = features_labels_df[labels]
        # Convert to numpy array
        self.features_val = features_df.values.astype(np.float)
        self.labels_val = labels_df.values.astype(np.float)

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
                print("Scaled Features Data W/ StandardScaler")

            # Normalize inputs (w/ MinMax for squared hinge)
            if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' \
                    and yamlConfig['InputType'] != 'Conv2D' \
                    and  yamlConfig['KerasLoss'] == 'squared_hinge':
                scaler = preprocessing.MinMaxScaler().fit(self.features_val)
                self.features_val = scaler.transform(self.features_val)
                print("Scaled Features Data W/ MinMaxScaler")

            # Normalize conv inputs
            if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] == 'Conv1D':
                reshape_X_train_val = self.features_val.reshape(self.features_val.shape[0] * self.features_val.shape[1],
                                                          self.features_val.shape[2])
                scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
                for p in range(self.features_val.shape[1]):
                    self.features_val[:, p, :] = scaler.transform(self.features_val[:, p, :])
                print("Reshaped data for conv and Scaled Features Data W/ StandardScaler")

    def __getitem__(self, index):
        return self.features_val[index], self.labels_val[index]

    def __len__(self):
        return len(self.features_val)

    def close(self):
        self.h5File.close()