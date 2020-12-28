import torch
import pandas as pd
import h5py
import io
import tables
import random
import os
import numpy as np
import yaml
import sklearn.utils as sklu
from optparse import OptionParser

def parse_config(config_file) :
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.FullLoader)


if __name__ == "__main__":
        parser = OptionParser()
        parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='',
                          help='location of data to train off of')
        parser.add_option('-o', '--output', action='store', type='string', dest='outputDir', default='train_simple/',
                          help='output directory')
        parser.add_option('-c', '--config', action='store', type='string', dest='config',
                          default='configs/train_config_threelayer.yml', help='tree name')
        parser.add_option('-k', '--kfolds', action='store', type='int', dest='kfolds',
                          default='4', help='How many folds to split the dataset into')
        parser.add_option('-s', '--no_shuffle', action='store_false', dest='shuffle', default=True,
                          help='disable shuffling of the dataset before splitting')

        (options, args) = parser.parse_args()
        yamlConfig = parse_config(options.config)

        if not os.path.exists(options.outputDir):  # create given output directory if it doesnt exist
            os.makedirs(options.outputDir)

        data_path = options.inputFile

        # List of features to use
        features = yamlConfig['Inputs']
        features_list = features
        # List of labels to use
        labels = yamlConfig['Labels']
        labels_list = labels

        columns_arr = np.array([])
        features_labels_df = pd.DataFrame()

        all_data = None
        feature_names = None

        if os.path.isdir(data_path):
            print("Directory of data files found!")
            first = True
            for file in os.listdir(data_path):
                if file.endswith(".h5") or file.endswith(".h5df"):
                    try:
                        print("Loading " + str(file))
                        h5File = h5py.File(os.path.join(data_path,file), 'r', libver='latest', swmr=True)
                        if first:
                            columns_arr = np.array(h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
                            first = False
                            feature_names = np.array(h5File['jetFeatureNames'][:])
                        this_file = pd.DataFrame(h5File["jets"][:], columns=columns_arr)
                        features_labels_df = pd.concat([features_labels_df,this_file],axis=0) #concat along axis 0 if doesn't work?
                        h5File.close()
                    except Exception as e:
                        print("Error! Failed to load jet file " + file)
                        print(e)

            if options.shuffle:
                #Shuffle dataset before split, using seed specified in yamlConfig for random shuffle
                features_labels_df = sklu.shuffle(features_labels_df, random_state=yamlConfig["Seed"])
                features_labels_df.reset_index(inplace=True,drop=True)

            all_data = features_labels_df.to_numpy(dtype=np.float)
            size = len(all_data)
            fold_size = int(size / options.kfolds)
            print("{} Total entries, split into {} folds, ~{} entries each".format(size,options.kfolds,fold_size))
            for i in range(0, options.kfolds):
                filename = "jetImage_kfold_{}.h5".format(i + 1)
                fold_start = (i * fold_size)
                fold_end = (i + 1) * fold_size
                print("Spitting fold {} as entries {} to {} ({} entries)".format(i + 1, fold_start, fold_end-1, len(all_data[fold_start:fold_end])))
                with h5py.File(os.path.join(options.outputDir, filename), 'w') as new_file:
                    new_file.create_dataset('jetFeatureNames', data=feature_names)
                    new_file.create_dataset('jets', data=all_data[fold_start:fold_end])
                print("-----------> Generated k-fold split data file at {}".format(os.path.join(options.outputDir, filename)))
        elif os.path.isfile(data_path):
            print("Single data file found!")
            h5File = h5py.File(data_path, 'r', libver='latest', swmr=True)
            # Convert to dataframe
            columns_arr = np.array(h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
            features_labels_df = pd.DataFrame(h5File["jets"][:], columns=columns_arr)
        else:
            print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")
            print("Given Path: {}".format(data_path))
