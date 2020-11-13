import torch
import pandas as pd
import h5py
import io
import tables
import random
import os
import numpy as np
import yaml
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
        parser.add_option('-r', '--random', action='store', type='int', dest='random',
                          default='25', help='% of labels to randomize')
        (options, args) = parser.parse_args()
        yamlConfig = parse_config(options.config)

        data_path = options.inputFile

        # List of features to use
        features = yamlConfig['Inputs']
        features_list = features
        # List of labels to use
        labels = yamlConfig['Labels']
        labels_list = labels

        if os.path.isdir(data_path):
            print("Directory of data files found!")
            first = True
            for file in os.listdir(data_path):
                if file.endswith(".h5") or file.endswith(".h5df"):
                    try:
                        print("Loading " + str(file))
                        h5File = h5py.File(os.path.join(data_path, file), 'r', libver='latest', swmr=True)
                        columns_arr = np.array(h5File['jetFeatureNames'][:]).astype(str)  # slicing h5 data because otherwise it's a reference to the actual file?
                        this_file = pd.DataFrame(h5File["jets"][:], columns=columns_arr)
                        features_labels_df = pd.DataFrame(h5File["jets"][:], columns=columns_arr)
                        features_df = features_labels_df[features]
                        labels_df = features_labels_df[labels]
                        success = True
                    except:
                        print("Error! Failed to load jet file " + file)
                        success = False

                    if success:
                        size = len(features_labels_df.index)
                        random_labels = int(size*(options.random/100.0))
                        for i, row in features_labels_df[:random_labels].iterrows(): # iterate over the first N% of data
                            valid_labels = labels_list[:]
                            current_val = labels_df.columns[(labels_df.iloc[i] == 1.0)][0] #find the current value
                            valid_labels.remove(current_val) #remove previous value to make sure new is different
                            features_labels_df.at[i, current_val] = 0.0 #set previous "label" column to 0
                            new_label_loc = random.choice(valid_labels)  # Randomize our label, making sure to set it to something different
                            features_labels_df.at[i, new_label_loc] = 1.0 #Set our new random label to true in the df
                            #print("\t -> Modified Row: {}".format(features_labels_df.loc[i, labels_list]))
                        filename = "{}rand_".format(options.random)+file
                        with h5py.File(os.path.join(options.outputDir, filename), 'w') as new_file:
                            new_file.create_dataset('jetFeatureNames', data=np.array(h5File['jetFeatureNames'][:]))
                            new_file.create_dataset('jets', data=features_labels_df.to_numpy(dtype=np.float))
                        h5File.close()
                        print("-----------> Generated randomized data file at {}".format(filename))
        else:
            print("Error! path specified is a special file (socket, FIFO, device file), or isn't valid")

        features_labels_df = features_labels_df.drop_duplicates()
        features_df = features_labels_df[features]
        labels_df = features_labels_df[labels]
