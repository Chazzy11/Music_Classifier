import os 

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5): 

    # dictionary to store data - mapping we need to store the mapping of the class labels to integers
    data = {
        "mapping": [],
        "mfcc": [], # training data
        "labels": [] # expected output
    }

    # loop through all the genres - enumarate returns the index and the genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)): # dirpath: path to the current directory, dirnames: list of directories in the current directory, filenames: list of files in the current directory
        
        # ensure that we are not at the root level