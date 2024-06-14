import os 
import librosa # audio processing library
import math
import json

DATASET_PATH = "Data/genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5): 

    # dictionary to store data - mapping we need to store the mapping of the class labels to integers
    data = {
        "mapping": [],
        "mfcc": [], # training data
        "labels": [] # expected output
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments) 
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
    # loop through all the genres - enumarate returns the index and the genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)): # dirpath: path to the current directory, dirnames: list of directories in the current directory, filenames: list of files in the current directory
        
        # ensure that we are not at the root level
        if dirpath is not dataset_path:
                
                # save the semantic label
                dirpath_components = os.path.split(dirpath) # genre/blues => ["genre", "blues"]
                semantic_label = dirpath_components[-1]
                data["mapping"].append(semantic_label)
                print("\nProcessing {}".format(semantic_label))
                # process files for a specific genre

                for f in filenames:
                         
                        # load the audio file
                        file_path = os.path.join(dirpath, f)
                        signal, sr = librosa.load(file_path, sr=22050) # sr = 22050 is the sample rate

                        # process segments extracting mfcc and storing data
                        for s in range(num_segments):
                            start_sample = num_samples_per_segment * s # s = 0 -> 0 current segment we are in
                            finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment
                           
                            # training data must always be of the same size
                            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], # analysing a slice of the signal in current segment
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        num_mfcc=num_mfcc,
                                                        hop_length=hop_length
                                                        )
                            mfcc = mfcc.T
                            # store mfcc for segment if it has the expected length
                            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(i-1)
                                print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
         json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)