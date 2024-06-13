import os 
import librosa # audio processing library

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

    # loop through all the genres - enumarate returns the index and the genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)): # dirpath: path to the current directory, dirnames: list of directories in the current directory, filenames: list of files in the current directory
        
        # ensure that we are not at the root level
        if dirpath is not dataset_path:
                
                # save the semantic label
                dirpath_components = dirpath.split("/") # genre/blues => ["genre", "blues"]
                semantic_label = dirpath_components[-1]
                data["mapping"].append(semantic_label)

                # process files for a specific genre

                for f in filenames:
                         
                        # load the audio file
                        file_path = os.path.join(dirpath, f)
                        signal, sr = librosa.load(file_path, sr=22050) # sr = 22050 is the sample rate

                        # process segments extracting mfcc and storing data
                        for s in range(num_segments):
                            start_sample = num_samples_per_segment * s # s = 0 -> 0
                            finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment

                            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        num_mfcc=num_mfcc,
                                                        hop_length=hop_length
                                                        )