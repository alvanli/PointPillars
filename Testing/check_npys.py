import glob 
import os
import numpy as np

# base_dir = "/mnt/wato-drive/perception_pcds/cars"
base_dir = "/mnt/wato-drive/perception_pcds/cars"

if __name__ == "__main__":
    for idx, path in enumerate(sorted(glob.glob(base_dir + "/*.npy"))):
        os.rename(path, base_dir + "/pc{:05d}.npy".format(idx))
        # if len(arr) < 3:
        #     print(arr.shape)