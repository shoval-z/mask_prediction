import os
import argparse
import numpy as np
import pandas as pd


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
# TODO - your prediction code here

# Example (A VERY BAD ONE):
bbox_pred = np.random.randint(0, high=224, size=(4, len(files)))
proper_mask_pred = np.random.randint(2, size=len(files)).astype(np.bool)
prediction_df = pd.DataFrame(zip(files, *bbox_pred, proper_mask_pred),
                             columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=True)
