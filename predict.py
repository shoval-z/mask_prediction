import os
import argparse
import pandas as pd
import pickle
import sys
import torch
from dataset import mask_dataset
from utils import *

def make_prediction(model, device, test_loader,test_dataset):
    data_dict = {'filename': [], 'x':[], 'y':[], 'w':[],'h':[], 'proper_mask':[]}
    print('strat eval')
    model.eval()
    with torch.no_grad():
        for idx,(x, y_bb, y_class) in enumerate(test_loader):
            file_name = test_dataset.image_id[idx]
            origin_size = test_dataset.image_sizes[idx]
            origin_size = origin_size.squeeze(1).to(device)
            x = x.to(device).float()
            out_class, out_bb = model(x)
            out_bb = torch.mul(out_bb, origin_size)

            pred = True if (out_class > 0.5).float() else False
            data_dict['filename'].append(file_name)
            data_dict['x'].append(out_bb[0])
            data_dict['y'].append(out_bb[1])
            data_dict['w'].append(out_bb[2])
            data_dict['h'].append(out_bb[3])
            data_dict['proper_mask'].append(pred)
    return data_dict


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
parser.add_argument('batch_size', type=int,default=32,help='Number of Batch wanted')
parser.add_argument('workers', type=int, default=4, help='Input folder path, containing images')
args = parser.parse_args(['example_images', '32' ,'4'])

# Load Trained model
# TODO- fill the model name
yolo_model_loaded = torch.load(open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'data/XXXXX.pkl'), "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = mask_dataset(dataset='test', path=args.input_folder)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

data_dict = make_prediction(yolo_model_loaded, device, test_loader,test_dataset)
prediction_df = pd.DataFrame(data_dict)
prediction_df.to_csv("prediction.csv", index=False, header=True)