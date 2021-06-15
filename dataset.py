import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from utils import transform
import os
from PIL import Image


class mask_dataset(Dataset):
    def __init__(self, dataset='train'):
        super(mask_dataset, self).__init__()
        self.path = f'/home/student/{dataset}'
        self.image_id = os.listdir(self.path)


    def __getitem__(self, index):
        img = self.image_id[index]
        img_items = img.strip(".jpg").split('__')
        box = json.loads(img_items[1])
        label = [2] if img_items[2] == 'True' else [1]
        image_tensor = Image.open(os.path.join(self.path, img)).convert('RGB')
        boxes = torch.FloatTensor(box)
        labels = torch.LongTensor(label)
        image, boxes, labels = transform(image_tensor, boxes, labels)
        return image, boxes, labels

    def __len__(self):
        return len(self.image_id)

if __name__ == '__main__':
    data = mask_dataset('train')
    for x in data:
        img, box, label = x
