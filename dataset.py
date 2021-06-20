import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from utils import transform
import os
from PIL import Image


class mask_dataset(Dataset):
    def __init__(self, dataset):
        super(mask_dataset, self).__init__()
        self.path = f'/home/student/{dataset}'
        self.dataset = dataset
        self.image_id = os.listdir(self.path)
        self.image_sizes = list()
        self.items = list()
        for img in self.image_id:
            img_items = img.strip(".jpg").split('__')
            # if img_items[0] in ['009266','008710','004828'] and dataset == 'train':
            #     continue
            cx, cy, w, h = json.loads(img_items[1])
            if (w <= 0 or h <= 0) and dataset == 'train':
                continue
            image = Image.open(os.path.join(self.path, img)).convert('RGB')
            self.image_sizes.append(
                torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0))
            label = [2] if img_items[2] == 'True' else [1]
            bbox = [cx, cy, cx + w, cy + h]
            for idx, (b_item,img_dim) in enumerate(zip(bbox,[image.width, image.height, image.width, image.height])):
                if b_item > img_dim:
                    bbox[idx] = img_dim
            bbox = torch.FloatTensor(bbox)
            label = torch.LongTensor(label)
            if self.dataset == 'test':
                image, bbox, label = transform(image, bbox, label,dataset=self.dataset)
            self.items.append((image, bbox, label))

    def __getitem__(self, index):
        if self.dataset == 'train':
            image, bbox, label = self.items[index]
            return transform(image.copy(), torch.clone(bbox), torch.clone(label), dataset=self.dataset)
        else:
            return self.items[index]
        # img = self.image_id[index]
        # img_items = img.strip(".jpg").split('__')
        # cx,cy,w,h = json.loads(img_items[1])
        # label = [2] if img_items[2] == 'True' else [1]
        # image_tensor = Image.open(os.path.join(self.path, img)).convert('RGB')
        # bbox = [cx, cy, cx + w, cy + h]
        # bbox = torch.FloatTensor(bbox)
        # labels = torch.LongTensor(label)
        # image, bbox, labels = transform(image_tensor, bbox, labels)
        # return image, bbox, labels

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    data = mask_dataset('train')
    for x in data:
        img, box, label = x
