import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.datasets as datasets

#Data Loader
class DatasetRetriever(Dataset):

    def __init__(self, image_ids, labels, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'../input/signlanguage/datasets/Images/{image_id}.jpg', cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image, label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)