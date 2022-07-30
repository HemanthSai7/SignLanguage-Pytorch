# !pip install -q albumentations
# !pip install -q efficientnet_pytorch
# !pip install -q catalyst

#Import important libraries
import cv2
import time
import json
import datetime
from pathlib import Path

import warnings
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from efficientnet_pytorch import EfficientNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data.sampler import SequentialSampler

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore",category=UserWarning)
SEED=42

#Read the datasets
df=pd.read_csv("./datasets/Train.csv")

def convert(df): #Convert the dataframe to a list of dictionaries
    SignDict={"Seat":0,"Enough/Satisfied":1,"Mosque":2,"Temple":3,"Friend":4,"Me":5,"Church":6,"You":7,"Love":8}
    df['Label']=df['Label'].apply(lambda x:SignDict[str(x)])
    return df['Label']
convert(df)

#EDA
print(df.shape)
print(df['Label'].value_counts())

#Get Model
def Net(model_name='b3',output=9):
    model=EfficientNet.from_pretrained(f'efficientnet-{model_name}')
    model._fc=nn.Linear(in_features=model._fc.in_features,
                        out_features=output,bias=True)
    return model  
Net()

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

#Image Augmentations
def train_augmentations(img_size=512):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1),
        A.RandomSizedCrop(min_max_height=(int(img_size-0.2*img_size), int(img_size-0.2*img_size)),
                          height=img_size, width=img_size, p=0.5),  
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(max_holes=8, max_width=12,
                        max_height=12, fill_value=0, p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.0)


def validation_augumentations(img_size=512):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

obj = DatasetRetriever(df.img_IDS.values, df.Label.values,
                       validation_augumentations())
images, labels = obj[6003]
plt.imshow(images.numpy().transpose(1, 2, 0))
plt.axis('off')
print(labels) 

#Configuration
class Config:
    output=9
    num_workers=4
    batch_size=32

    img_size=224
    n_epochs=200
    lr=0.0003
    patience=5

    SchedulerClass=torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params=dict(
        mode='min',
        factor=0.8,
        patience=1,
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )

#Train Loader    
train_loss=dict()
valid_loss=dict()

class Train:
    
    def __init__(self,model,device,config,fold,model_name='b0',image_size=384, weight_path='./'):
  
        self.model=model
        self.device=device
        self.config=config
        self.best_score=0
        self.best_loss=5000
        self.fold=fold
        self.model_name = model_name
        self.image_size = image_size
        self.weight_path = weight_path
        

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
    def fit(self,train_loader,validation_loader):
        for epoch in range(self.config.n_epochs):

            print("Training Started...")
            t=time.time()
            Loss = self.train_one_epoch(train_loader)
            train_loss[epoch] = Loss.avg

            print(f'Train : Epoch {epoch}: | Loss: {Loss.avg} | Training time: {time.time()-t}')
            
            t=time.time()
            print("Validation Started...")
            loss = self.validation(validation_loader)
            valid_loss[epoch] = Loss.avg

            print(f'Valid : Epoch {epoch}: | Loss: {Loss.avg} | Training time: {time.time()-t}')
            
            self.scheduler.step(metrics=Loss.avg)
            
            if not self.best_score:
                self.best_score = Loss.avg
                print(f'Saving model with best CE Loss as {self.best_score}')
                self.model.eval()   
                patience = self.config.patience
                torch.save({'model_state_dict': self.model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  f"{self.weight_path}/{self.image_size}_{self.model_name}_{self.fold}.pt")
                continue  

            if Loss.avg <= self.best_score:
                self.best_score = Loss.avg
                patience = self.config.patience
                print('Improved model with best CE loss as {}'.format(self.best_score))
                torch.save({'model_state_dict': self.model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  f"{self.weight_path}/{self.image_size}_{self.model_name}_{self.fold}.pt")
            else:
                patience -= 1
                print('Patience Reduced')
                if patience == 0:
                    print(f'Early stopping. Best CE Loss: {self.best_score}')
                    break
                    
    def validation(self, val_loader):
        self.model.eval()
        Loss = AverageMeter()

        t = time.time()

        for steps,(images, targets) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                targets = targets.to(self.device, dtype=torch.long)
                batch_size = images.shape[0]               
                images = images.to(self.device, dtype=torch.float32)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                Loss.update(loss.detach().item(), batch_size)
        return Loss

    def train_one_epoch(self, train_loader):
        self.model.train()

        Loss = AverageMeter()

        t = time.time()

#         print("hello before vala")
        for steps,(images, targets) in enumerate(tqdm(train_loader)):
#             print(targets)
#             print("hello after ")
#             targets=torch.tensor(targets)
#             print(type(targets))
            targets = torch.tensor(targets).to(self.device,dtype=torch.long)
#             print(type(targets))
            
            
            batch_size = images.shape[0]               
            images = images.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            Loss.update(loss.detach().item(), batch_size)

        return Loss
                           
#Helper Function 
#this is just a good practice of calcualting and storing the values. Not a compulsion. Check ImageNet code from pytorch examples for more info
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
#Stratified KFold Split
base_dir = Path('datasets')
train_df = df
with open(f'../input/signlanguage/datasets/label_num_to_sign_map.json') as f:
    class_names = json.loads(f.read())
f.close() 

train_df['label_name'] = train_df['Label'].apply(lambda x: class_names[str(x)])

sk = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
for fold, (train, val) in enumerate(sk.split(train_df, train_df['Label'])):
    train_df.loc[val, 'fold'] = fold
train_df.fold = train_df.fold.astype(int)
print(train_df.fold.value_counts())
print(train_df[train_df['fold']==0].img_IDS.values)

#Test Loader
def test(fold_number=0,model_name='b3',image_size=512,weight_path='./',load_weights_path=None):
    train_X = train_df[train_df['fold'] != fold_number].img_IDS.values
    train_Y = train_df[train_df['fold'] != fold_number].Label.values
    valid_X = train_df[train_df['fold'] == fold_number].img_IDS.values
    valid_Y = train_df[train_df['fold'] == fold_number].Label.values

    train_dataset=DatasetRetriever(train_X,train_Y,train_augmentations(img_size=image_size))
    valid_dataset=DatasetRetriever(valid_X,valid_Y,validation_augumentations(img_size=image_size))

    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(),mode='downsampling'),
        batch_size=Config.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=Config.num_workers,
    )
    validation_loader=torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
    )
    model=Net(model_name=model_name).cuda()
    if load_weights_path is not None:
        model.load_state_dict(torch.load(load_weights_path + f"{image_size} {model_name}_{fold_number}.pt")["model_state_dict"])
        print("Weight Loaded")
    engine = Train(model=model, device=torch.device('cuda'), config=Config, fold=fold_number,
    model_name=model_name, image_size=image_size, weight_path=weight_path)
    engine.fit(train_loader, validation_loader)

test(fold_number=0,model_name='b3',image_size=512)    