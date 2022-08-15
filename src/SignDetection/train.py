import pandas as pd

import torch
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data.sampler import SequentialSampler

from dataloader import DatasetRetriever
from augmentations import train_augmentations,validation_augumentations
from train_config import Config
from utils.get_dataset import convert
from StratifiedKFold import data_split_StratifiedKFold
from model import Net
from engine import Train

#Get Dataset
df=pd.read_csv("./datasets/Train.csv")
df=convert(df)

#Stratified KFold Split
train_df = data_split_StratifiedKFold(df)

#trainig and validation
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