import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pandas as pd

from SignDetection.dataloader import DatasetRetriever

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