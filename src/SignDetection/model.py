import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from SignDetection.train_config import Config

#Get Model
def Net(model_name='b3',output=9):
    model=EfficientNet.from_pretrained(f'efficientnet-{model_name}')
    model._fc=nn.Linear(in_features=model._fc.in_features,
                        out_features=output,bias=True)
    return model  