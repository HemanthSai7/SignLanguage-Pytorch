import time

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

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
