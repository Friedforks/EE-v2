from torch import nn
import torch
from fastkan import FastKAN as KAN
# from kan import KAN,MultKAN

class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model:int,hidden: int,drop_prob:float=0.1,kan: bool=False)->None:
        super(PositionWiseFeedForward,self).__init__()
        self.kan=kan
        if kan: 
            self.fc1=KAN([d_model,hidden])
            self.fc2=KAN([hidden,d_model])
            self.dropout=nn.Dropout(p=drop_prob)
            
        else:
            self.fc1=nn.Linear(d_model,hidden)
            self.relu=nn.ReLU()
            self.fc2=nn.Linear(hidden,d_model)
            self.dropout=nn.Dropout(p=drop_prob)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.fc1(x)
        if not self.kan:
            x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x