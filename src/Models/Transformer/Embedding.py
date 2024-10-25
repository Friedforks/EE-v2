import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, d_model:int,max_len:int,device:torch.device)->None:
        super(PositionEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad=False
        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        _2i=torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))
        
    # x is a 2D tensor of shape (seq_len,d_model)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]
    
class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int)->None:
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)
    
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size: int,d_model: int,max_len: int,drop_prob: int,device: torch.device)->None:
        super(TransformerEmbedding,self).__init__()
        self.token_embedding=TokenEmbedding(vocab_size,d_model)
        self.position_embedding=PositionEmbedding(d_model,max_len,device)
        self.dropout=nn.Dropout(drop_prob)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        token_embed=self.token_embedding(x)
        position_embed=self.position_embedding(x)
        return self.dropout(token_embed+position_embed)
    
    