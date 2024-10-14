import torch
import torch.nn as nn
from typing import Tuple
import math

class ScaledDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self)->None:
        super(ScaledDotProductAttention,self).__init__()
        # dim=-1 means the last dimension
        self.softmax=nn.Softmax(dim=-1)  
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size,head,length,d_tensor=k.size()
        
        k_transpose=k.transpose(2,3)
        # attention score computation by attention function
        # (batch,head,seq,d_tensor) @ (batch,head,d_tensor,seq) -> (batch,head,seq,seq)
        score=(q@k_transpose)/math.sqrt(d_tensor) 
        
        # applying mask if mask is not None
        if mask is not None:
            # fill with -1e9 to make it 0 after softmax operation
            score=score.masked_fill(mask==0,-1e9)
            
        # softmax operation
        attention=self.softmax(score)
        
        # multiply attention score with value
        
        value=attention@v
        return value,score