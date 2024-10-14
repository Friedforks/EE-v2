import torch
import torch.nn as nn
from typing import Tuple
from .ScaledProductAttention import ScaledDotProductAttention
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model: int,num_heads: int)->None:
        super(MultiHeadAttention,self).__init__()
        self.num_heads=num_heads
        self.d_model=d_model
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        # w_concat is the linear layer that will be applied to the concatenated outputs of all heads
        self.w_concat=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=-1)
        self.attention=ScaledDotProductAttention()
        
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask: torch.Tensor=None)->torch.Tensor:
        # project the queries, keys, and values in the multi-head space
        # (batch,seq,d_model) @ (d_model,d_model) -> (batch,seq,d_model)
        # print(f"q_prime: {q.shape}, k_prime: {k.shape}, v_prime: {v.shape}")
        q_prime,k_prime,v_prime=self.w_q(q),self.w_k(k),self.w_v(v)
        
        # split the queries, keys, and values in the multi-head space
        # (batch,seq,d_model) -> (batch,seq,head,d_model//head)
        q_prime,k_prime,v_prime=self.split(q_prime),self.split(k_prime),self.split(v_prime)
        
        value,score=self.attention(q_prime,k_prime,v_prime,mask)
        
        out=self.concat(value)
        out=self.w_concat(out)
        # out is [batch_size, length, d_model]
        return out
    
    def split(self,tensor:torch.Tensor)->torch.Tensor:
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size,length,d_model=tensor.size()
        d_tensor=d_model//self.num_heads
        tensor=tensor.view(batch_size,length,self.num_heads,d_tensor).transpose(1,2)
        return tensor
    
    def concat(self,tensor:torch.Tensor)->torch.Tensor:
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor