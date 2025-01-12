import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from torch import Tensor


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out



class SparsePromptLearner(nn.Module):
    def __init__(self):
        super().__init__()

        self.crossAttention = nn.MultiheadAttention(embed_dim=256, num_heads=8)  
        self.selfAttention_prototype = nn.MultiheadAttention(embed_dim=256, num_heads=8) 
        self.selfAttention_Correlation = nn.MultiheadAttention(embed_dim=256, num_heads=8)

        self.learnable_query_embed=(nn.Embedding(100, 256).weight).unsqueeze(1)
        self.learnable_query_embed=self.learnable_query_embed.cuda()

        self.prototypeFgConv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.CorrelationFgConv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn = nn.BatchNorm2d(1)
        self.FFN = FFN(256, 128)
        self.FinalConv = nn.Conv1d(200, 10, kernel_size=1, stride=1, padding=0)


    def forward(self, ImgEmbedding: torch.Tensor,prototypeFg: torch.Tensor,CorrelationFg: torch.Tensor) -> torch.Tensor:

        latentEmbedding=self.crossAttention(query=self.learnable_query_embed,
                                            key=ImgEmbedding,
                                            value=ImgEmbedding)[0]  
        
        prototypeFgConv=self.prototypeFgConv(prototypeFg).unsqueeze(0) 
        CorrelationFgConv=self.CorrelationFgConv(CorrelationFg).unsqueeze(0) 

        prototypeFgEmbedding=prototypeFgConv.flatten(2).permute(2,0,1)
        CorrelationFgEmbedding=CorrelationFgConv.flatten(2).permute(2,0,1)

        prototypeFgEmbedding=self.crossAttention(query=latentEmbedding,
                                    key=prototypeFgEmbedding,
                                    value=prototypeFgEmbedding)[0].permute(1,0,2)  
        
        CorrelationFgEmbedding=self.crossAttention(query=latentEmbedding,
                                    key=CorrelationFgEmbedding,
                                    value=CorrelationFgEmbedding)[0].permute(1,0,2) 
        
        sparseEmbedding=torch.cat([prototypeFgEmbedding,CorrelationFgEmbedding], dim=1)
        sparseEmbedding=self.FFN(sparseEmbedding)
        sparseEmbedding=self.FinalConv(sparseEmbedding)

        return sparseEmbedding
