# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:10:08 2023

@author: 28257
"""

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
import whisper

from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torch.nn.functional as F
from loguru import logger

class WhisPromptModel(nn.Module):
    def __init__(self,cls_num,prompt_len):
        super(WhisPromptModel, self).__init__()
        self.cls_num=cls_num
        self.prompt_len=prompt_len
        
        model=whisper.load_model("base")
        self.dim=model.dims.n_audio_state
        
        whisper_encoder=model.encoder
        for param in whisper_encoder.parameters():
            param.requires_grad=False
        
        self.conv1=whisper_encoder.conv1
        self.conv2=whisper_encoder.conv2
        self.position_embed=whisper_encoder.positional_embedding
        
        self.cls_prompt=nn.Parameter(torch.zeros(1+self.prompt_len,self.dim), requires_grad=True)
        
        self.encoder_blocks=whisper_encoder.blocks
        self.ln_post=whisper_encoder.ln_post
        
        self.full_connect=nn.Linear(self.dim, self.cls_num)
        self.softmax=nn.Softmax(dim=1)
        self.loss_fn=nn.CrossEntropyLoss()
    
    
    def forward(self,x,labels=None):
        bs=x.size(0)
        
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.position_embed).to(x.dtype)
        cls_prompt=self.cls_prompt.unsqueeze(0).expand(bs,1+self.prompt_len,self.dim)
        x=torch.cat([cls_prompt,x],dim=1)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.ln_post(x)
        
        logits=self.full_connect(x[:,0,:])
        prob=self.softmax(logits)
        if labels is not None:
            loss=self.loss_fn(prob,labels)
            return (prob,loss)
        return prob
