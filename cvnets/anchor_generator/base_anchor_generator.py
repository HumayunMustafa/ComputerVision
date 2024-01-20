import argparse
from typing import Optional,Tuple,Union

import torch 
from torch import Tensor

class BaseAnchorGenerator(torch.nn.Module):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__()
        self.anchor_dict=dict()
