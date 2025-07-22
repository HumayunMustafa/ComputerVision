import argparse
from typing import Optional,Tuple,Union

import torch 
from torch import Tensor

class BaseAnchorGenerator(torch.nn.Module):
    """
    Base class for anchor generators for the task of object detection.
    """
    def __init__(self,*args,**kwargs) -> None:
        super().__init__()
        self.anchor_dict=dict()

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add anchor generator-specific arguments to the parser
        """
        return parser

    def number_of_anchors_per_os(self):
        """Returns anchors per output stride. Child classes must implement this function."""
        raise NotImplementedError

    @torch.no_grad()
    def __generate_anchors(
            self,
            height: int,
            width: int,
            output_stride: int,
            device: Optional[str] = "cpu",
            *args,
            **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError

    @torch.no_grad()
    def __get_anchors(
            self,
            fm_height : int,
            fm_width : int,
            fm_output_stride : int,
            device: Optional[str] = "cpu",
            *args,
            **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        key = "h_{}_w_{}_os_{}".format(fm_height, fm_width, fm_output_stride)
        if key not in self.anchor_dict:
            default_anchors_ctr = self.__generate_anchors(
                height=fm_height,
                width=fm_width,
                output_stride=fm_output_stride,
                device=device,
                *args,
                **kwargs
            )
            self.anchor_dict[key] = default_anchors_ctr
            return default_anchors_ctr
        else:
            return self.anchor_dict[key]

    @torch.no_grad()
    def forward(
            self,
            fm_height : int,
            fm_width : int,
            fm_output_stride : int,
            device: Optional[str] = "cpu",
            *args,
            **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Returns anchors for the feature map

        Args:
            fm_height (int): Height of the feature map
            fm_width (int): Width of the feature map
            fm_output_stride (int): Output stride of the feature map
            device (Optional, str): Device (cpu or cuda). Defaults to cpu

        Returns:
            Tensor or Tuple of Tensors
        """
        return self.__get_anchors(
            fm_height=fm_height,
            fm_width=fm_width,
            fm_output_stride=fm_output_stride,
            device=device,
            *args,
            **kwargs
        )



