from functools import reduce
from dataclasses import dataclass
from torchvision.datasets import VisionDataset
from typing import Callable, Type, Tuple, Optional
from torch.utils.data import DataLoader, random_split

from .nn_dataset_base import NNDatasetBase

@dataclass(frozen=True, kw_only=True, slots=True)
class NNDataset(NNDatasetBase):
    ds_class        : Type[VisionDataset]
    root_dir        : str                   = "./data"
    download        : bool                  = True
    transform       : Optional[Callable]    = None
    batch_sizes     : Tuple[int, int, int]  = (None, None, None)
    val_proportion  : float                 = 0.1
    
    def __post_init__(self):        
        train_dataset, non_train_dataset = (
            self.ds_class(root=self.root_dir, train=True, download=self.download, transform=self.transform)
            , self.ds_class(root=self.root_dir, train=False, download=self.download, transform=self.transform)
        )
        
        val_dataset, test_dataset = random_split(
            non_train_dataset
            , [
                int(len(non_train_dataset) * self.val_proportion)
                , int(len(non_train_dataset) * (1 - self.val_proportion))
            ]
        )

        object.__setattr__(
            self
            , 'train_loader'
            , DataLoader(
                shuffle=True
                , dataset=train_dataset
                , batch_size=self.batch_sizes[0] or len(train_dataset)
            )
        )

        object.__setattr__(
            self
            , 'val_loader'
            , DataLoader(
                shuffle=False
                , dataset=val_dataset
                , batch_size=self.batch_sizes[1] or len(val_dataset)
            )
        )

        object.__setattr__(
            self
            , 'test_loader'
            , DataLoader(
                shuffle=False
                , dataset=test_dataset
                , batch_size=self.batch_sizes[2] or len(test_dataset)
            )
        )
        
        object.__setattr__(
            self
            , 'input_dim'
            , reduce(lambda x, y: x * y, self.train_loader.dataset[0][0].shape)
        )
        
        object.__setattr__(
            self
            , 'output_dim'
            , len(self.train_loader.dataset.classes)
        )