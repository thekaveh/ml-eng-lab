import torch

from torchvision.datasets import VisionDataset
from typing import Callable, Type, Tuple, Optional
from torch.utils.data import DataLoader, random_split

class NNDataLoader:
    def __init__(
        self
        , ds_class      : Type[VisionDataset]
        , root_dir      : str                   = "./data"
        , download      : bool                  = True
        , transform     : Optional[Callable]    = None
        , batch_sizes   : Tuple[int, int, int]  = (None, None, None)
        , val_proportion: float                 = 0.1
    ) -> None:  
        train_dataset, non_train_dataset = (
            ds_class(root=root_dir, train=True, download=download, transform=transform)
            , ds_class(root=root_dir, train=False, download=download, transform=transform)
        )
        
        val_dataset, test_dataset = random_split(
            non_train_dataset
            , [
                int(len(non_train_dataset) * val_proportion)
                , int(len(non_train_dataset) * (1 - val_proportion))
            ]
        )
        
        self.train_loader, self.val_loader, self.test_loader = (
            DataLoader(
                shuffle=True
                , dataset=train_dataset
                , batch_size=batch_sizes[0] or len(train_dataset)
            )
            , DataLoader(
                shuffle=False
                , dataset=val_dataset
                , batch_size=batch_sizes[1] or len(val_dataset)
            )
            , DataLoader(
                shuffle=False
                , dataset=test_dataset
                , batch_size=batch_sizes[2] or len(test_dataset)
            )
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        return self.train_loader, self.val_loader, self.test_loader