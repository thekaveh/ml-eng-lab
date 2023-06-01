import torch_geometric as pyg

from dataclasses import dataclass
from torch_geometric.data import Dataset
from torch_geometric.loader import NeighborLoader
from typing import Callable, Type, Tuple, Optional, List

from .nn_dataset_base import NNDatasetBase

@dataclass(frozen=True, kw_only=True, slots=True)
class NNGraphDataset(NNDatasetBase):
    ds_class        : Type[Dataset]
    n_neighbors     : List[int]
    root_dir        : str                   = "./data"
    transform       : Optional[Callable]    = None
    n_workers       : int                   = 4
    batch_sizes     : Tuple[int, int, int]  = (None, None, None)
    
    def __post_init__(self):
        dataset = self.ds_class(root=self.root_dir, transform=self.transform)
        
        object.__setattr__(
            self
            , 'train_loader'
            , NeighborLoader(
                shuffle=True
                , data=dataset._data
                , num_workers=self.n_workers
                , num_neighbors=self.n_neighbors
                , input_nodes=dataset._data.train_mask
                , batch_size=self.batch_sizes[0] or int(dataset._data.train_mask.sum())
            )
        )
        
        object.__setattr__(
            self
            , 'val_loader'
            , NeighborLoader(
                shuffle=False
                , data=dataset._data
                , num_workers=self.n_workers
                , num_neighbors=self.n_neighbors
                , input_nodes=dataset._data.val_mask
                , batch_size=self.batch_sizes[1] or int(dataset._data.val_mask.sum())
            )
        )
        
        object.__setattr__(
            self
            , 'test_loader'
            , NeighborLoader(
                shuffle=False
                , data=dataset._data
                , num_workers=self.n_workers
                , num_neighbors=self.n_neighbors
                , input_nodes=dataset._data.test_mask
                , batch_size=self.batch_sizes[2] or int(dataset._data.test_mask.sum())
            )
        )
        
        object.__setattr__(
            self
            , 'input_dim'
            , dataset.num_features
        )
        
        object.__setattr__(
            self
            , 'output_dim'
            , dataset.num_classes
        )