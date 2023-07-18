from __future__ import annotations

from dataclasses import dataclass

from ..enum.loss import Loss
from ..enum.device import Device

@dataclass(frozen=True, kw_only=True, slots=True)
class NNModelParams:
    loss  : Loss
    device: Device
    
    def __str__(self):
        return f"[device={self.device}, loss={self.loss}]"
    
    def is_valid(self):
        return (
            self.device is not None
            and self.loss is not None
        )
        
    def to_dict(self) -> dict:
        return dict(
            loss        = str(self.loss)
            , device    = str(self.device)
        )
        
    @staticmethod
    def from_dict(rep: dict) -> NNModelParams:
        return NNModelParams(
            loss        = Loss(rep['loss'])
            , device    = Device(rep['device'])
        )