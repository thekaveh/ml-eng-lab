from typing import List

class Consts:
    FEATURES_SIZE_OUT       : int       = 10
    DS_SAMPLING_RATIO       : float     = 1.0
    LR                      : float     = 0.1
    N_EPOCHS                : int       = 20
    DROPOUT_P               : float     = 0.2
    DEVICE                  : str       = "cpu"
    OPTIMIZER_ALG           : str       = "sgd"
    MINI_BATCH_SIZE         : int       = 60000
    FEATURES_SIZE_IN        : int       = 28 * 28
    FEATURES_SIZES_HIDDEN   : List[int] = [1000]