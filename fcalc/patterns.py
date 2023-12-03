import numpy as np
import torch


class IntervalPattern:
    def __init__(self, test, train, device=torch.device("cuda")) -> None:
        if isinstance(test, np.ndarray):
            self.low = np.minimum(test, train)
            self.high = np.maximum(test, train)
        else:
            self.low = torch.minimum(test, train)
            self.high = torch.maximum(test, train)

class CategoricalPattern:
    def __init__(self, test, train) -> None:
        self.mask = list(map(lambda x, y: x == y, test, train))
        self.vals = test[self.mask]
