from typing import List
import numpy as np

class Transforms:
    def __init__(self, transforms : List) -> None:
        self.transforms = transforms

    def transform(self, input : np.ndarray) -> np.ndarray:
        for transformer in self.transforms:
            input = transformer.transform(input)

        return input
    

# Lớp Transforms bao gồm các phép biến đổi: resize, normalize, RandomHorizontalFlip, ColorShift...
# Mỗi phép biến đổi trong lớp đều có phương thức .transform() 