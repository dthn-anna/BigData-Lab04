import numpy as np


# Lật ảnh theo chiều ngang (trái ↔ phải) với xác suất p.
class RandomHorizontalFlip:
    def __init__(self, p:float) -> None:
        self.p = p # Xác suất áp dụng lật ngang

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        # (height, weight, chanel) --> (chanel, height, weight)
        matrix = matrix.transpose(2,0,1)

        # Đảo ngược trục chiều rộng (chiều ngang)
        if np.random.rand() >= self.p:
            matrix = matrix[:,:,::-1]

        return matrix.transpose(1,2,0)


# Lật ảnh theo chiều dọc (trên ↕ dưới) với xác suất p.
class RandomVerticalFlip:
    def __init__(self, p:float) -> None:
        self.p = p

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.transpose(2,0,1)
        if np.random.rand() >= self.p:
            matrix = matrix[:,::-1]
        return matrix.transpose(1,2,0)