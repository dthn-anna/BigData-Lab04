from typing import Tuple
import numpy as np

class Normalize:

    # Chuẩn hóa ảnh đầu vào theo từng kênh màu 
    """
    Normalizes the input 3D matrix with the given mean and standard deviation. 
    Performs per channel normalization of input matrix.

    Args
    ----

    mean (Tuple) : Tuple of means of Red Channel, Green Channel, Blue Channel
    std (Tuple) : Tuple of std of Red Channel, Green Channel, Blue Channel

    Returns
    -------
    Normalized `numpy.ndarray` with same shape as input 
    """

    def __init__(self,mean: Tuple, std: Tuple) -> None:
        self.mean = mean # Tuple chứa 3 giá trị trung bình của các kênh R, G, B.
        self.std = std # Tuple chứa 3 độ lệch chuẩn của các kênh R, G, B.

    def transform(self,matrix: np.ndarray) -> np.ndarray:
        # input: ảnh đầu vào dạng np.ndarray, kích thước (height, width, 3). Tức là ảnh màu RGB 
        shape = matrix.shape

        # chuẩn hóa giá trị pixel về [0,1]. 
        # Ảnh RGB gốc thường có giá trị pixel từ 0-255 (uint8), nền cần chuẩn hóa về khoảng [0,1]
        matrix = matrix/255.0

        # Chuyển shape từ (height, width, chanel) thành (chanel, height, width)
        matrix = matrix.transpose(2,0,1)
        r = matrix[0] # kênh đỏ
        g = matrix[1] # kênh xanh lá
        b = matrix[2] # kênh xanh dương 

        # áp dụng chuẩn hóa cho từng kênh. 
        # Công thức: X_normalized  =  (X−μ) / σ
        r = (r-self.mean[0])/self.std[0]
        g = (g-self.mean[1])/self.std[1]
        b = (b-self.mean[2])/self.std[2]

        # gán ngược các kênh đã chuẩn hóa vào ma trận 
        matrix[0] = r
        matrix[1] = g
        matrix[2] = b

        # Chuyển về shape ban đầu (chanel, height, width) --> (height, width, chanel)
        matrix = matrix.transpose(1,2,0)

        # Kiểm tra lại shape --> nhằm đảm bảo shape đầu ra giống shape ảnh đầu vào. 
        assert matrix.shape == shape

        # Trà về ma trận đã chuẩn hóa 
        return matrix
