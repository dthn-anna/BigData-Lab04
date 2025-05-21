from typing import Tuple
import numpy as np
from PIL import Image # thư viện Pillow - chuyển đổi và resize ảnh 

class Resize:

    # Nhận ảnh dạng np.ndarray. Ví dụ: ảnh RGB (height, weight, 3)
    # Resize về kích thước chỉ định. Ví dụ: (height, weight) mong muốn. 
    """
    Resizes the input image to the given image size. 

    Args
    ----
    size (Tuple) : Size of the new image.

    Returns
    -------
    Resized `numpy.ndarray`
    """
    
    def __init__(self, size: Tuple) -> None:
        self.size = size
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        # Ép kiểu ảnh về uint8 (0-255) để tương thích với định dạng ảnh thông thường. 
        # mode = 'RGB' : ảnh có 3 kênh màu 
        pil_image = Image.fromarray(image.astype(np.uint8),mode='RGB')
        pil_image = pil_image.resize(self.size)
        image = np.array(pil_image)
        return image