import numpy as np

class ColorShift:
    def __init__(self,r: int, g: int, b: int, p: float) -> None:
        self.r = r # Số lượng pixel dịch kênh đỏ (R)
        self.g = g # Số lượng pixel dịch kênh xanh lá (G)
        self.b = b # Số lượng pixel dịch kênh xanh dương (B)
        self.p = p # Xác suất áp dụng phép biến đổi

    def transform(self, image: np.ndarray) -> np.ndarray:
        # image: [height, width, 3]
        color_shift_image = image 
        # sinh số ngẫu nhiên ∈ [0, 1). Nếu số đó lớn hơn p thì áp dụng dịch màu. 
        if np.random.rand() >= self.p:
            # Tách riêng từng kênh màu từ ảnh 
            # 0, 1, 2: số kênh màu (Red, Green, Blue)
            R = image[:,:,0]
            G = image[:,:,1]
            B = image[:,:,2]

            # np.dstack : ghép 3 ma trận kênh R,G,B thành hình ảnh 3 kênh RGB mới 
            color_shift_image = np.dstack( (
                # np.roll(arr, shift, axis): dịch mảng arr, số lượng pixel là shift, axis: quy định chiều dịch 
                np.roll(R, self.r, axis=0), # Dịch kênh R theo chiều dọc
                np.roll(G, self.g, axis=1), # Dịch kênh G theo chiều ngang
                np.roll(B, self.b, axis=0)  # Dịch kênh B theo chiều dọc
                ))
        return color_shift_image


# Data augmentation:  tăng cường dữ liệu ảnh 
# Tăng sự đa dạng ảnh đầu vào cho mô hình học sâu
# Dịch từng kênh màu RGB trong ảnh theo 1 số lượng pixel cụ thể
# Tạo ảnh mới bị lệch màu một cách ngẫu nhiên mà vãn giữ được cấu trúc ảnh gốc. 
