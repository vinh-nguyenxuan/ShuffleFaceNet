import mxnet as mx
from torch.utils.data import Dataset
import torch
import numpy as np

class MXNetDataset(Dataset):
    def __init__(self, rec_path, idx_path):
        # Mở tệp .rec và .idx
        self.rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        
        # Đọc các chỉ số từ tệp .idx để tính số lượng mẫu
        with open(idx_path, 'r') as f:
            self.num_samples = len(f.readlines())  # Đếm số dòng trong tệp chỉ mục .idx để tính số mẫu

    def __getitem__(self, index):
        # Đọc mục từ tệp .rec theo chỉ số
        item = self.rec.read_idx(index)
        header, img = mx.recordio.unpack(item)
        # Xử lý ảnh: chuẩn hóa và chuyển đổi sang Tensor PyTorch
        img = np.array(img)
        img = (img - 127.5) / 128.0  # Chuẩn hóa
        img = torch.from_numpy(img).float()  # Chuyển sang Tensor PyTorch
        return img

    def __len__(self):
        return self.num_samples
