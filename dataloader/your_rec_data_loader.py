import mxnet as mx
from torch.utils.data import Dataset
import torch
import numpy as np

class MXNetDataset(Dataset):
    def __init__(self, rec_path, idx_path):
        self.rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        self.num_samples = len(self.rec)  # Sử dụng số lượng mẫu có sẵn từ MXIndexedRecordIO

    def __getitem__(self, index):
        item = self.rec.read_idx(index)
        header, img = mx.recordio.unpack(item)
        # Xử lý ảnh, ví dụ: chuẩn hóa hoặc chuyển sang tensor PyTorch
        img = np.array(img)
        img = (img - 127.5) / 128.0  # Chuẩn hóa
        img = torch.from_numpy(img).float()  # Chuyển sang Tensor PyTorch
        return img

    def __len__(self):
        return self.num_samples
