import mxnet as mx
from torch.utils.data import Dataset

class MXNetDataset(Dataset):
    def __init__(self, rec_path, idx_path):
        self.rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

    def __getitem__(self, index):
        item = self.rec.read_idx(index)
        header, img = mx.recordio.unpack(item)
        # Bạn có thể thêm các bước tiền xử lý hình ảnh tại đây
        return img

    def __len__(self):
        return len(self.rec.keys())
