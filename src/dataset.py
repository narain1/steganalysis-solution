import torch
from torch import nn, optim
import pandas as pd
import jpegio
import numpy as np

# dataframe read
df = pd.read_csv('../data/train.csv')

# image_read functions
def JPEFdecompressYCbCr(f):
    jpegStruct = jpegio.read(f)
    [col, row] = np.meshgrid(range(8), range(8))
    T = 0.5* np.cos(np.pi * (2*col +1) * row / (2*8))
    T[0, :] = T[0, :]/ np.sqrt(2)

    img_dims = np.array(jpegStruct.coef_arrays[0].shape)
    n_blocks = img_dims//8
    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)

    YCbCr = []
    for i, dct_coeffs in enumerate(jpegStruct.coef_arrays):
        if i==0:
            QM = jpegStruct.quant_tables[i]
        else:
            QM = jpegStruct.quant_tables[1]

        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)
        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)
        dct_coeffs = dct_coeffs.reshape(broadcast_dims)

        a = np.transpose(t, axes = (0, 2, 3, 1))
        b = (qm * dct_coeffs).transpose(0,2,1,3)
        c = t.transpose(0, 2, 1, 3)

        z = a @ b @ c
        z = z.transpose(0, 2, 1, 3)
        YCbCr.append(z.reshape(img_dims))

    return np.stack(YCbCr, -1).astype(np.float32)

encode_qf_dict = {j:i for i, j in enumerate(sorted(df['qf'].values))}

class ImageOpen:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

proc_1 = ImageOpen(cv2.imread(), cv2.COLOR_BGR2RGB)

# Dataset
class ImageDataset(Dataset):
    def __init__(self, df, tfms, f_open):
        self.image_path = df['path']
        self.ys = df['target'].astype(np.float32).values
        self.qf = df['qf'].values
        self.tfms = tfms
        self.f_open = f_open

    def __getitem__(self, idx):
        img = self.f_open(self.image_path[idx])
        return self.tfms(image = img), self.qf[idx], self.ys[idx]

    def __len__(self):
        return len(self.image_path)
