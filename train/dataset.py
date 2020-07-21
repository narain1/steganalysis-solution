import torch
from torch import nn, optim
import pandas as pd

df = pd.read_csv('../data/train.csv')

encode_qf_dict = {j:i for i, j in enumerate(sorted(df['qf'].values))}

class ImageDataset(Dataset):
    def __init__(self, df, tfms):
        self.image_path = df['path']
        self.ys = df['target'].astype(np.float32).values
        self.qf = df['qf'].values
        self.tfms = tfms

    def __getitem__(self, idx):
        img = cv2.imread(self.image_path[idx])
        return self.tfms(image = img), self.qf[idx], self.ys[idx]

    def __len__(self):
        return len(self.image_path)
