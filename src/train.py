import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from model import *
from dataset import *

train_aug = A.Compose([A.Flip(),
    A.Normalize(),
    A.pytorch.transforms.ToTensorV2()])

valid_aug = A.Compose([A.Normalize(), A.pytorch.transforms.ToTensorV2()])

df = pd.read_csv('../data/train.csv')

model_names = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b3']
folds = 5

def get_cbs():
    early_stopping_cb = EarlyStopping('score',
            patience = 3,
            mode = 'max'
            min_delta = 1e-4,
            verbose = True)

    checkpoint_callback = ModelCheckPoint(filepath = 'models',
            monitor = 'score'
            mode = 'max')
    return early_stopping_cb, checkpoint_callback

def get_dl(train_df, val_df):
    train_ds, val_ds = ImageDataset(train_df, train_aug, proc_1), ImageDataset(val_df, valid_aug, proc_1)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
    val_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, pin_memory=True)
    return train_dl, val_dl

def train_cv():
    for model_name in model_names:
        for bi, (train_ids, val_ids) in enumerate(kfs.split(df, df['class'])):
            train_df, val_df = df.iloc[train_ids, :], df.iloc[val_ids, :]
            train_ds, val_ds = ImageDataset(train_df, train_aug), ImageDataset(val_df, val_aug)
            train_dl = DataLoader(train_ds, pin_memory=True, shuffle=True, batch_size=32, num_workers=4)
            val_dl = DataLoader(val_ds, pin_memory=True, batch_size=32, num_workers=4)
            model = Model(model_name)
            trainer = Trainer(gpus=1,
                    checkpoint_callback = 1.0,
                    early_stop_callback = early_stopping_cb,
                    max_epochs = 20,
                    num_sanity_val_steps = 10
                    )
            trainer.fit(model, train_dl, val_dl)
            model = Model.load_from_checkpoint(early_stopping_cb.best_model_path)
            torch.save(model.state_dict(), f'../model/model_{model_name}_{bi}')

