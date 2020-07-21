import os
import torch, timm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning.core.lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler, EarlyStopping, ModelCheckpoint
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import auroc

def get_eff(name: str):
    model = getattr(timm.models, name)(pretrained=True)
    stem_weights = model.conv_stem.weight
    model.conv_stem = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
    model.conv_stem.weight = stem_weights
    model.act2 = Mish()
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten())

class Eff_b0(pl.LightningModule):
    def __init__(self, name):
        super().__init__()
        m = get_eff(name)
        self.enc = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
        nc = m.classifier.in_features
        self.emb = nn.Embedding(3, 5)
        self.lin = nn.Sequential(nn.Linear(2*nc, 512), Mish(),
                nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, 4))

    def forward(self, x_img, x_meta):
        x1 = self.enc(x_img)
        x2 = self.emb(x_meta)
        return self.lin(torch.cat([x1, x2], dim=1))

    def training_step(self, batch, batch_idx):
        x_img, x_meta, yb = batch
        y_hat = self(x_img = x_img, x_meta = x_meta)
        loss = loss_fn(y_hat, yb)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x_img, x_meta, yb = batch
        y_hat = self(x_img = x_img, x_meta = x_meta)
        loss = loss_fn(y_hat, yb)
        return {'valid_loss': loss, 'yb': yb, 'predictions': y_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        yb = torch.cat([x['yb'] for x in outputs], 0)
        predictions = torch.cat[x['yb'] for x in outputs], 0)
        score = auroc(predictions, yb)
        return {'val_loss': avg_loss, 'score': score}

    def configure_optimizer(self):
        opt = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = opt.lr_scheduler.OneCycleLR(opt, max_lr= 1e-4, epochs= 5, steps_per_epoch=self.len_train)
        return [opt], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        train_dl = DataLoader(train_ds, pin_memory=True, shuffle=True, batch_size=32, num_workers=4)
        self.len_train = len(train_dl)
        return train_dl

    def val_dataloader(self):
        return DataLoader(valid_ds, pin_memory=True, batch_size=32, num_workers = 4)


# callbacks

def reduce_loss(loss, reduce='mean'):
    return loss.mean() if reduce=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε: float=0.1, reduction='mean'):
        super().__init__()
        self.ε, self.reduction = ε, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return (1-self.ε)*nll + self.ε*(loss/c)
