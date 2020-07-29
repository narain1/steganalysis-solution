import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
import numpy as np

def cutmix(batch, alpha):
    x, y= batch
    ids = torch.randperm(x.size(0))
    shuf_x, shuf_y = x[ids], y[ids]
    distr = Beta(tensor([α]), tensor([α]))
    img_h, img_w = x.shape[2:]
    cx, cy = torch.random.uniform(0, img_w), torch.random.uniform(0, img_h)
    w, h = img_w*torch.sqrt(1-distr), img_h*torch.sqrt(1-distr)
    x0, x1 = int(np.round(max(cx - w/2, 0))), int(np.round(min(cx + w/2, img_w)))
    y0, y1 = int(np.round(max(cy - h/2, 0))), int(np.round(min(cy + h/2, img_h)))
    x[:,:, y0:y1, x0:x1] = shuf_x[:,:,y0:y1, x0:x1]
    y = (y, shuf_y, distr)
    return s, y

class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = DataLoader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch

class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction = reduction)

    def __call__(self, preds, targets):
        y1, y2, distr = targets
        return distr* self.criterion(preds, y1)+ (1-distr)*self.criterion(preds, y2)

@torch.no_grad()
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



