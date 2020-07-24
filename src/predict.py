import torch
from torch import nn
import timm
from torch.utils.data import Dataset, DataLoader

model_dir = '../models'
models_path = os.listdir(model_dir)

def get_eff(name: str):
    model = getattr(timm.models, name)(pretrained=False)
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
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x_img, x_meta):
        x1 = self.enc(x_img)
        x2 = self.emb(x_meta)
        return self.lin(torch.cat([x1, x2], dim=1))

# loading models and pushing to GPU
models = []
for path in models_path:
    state_dict = torch.load(path, map_location= torch.device('gpu'))
    model = Eff_b0(path.split('_')[1])
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.cuda()
    model.append(model)

class ImageOpen:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

proc_1 = ImageOpen(cv2.imread(), cv2.COLOR_BGR2RGB)

class ImageDataset(Dataset):
    def __init__(self, df, tfms, f_open = proc_1):
        self.image_path = df['path']
        self.qf = df['qf'].values
        self.tfms = tfms
        self.f_open = f_open

    def __getitem__(self, idx):
        img = self.f_open(self.image_path[idx])
        return self.tfms(image = img), self.qf[idx]

    def __len__(self):
        return len(self.image_path)

df_test = pd.read_csv('../data/test.csv')

test_ds = ImageDataset(df_test, tfms)
test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, pin_memory=True)

preds = []
for bi, (x_img, x_meta) in enumerate(tqdm(test_dl)):
    with torch.no_grad():
        x_img, x_meta = x_img.cuda(), x_meta.cuda()
        p = [model(x) for model in models]
        p = torch.stack(p, 1)
        p = p.view(8, 8*len(models), -1).mean(1).cpu()
        preds.append(p)

preds = torch.cat(preds).numpy()
sub_df = pd.DataFrame({'image_id': df_test.iloc[:, 0].values, 'preds': preds})
sub_df.to_csv('submission.csv', index=False)
sub_df.head()
