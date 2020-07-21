import numpy as np
import pandas as pd
import os
import jpegio
from pandarallel import pandarallel
from pathlib import Path

cur_dir = Path(os.getcwd())
data_path = cur_dir.parent/'data'


dirs = [i for i in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, i)) and i!= 'Test']

train_fs, test_fs = [], []

for k in classes:
    for p, d, fs in os.walk(os.path.join(data_path, k)):
        for f in fs:
            train_fs.append(Path(os.path.join(p, f)))

for p, d, fs in os.walk('../data/Test'):
    for f in fs:
        test_fs.append(os.path.join(p, f))

df_test = pd.DataFrame({'path': test_fs})
df = pd.DataFrame({'path': train_fs})
df['class'] = df['path'].apply(lambda x: x.parent.name)
df['target'] = df['class']!='Cover'

pandarallel.initialize(progress_bar = True)

def calc_qf(p):
    jpegStruct = jpegio.read(str(p))
    if (jpegStruct.quant_tables[0][0,0] == 2):
        return 95
    elif (jpegStruct.quant_tables[0][0,0]==3):
        return 90
    elif (jpegStruct.quant_tables[0][0,0]==8):
        return 75

df['qf'] = df['path'].parallel_apply(lambda x: calc_qf(x))
df_test['qf'] = df_test['path'].parallel_apply(lambda x: calc_qf(x))

df['qf'] = df['qf'].astype(np.int16)
df.to_csv('../data/train.csv', index=False)
df_test.to_csv('../data/test.csv', index=False)
