import numpy as np
import jpegio as jpio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def decompress_YCbCr(p):
    o = jio.read(p)
    col, row = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    img_dims = np.array(o.coef_arrays[0].shape)
    n_blocks = img_dims//8
    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)

    YCbCr = []
    for i, dct_coeffs in enumerate(o.coef_arrays):
        if i==0:
            QM = o.quant_tables[i]
        else:
            QM = o.quant_tables[1]

        t = np.broadcast_to(T.reshape(1, 8, 1, 8), broadcast_dims)
        qm = np.broadcast_to(QM.reshape(1, 8, 1, 8), broadcast_dims)
        dct_coeffs = dct_coeffs.reshape(broadcast_dims)

        a = np.transpose(t, axes = (0, 2, 3, 1))
        b = (QM*dct_coeffs).transpose(0,2,1,3)
        c = t.transpose(0, 2, 1, 3)

        z = a @ b @ c
        z = z.transpose(0, 2, 1, 3)
        YCbCr.append(z.reshape(img_dims))

    return np.stack(YCbCr, -1).astype(np.float32)
