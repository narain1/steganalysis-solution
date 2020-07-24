import numpy as np
import jpegio as jpio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def JPEGdecompressYCbCr(jpegStruct):
    nb_colors = jpegStruct.coef_arrays.shape[0]
    [col, row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2*col + 1) * row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    sz = np.array(jpegStruct.coef_arrays[0].shape)
    imDecompressYCbCr = np.zeros([sz[0], sz[1], nb_colors])
    szDct = (sz/8).astype('int')

    for ColorChannel in range(nb_colors):
        tmpPixels = np.zeros(sz)
        DCTcoefs = jpegStruct.coef_arrays[ColorChannel]
        if ColorChannel == 0:
            QM = jpegStruct.quant_tables[ColorChannel]
        else:
            QM = jpegStruct.quant_table[1]

        for idxRow in range(szDct[0]):
            for idxCol in range(szDct[1]):
                D = DCTcoeffs[idxRow*8: (idxRow+1)*8, idxCol*8: (idxCol+1)*8]
                tmpPixels[idxRow*8: (idxRow+1)*8, idxCol*8:(idxCol+1)*8] = np.dot(np.transpose(T), np.dot(QM*D, T))
        imDecompressYCbCr[:, :, ColorChannel] = tmpPixels
    return ImDecompressYCbCr


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
