# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_ax_mean(axis, img):
    ax_mean_ls = list()
    img_dim = img.shape
    ax = 0 if axis == 'row' else 1
    idx_ls = [i for i in range(img_dim[ax])]
    for i in idx_ls:
        line = img[i, :] if ax == 0 else img[:, i]
        line_mean = np.mean(line)
        ax_mean_ls.append(line_mean)
    return dict(zip(idx_ls, ax_mean_ls))


def dict_plot(dict_, img, axis):
    # aspect = img.shape[0]/img.shape[1]
    fig, ax = plt.subplots(2, figsize=(12, 9))
    items = sorted(dict_.items())
    x, y = zip(*items)
    if axis == 'row':
        ax[0].barh(y, x)
        ax[0].invert_yaxis()
    else:
        ax[0].bar(x, y)
    ax[1].imshow(img, cmap='gray')
    plt.show()
    
def get_split_id(idx):
    diff = list()
    for n, i in enumerate(idx):
        if n == len(idx) - 1:
            diff.append(idx[n] - idx[n-1])
        else:
            diff.append(idx[n+1] - idx[n])
    id_diff = dict(zip(idx, diff))
    mean_diff = np.mean(diff)
    jump_id = [i for i, diff in id_diff.items() if diff >= mean_diff]
    
    return jump_id
    


