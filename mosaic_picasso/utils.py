import numpy as np
import tifffile, skimage
from scipy import ndimage
import skimage.registration
import skimage.exposure

from datetime import datetime


def tstamp():
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%m%d_h%H_m%M")
    return formatted_timestamp


def save_fiji(img, fname, dimension_order="TZCYX", normalization=False):
    if normalization:
        img_fiji = np.array(img / img.max() * (2**15)).astype(np.uint16)
    else:
        img_fiji = np.array(img).astype(np.uint16)
    tifffile.imwrite(
        fname,
        img_fiji.astype("uint16"),
        shape=img_fiji.shape,
        imagej=True,
        metadata={
            "axes": dimension_order,
        },
    )


def drift_corr(img, display=True):
    nchs = img.shape[-1]
    for i in range(1, nchs):
        shift, _, _ = skimage.registration.phase_cross_correlation(img[:, :, 0], img[:, :, i], upsample_factor=100)
        img[:, :, i] = ndimage.shift(img[:, :, i], shift)
        if display:
            print(f"shift: {shift}, img: {img.shape}")
    return img


def bg_remove(img, bg_percentile=25):  #'YXC" img
    n_ch = img.shape[-1]
    bgs = [np.percentile(img[..., i].squeeze(), bg_percentile) for i in range(n_ch)]
    img_out = img.copy()
    for i in range(n_ch):
        img_out[:, :, i] -= bgs[i]
    return img_out


def gray2rgb_2c(im, color=-1, contrastEnhancement=True):
    nch = im.shape[-1]
    im2 = im.copy()
    for i in range(nch):
        im2[:, :, i] = np.round(im[:, :, i] / im[:, :, i].max() * 255).astype(np.uint8)
    rgb = np.zeros((im2.shape[0], im2.shape[1], 3), dtype=np.uint8)
    if color == -1:
        rgb[:, :, :2] = im2
    if color == 0:
        rgb[:, :, 0] = im2[:, :, 0]
    if color == 1:
        rgb[:, :, 1] = im2[:, :, 1]

    if contrastEnhancement:
        rgb = skimage.exposure.equalize_adapthist(rgb)
    return rgb


from skimage.metrics import structural_similarity as ssim


def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img1.max() - img1.min())


from scipy.stats import pearsonr


def calculate_pearson_correlation(img1, img2):
    return pearsonr(img1.flatten(), img2.flatten())[0]
