import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import torchvision.transforms.functional as TF
from joblib import Parallel, delayed
import mosaic_picasso.utils as utils


class MosaicPicasso:
    def __init__(self, bins=256, beta=0.0, gamma=0.1, cycles=40, subunit_sz=40, stride=40, nch=3, threshold=1, mode='ssim'):
        self.bins = bins
        self.beta = beta
        self.gamma = gamma
        self.cycles = cycles
        self.subunit_sz = 40
        self.stride = 40
        self.nch = nch

        # user defined threshold
        self.th = threshold  
        # user defined mode
        self.mode = mode  

    # chop
    def create_chopedImg(self, img):
        chopedImg = []
        coordinates = []
        subunit_sz, stride = self.subunit_sz, self.stride
        for i in range(2):
            tensor_im = TF.to_tensor(img[:, :, i].astype(float))
            patches = tensor_im.unfold(1, subunit_sz, stride).unfold(2, subunit_sz, stride)
            chopedImg.append(patches.reshape(-1, subunit_sz, subunit_sz).numpy())
            _, h, w, _, _ = patches.shape
            coords = [(x * stride, y * stride) for x in range(h) for y in range(w)]
            coordinates.append(coords)
        return np.array(chopedImg), coordinates[0]

    def normalized_mutual_info_score(self, img1, img2):
        hist, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=self.bins)
        hist /= np.sum(hist)  # Normalize the histogram
        entropy1 = entropy(np.sum(hist, axis=1))  # Calculate entropy of hist along axis 1
        entropy2 = entropy(np.sum(hist, axis=0))  # Calculate entropy of hist along axis 0
        joint_entropy = -np.sum(hist * np.log(hist + (hist == 0)))
        return (entropy1 + entropy2 - joint_entropy) / np.maximum(entropy1, entropy2)


    def cal_MI(self, chopedImg, mode='ssim'):
        im1 = chopedImg[0, :, :, :].copy()
        im2 = chopedImg[1, :, :, :].copy()
        # ms, pp, ss = [], [], []
        if mode == 'ssim':
            ss = np.array([utils.calculate_ssim(im1[i], im2[i]) for i in range(im1.shape[0])])
        elif mode == 'pearson':
            ss = np.array([utils.calculate_pearson_correlation(im1[i], im2[i]) for i in range(im1.shape[0])])
        else: # using mutual information
            ss = np.array([self.normalized_mutual_info_score(im1[i], im2[i]) for i in range(im1.shape[0])])
        return ss


    def cal_ij(self, X, i, j):  # mosaic
        # chop
        X2 = X[:, :, [i, j]].copy()  # convert into pair
        X2_chop, _ = self.create_chopedImg(X2)
        ssim = self.cal_MI(X2_chop, mode=self.mode)

        th = np.percentile(ssim, self.th)
        X2_chop_low = X2_chop[:, ssim <= th].copy()
        Y = X2_chop_low.reshape(2, -1, self.subunit_sz).transpose((1, 2, 0))

        obj_func = lambda alpha: self.normalized_mutual_info_score(Y[:, :, 0], Y[:, :, 1] - alpha * Y[:, :, 0])
        bounds = [(-0.01, 0.6)]
        initial_guess = [0.0]
        result = minimize(obj_func, initial_guess, method="Powell", bounds=bounds)
        return result.x

    def compute_P(self, X):
        n_ch = self.nch
        p_matrix = np.eye(n_ch)
        indices = [(i, j) for i in range(n_ch) for j in range(n_ch) if i != j]
        alphas = Parallel(n_jobs=-1)(delayed(self.cal_ij)(X, i, j) for i, j in indices)
        for k, (i, j) in enumerate(indices):
            alpha_ij = alphas[k]
            p_matrix[i, j] = -alpha_ij * self.gamma
        return p_matrix

    def update_P_matrix(self, P0, P1):
        return np.matmul(P1, P0)

    def update(self, P):
        yn = np.einsum("ijm, mk -> ijk", self.img, P)  # new
        return yn

    def mosaic(self, img):  # "YXC" shape
        n_ch = self.nch
        self.img = img.copy()

        P0, P1 = np.eye(n_ch), np.eye(n_ch)
        for cnt in range(self.cycles):
            tmp = self.update(P0)
            Ptmp = self.compute_P(tmp)
            P1 = self.update_P_matrix(P0, Ptmp)
            P0 = P1.copy()
            print(cnt, end=",")

        img_c = self.update(P1)
        img_c[img_c < 0] = 0
        return img_c, P1
